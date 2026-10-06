# model.py — Actor partilhado + Crítico centralizado 
import os
import numpy as np
import tensorflow as tf
from typing import Optional
from tensorflow import keras
from tensorflow.keras import layers

# ---------- utils ----------
def apply_action_mask(logits: tf.Tensor, mask: Optional[tf.Tensor]):
    """Coloca -1e9 nos logits de ações inválidas (mask=0)."""
    if mask is None:
        return logits
    mask = tf.cast(mask, tf.float32)
    huge_neg = tf.constant(-1e9, dtype=logits.dtype)
    any_valid = tf.reduce_any(mask > 0.0, axis=-1, keepdims=True)
    masked = tf.where(tf.equal(mask, 1.0), logits, huge_neg)
    return tf.where(any_valid, masked, logits)

def log_prob_from_logits(logits: tf.Tensor, actions: tf.Tensor):
    """log π(a|s) a partir de logits e ações inteiras."""
    actions = tf.cast(actions, tf.int32)
    logp_all = tf.nn.log_softmax(logits, axis=-1)
    idx = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
    return tf.gather_nd(logp_all, idx)


def categorical_entropy_from_logits(logits: tf.Tensor):
    """Entropia de uma distribuição categórica definida por logits."""
    probs = tf.nn.softmax(logits, axis=-1)
    logp = tf.nn.log_softmax(logits, axis=-1)
    return -tf.reduce_sum(probs * logp, axis=-1)

# ---------- Actor (partilhado) ----------
class Actor(keras.Model):
    """
    Input = [obs_local, one_hot(agent_id)]  ->  Output = logits (n_actions)
    """
    def __init__(self, obs_dim: int, agent_id_dim: int, n_actions: int,
                 width: int = 128, layers_n: int = 2, activation: str = "relu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.agent_id_dim = agent_id_dim
        self.n_actions = n_actions

        inputs = layers.Input((obs_dim + agent_id_dim,))
        x = inputs
        x = layers.Dense(width, activation=activation)(x)
        for _ in range(max(0, layers_n - 1)):
            x = layers.Dense(width, activation=activation)(x)
        logits = layers.Dense(n_actions)(x)
        self.net = keras.Model(inputs, logits, name="actor")

    def call(self, obs_and_id, training: bool = False):
        return self.net(obs_and_id, training=training)

    def act(self, obs, agent_onehot, action_mask=None, deterministic=False):
        """obs:[B,obs_dim], agent_onehot:[B,agent_id_dim], mask:[B,n_actions]"""
        x = np.concatenate([obs, agent_onehot], axis=-1).astype(np.float32)
        logits = self(x, training=False)

        if action_mask is not None:
            logits = apply_action_mask(logits, tf.convert_to_tensor(action_mask, dtype=tf.float32))

        if deterministic:
            actions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            # amostra em int32 para evitar mismatch
            actions = tf.random.categorical(logits, 1, dtype=tf.int32)
            actions = tf.squeeze(actions, axis=-1)  # [B]

        logp = log_prob_from_logits(logits, actions)  # já lida com int32
        return actions.numpy(), logp.numpy(), logits.numpy()

# ---------- Crítico centralizado ----------
class CentralCritic(keras.Model):
    def __init__(self, global_input_dim: int,
                 width: int = 256, layers_n: int = 2, activation: str = "relu"):
        super().__init__()
        inputs = layers.Input((global_input_dim,))
        x = inputs
        x = layers.Dense(width, activation=activation)(x)
        for _ in range(max(0, layers_n - 1)):
            x = layers.Dense(width, activation=activation)(x)
        v = layers.Dense(1)(x)
        self.net = keras.Model(inputs, v, name="critic")

    def call(self, global_obs, training: bool = False):
        v = self.net(global_obs, training=training)
        return tf.squeeze(v, axis=-1)  # [B]

# ---------- Wrappers simples ----------
class MAPPOAgent:
    def __init__(self, obs_dim: int, agent_id_dim: int, n_actions: int, lr: float = 3e-4):
        self.actor = Actor(obs_dim, agent_id_dim, n_actions)
        self.opt = keras.optimizers.Adam(lr)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # força construção
        dummy = np.zeros((1, self.actor.obs_dim + self.actor.agent_id_dim), np.float32)
        _ = self.actor(dummy, training=False)
        # TF 2.10: usar .h5 (ou SavedModel)
        self.actor.net.save(os.path.join(path, "actor.h5"))

    def load(self, path: str):
        self.actor.net = keras.models.load_model(os.path.join(path, "actor.h5"))

class MAPPOCritic:
    def __init__(self, global_obs_dim: int, lr: float = 3e-4):
        self.critic = CentralCritic(global_obs_dim)
        self.opt = keras.optimizers.Adam(lr)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        dummy = np.zeros((1, self.critic.net.input_shape[-1]), np.float32)
        _ = self.critic(dummy, training=False)
        self.critic.net.save(os.path.join(path, "critic.h5"))

    def load(self, path: str):
        self.critic.net = keras.models.load_model(os.path.join(path, "critic.h5"))

# ---------- passos de treino ----------
@tf.function
def ppo_actor_step(actor: Actor, opt, obs_and_id, actions, old_logp, advantages,
                   clip_eps=0.2, ent_coef=1e-2, action_mask=None, max_grad_norm=0.5):
    # assegura dtypes coerentes
    actions   = tf.cast(actions, tf.int32)
    old_logp  = tf.cast(old_logp, tf.float32)
    advantages = tf.cast(advantages, tf.float32)

    with tf.GradientTape() as tape:
        logits = actor(obs_and_id, training=True)
        if action_mask is not None:
            logits = apply_action_mask(logits, tf.cast(action_mask, tf.float32))
        new_logp = log_prob_from_logits(logits, actions)
        entropy = tf.reduce_mean(categorical_entropy_from_logits(logits))

        ratio = tf.exp(new_logp - old_logp)
        adv = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        pg1 = -adv * ratio
        pg2 = -adv * tf.clip_by_value(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        loss = tf.reduce_mean(tf.maximum(pg1, pg2)) - ent_coef * entropy

    grads = tape.gradient(loss, actor.trainable_variables)
    if max_grad_norm:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    opt.apply_gradients(zip(grads, actor.trainable_variables))
    return loss, entropy


@tf.function
def critic_step(critic: CentralCritic, opt,
                global_obs: tf.Tensor,     # [T, global_obs_dim]
                returns: tf.Tensor,        # [T]
                vf_coef: float = 0.5,
                max_grad_norm: float = 0.5):
    with tf.GradientTape() as tape:
        v = critic(global_obs, training=True)  # [T]
        v_loss = tf.reduce_mean(tf.square(returns - v)) * vf_coef
    grads = tape.gradient(v_loss, critic.trainable_variables)
    if max_grad_norm:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    opt.apply_gradients(zip(grads, critic.trainable_variables))
    return v_loss
