# memory.py — Rollout buffer on-policy para MAPPO (discreto)
import numpy as np

class RolloutBuffer:
    """
    Guarda um conjunto de passos on-policy para N agentes (política partilhada)
    e calcula GAE/returns no final do rollout.

    Notas:
      - O crítico é centralizado: recebe global_obs e estima V(s).
      - As vantagens são por agente, usando V(s) comum (centralizado).
      - Para o crítico, podes treinar com a média dos returns dos agentes
        em cada passo (target mais estável) ou, se preferires, com outro
        esquema — aqui exponho ambos os valores no output.
    """

    def __init__(self, n_agents, gamma=0.99, lam=0.95, horizon=2048):
        self.n_agents = n_agents
        self.gamma = gamma
        self.lam = lam
        self.horizon = horizon
        self.reset()

    # --------------------------------------------------------------------- #
    # API de escrita
    # --------------------------------------------------------------------- #
    def reset(self):
        # Por passo (t): listas de arrays shape (n_agents, dim)
        self.obs = []               # [T][n_agents, obs_dim]
        self.agent_onehot = []      # [T][n_agents, agent_id_dim]
        self.actions = []           # [T][n_agents]
        self.logp = []              # [T][n_agents]
        self.rewards = []           # [T][n_agents]
        self.action_mask = []       # [T][n_agents, n_actions] (opcional; pode ser None)

        # Global (partilhado por todos)
        self.global_obs = []        # [T][global_obs_dim]
        self.values = []            # [T] V(s_t) do crítico centralizado

        # flags
        self.dones = []             # [T] done do episódio (boolean)

        # último valor para bootstrapping quando calculamos GAE
        self._last_value = None
        self._last_global_obs = None

    def add_step(self,
                 obs_t,             # np.array (n_agents, obs_dim)
                 agent_onehot_t,    # np.array (n_agents, agent_id_dim)
                 actions_t,         # np.array (n_agents,) int
                 logp_t,            # np.array (n_agents,) float
                 rewards_t,         # np.array (n_agents,) float
                 global_obs_t,      # np.array (global_obs_dim,)
                 value_t,           # float (V(s_t))
                 done_t,            # bool
                 action_mask_t=None # np.array (n_agents, n_actions) ou None
                 ):
        """Adiciona um passo do rollout (todos os agentes nesse passo)."""

        # Verificações básicas
        assert obs_t.shape[0] == self.n_agents
        assert agent_onehot_t.shape[0] == self.n_agents
        assert actions_t.shape[0] == self.n_agents
        assert logp_t.shape[0] == self.n_agents
        assert rewards_t.shape[0] == self.n_agents

        self.obs.append(obs_t.astype(np.float32))
        self.agent_onehot.append(agent_onehot_t.astype(np.float32))
        self.actions.append(actions_t.astype(np.int32))
        self.logp.append(logp_t.astype(np.float32))
        self.rewards.append(rewards_t.astype(np.float32))

        if action_mask_t is not None:
            self.action_mask.append(action_mask_t.astype(np.float32))
        else:
            self.action_mask.append(None)

        self.global_obs.append(global_obs_t.astype(np.float32))
        self.values.append(float(value_t))
        self.dones.append(bool(done_t))

        # guarda último global_obs para conveniência (opcional)
        self._last_global_obs = global_obs_t.astype(np.float32)

    def set_last_value(self, last_value):
        """
        Define V(s_{T}) para bootstrapping no cálculo de GAE/returns.
        Deve ser chamado após o último passo do rollout (com o estado seguinte).
        """
        self._last_value = float(last_value)

    # --------------------------------------------------------------------- #
    # Cálculo de GAE/returns e preparação de lotes
    # --------------------------------------------------------------------- #
    def compute_gae_and_returns(self):
        """
        Devolve um dicionário com tensores prontos para treino:

        - obs:           [T*n_agents, obs_dim]
        - agent_onehot:  [T*n_agents, agent_id_dim]
        - actions:       [T*n_agents]
        - old_logp:      [T*n_agents]
        - advantages:    [T*n_agents]
        - returns:       [T*n_agents]
        - action_mask:   [T*n_agents, n_actions]  (se foi fornecida; caso
                          contrário, None)
        - global_obs:    [T, global_obs_dim]      (para o crítico)
        - critic_returns:[T]                      (target do crítico — média
                          dos returns dos agentes em cada passo)
        """
        T = len(self.values)
        assert T > 0, "Buffer vazio; adiciona passos antes de calcular GAE."
        assert self._last_value is not None, (
            "Chama set_last_value(V(s_T)) antes de compute_gae_and_returns()."
        )

        # Empilha por tempo
        obs = np.stack(self.obs, axis=0)                     # [T, n_agents, obs_dim]
        agent_oh = np.stack(self.agent_onehot, axis=0)       # [T, n_agents, id_dim]
        actions = np.stack(self.actions, axis=0)             # [T, n_agents]
        logp = np.stack(self.logp, axis=0)                   # [T, n_agents]
        rewards = np.stack(self.rewards, axis=0)             # [T, n_agents]
        global_obs = np.stack(self.global_obs, axis=0)       # [T, global_obs_dim]
        values = np.array(self.values, dtype=np.float32)     # [T]
        dones = np.array(self.dones, dtype=np.bool_)         # [T]

        # action masks: lista possivelmente com None
        if any(m is None for m in self.action_mask):
            action_mask = None
        else:
            action_mask = np.stack(self.action_mask, axis=0) # [T, n_agents, n_actions]

        # GAE por agente (usa V(s) centralizado igual para todos)
        advantages = np.zeros((T, self.n_agents), dtype=np.float32)
        returns = np.zeros((T, self.n_agents), dtype=np.float32)

        next_value = self._last_value
        gae_next = np.zeros((self.n_agents,), dtype=np.float32)

        for t in reversed(range(T)):
            done = dones[t]
            # se o episódio terminou em t, não bootstrappamos para o t+1
            nonterminal = 0.0 if done else 1.0
            v = values[t]
            # TD residual por agente (recompensas individuais)
            delta = rewards[t] + self.gamma * next_value * nonterminal - v
            gae = delta + self.gamma * self.lam * nonterminal * gae_next

            advantages[t] = gae
            returns[t] = gae + v

            # prepara para o passo anterior
            gae_next = gae
            next_value = v

        # Targets para o crítico: média dos returns dos agentes por passo (mais estável)
        critic_returns = returns.mean(axis=1)  # [T]

        # Achatar para [T*n_agents, ...] (actor)
        Tn = T * self.n_agents
        obs_flat = obs.reshape(Tn, -1)
        agent_oh_flat = agent_oh.reshape(Tn, -1)
        actions_flat = actions.reshape(Tn)
        logp_flat = logp.reshape(Tn)
        adv_flat = advantages.reshape(Tn)
        ret_flat = returns.reshape(Tn)

        # Normalização das vantagens (boa prática no PPO)
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std() + 1e-8
        adv_flat = (adv_flat - adv_mean) / adv_std

        # Flatten da máscara de ações se existir
        if action_mask is not None:
            action_mask_flat = action_mask.reshape(Tn, -1)
        else:
            action_mask_flat = None

        return {
            "obs": obs_flat,                       # para o ator (concat com one-hot fora se preferires)
            "agent_onehot": agent_oh_flat,
            "actions": actions_flat,
            "old_logp": logp_flat,
            "advantages": adv_flat,
            "returns": ret_flat,                   # returns por agente (podes não usar no ator)
            "action_mask": action_mask_flat,
            "global_obs": global_obs,              # para o crítico
            "critic_returns": critic_returns,      # target V(s)
        }

    # --------------------------------------------------------------------- #
    # Minibatches (opcional)
    # --------------------------------------------------------------------- #
    def iter_minibatches(self, data_dict, batch_size, shuffle=True):
        """
        Iterador de minibatches para os tensores do ator (flattened).
        data_dict deve ser o output de compute_gae_and_returns().
        """
        N = data_dict["obs"].shape[0]
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, N, batch_size):
            end = start + batch_size
            mb_idx = idx[start:end]
            yield {
                "obs": data_dict["obs"][mb_idx],
                "agent_onehot": data_dict["agent_onehot"][mb_idx],
                "actions": data_dict["actions"][mb_idx],
                "old_logp": data_dict["old_logp"][mb_idx],
                "advantages": data_dict["advantages"][mb_idx],
                "returns": data_dict["returns"][mb_idx],
                "action_mask": (
                    None if data_dict["action_mask"] is None
                    else data_dict["action_mask"][mb_idx]
                ),
            }
