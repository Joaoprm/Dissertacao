import traci
import numpy as np
import random
import timeit
import os
import warnings

warnings.filterwarnings("ignore")
import intersection_manager
import sapa
import tensorflow as tf
from model_MAPPO import MAPPOAgent, MAPPOCritic, ppo_actor_step, critic_step
from memory import RolloutBuffer

class Simulation:
    def __init__(self, traffic_gen, ped_gen, sumo_cmd, save_path, n_agents, num_states, num_actions, max_steps, green_duration, yellow_duration, gamma=0.99, gae_lambda=0.95, actor_lr=0.0001, critic_lr=0.0003, horizon=512, ppo_epochs=6, minibatch_size=512, clip_eps=0.2, ent_coef=0.05, vf_coef=0.5):
        self.traffic_gen = traffic_gen
        self.ped_gen = ped_gen
        self.sumo_cmd = sumo_cmd
        self.save_path = save_path
        self.n_agents = n_agents
        self.num_states = num_states
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.gamma = gamma
        self.lam = gae_lambda
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.horizon = horizon
        self.ppo_epochs = ppo_epochs
        self.minibatch = minibatch_size
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.actor = MAPPOAgent(obs_dim=num_states, agent_id_dim=n_agents, n_actions=num_actions, lr=actor_lr)
        self.critic = MAPPOCritic(global_obs_dim=n_agents * num_states, lr=critic_lr)
        self.buffer = RolloutBuffer(n_agents=n_agents, gamma=gamma, lam=gae_lambda, horizon=horizon)

        self._ep_returns = np.zeros(n_agents, dtype=np.float32)
        self._history_returns = [[] for _ in range(n_agents)]
        self.actor_loss_hist = []
        self.critic_loss_hist = []
        self.IDENTITY = np.eye(n_agents, dtype=np.float32)

        self._Pveh = 0.5
        self._Pped = 0.5

        self.intersections = intersection_manager.create_intersections(self.num_states)
        self.routes = intersection_manager.create_routes()
        self.waiting_ped = intersection_manager.create_waiting_zones()
        self.tl_names = intersection_manager.create_tl_names()
        self.incoming_roads = intersection_manager.create_incoming_routes()
        self.lanes_200_400 = intersection_manager.create_200_400_routes()
        self.map_env = intersection_manager.create_map_environment_()
        self.sapa = sapa.sapa_module()

        # --- NOVAS MÉTRICAS DE AUTOCARROS ---
        self._minute_step = 0
        self.bus_speed_data = {idx: [] for idx in range(n_agents)}
        self.bus_impedance_data = {idx: [] for idx in range(n_agents)}

    def _choose_action_mappo(self, obs, agent_idx):
        obs = obs.astype(np.float32)[None, :]
        agent_onehot = self.IDENTITY[agent_idx][None, :]
        action, logp, _ = self.actor.actor.act(obs, agent_onehot, deterministic=False)
        return int(action[0]), float(logp[0])
    
 # ---- Função de treino MAPPO ----
    def _train_mappo(self):
        """Treina ator e crítico usando os dados do buffer"""
        start_train = timeit.default_timer()

        # Computar GAE e retornos
        last_global = self.buffer._last_global_obs if self.buffer._last_global_obs is not None else \
        self.buffer.global_obs[-1]
        last_v = self.critic.critic(last_global[None, :], training=False).numpy()[0].astype(np.float32)
        self.buffer.set_last_value(last_v)
        data = self.buffer.compute_gae_and_returns()

        obs_and_id = np.concatenate([data["obs"], data["agent_onehot"]], axis=-1).astype(np.float32)
        actions = data["actions"].astype(np.int32)
        old_logp = data["old_logp"].astype(np.float32)
        adv = data["advantages"].astype(np.float32)
        global_obs = data["global_obs"].astype(np.float32)
        critic_ret = data["critic_returns"].astype(np.float32)

        
        for _ in range(self.ppo_epochs):
            # Treinar crítico
            v_loss = critic_step(self.critic.critic, self.critic.opt,
                                 tf.convert_to_tensor(global_obs),
                                 tf.convert_to_tensor(critic_ret),
                                 vf_coef=self.vf_coef)
            self.critic_loss_hist.append(float(v_loss.numpy()))

            # Treinar ator
            N = obs_and_id.shape[0]
            idx = np.arange(N)
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch):
                mb = slice(start, min(start + self.minibatch, N))
                a_loss, _ = ppo_actor_step(
                    self.actor.actor, self.actor.opt,
                    tf.convert_to_tensor(obs_and_id[idx][mb]),
                    tf.convert_to_tensor(actions[idx][mb]),
                    tf.convert_to_tensor(old_logp[idx][mb]),
                    tf.convert_to_tensor(adv[idx][mb]),
                    clip_eps=self.clip_eps,
                    ent_coef=self.ent_coef,
                    action_mask=None
                )
                self.actor_loss_hist.append(float(a_loss.numpy()))

        training_time = round(timeit.default_timer() - start_train, 1)
        return training_time

    def run(self, episode):
        start_time = timeit.default_timer()
        self.traffic_gen.generate_routefile(seed=episode)
        self.ped_gen.generate_ped_routefile(seed=episode)
        self.ent_coef = max(0.005, 0.05 * (1 - episode / 200))
        traci.start(self.sumo_cmd)
        # --- CORREÇÃO PARA A REDE EDITADA ---
        # Forçamos o SUMO a usar o programa '100' que está no ficheiro .add.xml
        # Isso garante que os índices (12, 32, etc.) voltem a existir
        for tl_id in traci.trafficlight.getIDList():
            try:
                traci.trafficlight.setProgram(tl_id, "100")
            except traci.exceptions.TraCIException:
                pass # Ignora se algum semáforo não tiver o programa 100
        # ------------------------------------

        self._step = 0
        while self._step < self.max_steps:
            if self.intersections[0].dur == -1:
                for idx, C in self.intersections.items():
                    current_state = C.get_state(idx, self.waiting_ped[idx], self.routes, self.lanes_200_400, 0)
                    current_total_wait = C.collect_waiting_times(self.incoming_roads[idx])
                    ped_wait = C.pedestrians_WaitingTime(self.waiting_ped[idx])
                    C.action, C.logp = self._choose_action_mappo(current_state, idx)
                    C.old_state, C.old_action = current_state, C.action
                    C.old_total_wait, C.old_ped_wait = current_total_wait, ped_wait
                    C.dur, C.yellow = C.choose_phase(self._step, C.action, C.old_action, self.tl_names[idx], C.yellow, idx, self.routes, self.map_env, self.sapa)

            for idx, C in self.intersections.items():
                if C.dur == 0:
                    if C.yellow == 0:
                        current_state = C.get_state(idx, self.waiting_ped[idx], self.routes, self.lanes_200_400, C.old_action)
                        #current_total_wait = C.collect_waiting_times(self.incoming_roads[idx])
                        #ped_wait = C.pedestrians_WaitingTime(self.waiting_ped[idx])
                        #reward = self._Pveh * (C.old_total_wait - current_total_wait) + self._Pped * (C.old_ped_wait - ped_wait)
                        
                        current_total_wait = C.collect_waiting_times(self.incoming_roads[idx])
                        ped_wait = C.pedestrians_WaitingTime(self.waiting_ped[idx])

                        #autocarros
                        current_bus_wait = self._collect_bus_waiting_time(self.incoming_roads[idx])
                        if not hasattr(C, 'old_bus_wait'): C.old_bus_wait = 0

                        # Penalizamos a espera do autocarro com um multiplicador 
                        bus_priority_factor = 2
                        bus_reward = bus_priority_factor * (C.old_bus_wait - current_bus_wait)

                        # Nova reward somando o impacto do autocarro
                        reward = (self._Pveh * (C.old_total_wait - current_total_wait)) + \
                                (self._Pped * (C.old_ped_wait - ped_wait)) + \
                                bus_reward

                        # Atualiza o valor antigo para o próximo passo
                        C.old_bus_wait = current_bus_wait

                        C.action, C.logp = self._choose_action_mappo(current_state, idx)
                        global_obs = np.concatenate([ag.old_state for ag in self.intersections.values()])
                        self.buffer.add_step(
                            obs_t=np.array([C.old_state for C in self.intersections.values()]),
                            agent_onehot_t=self.IDENTITY,
                            actions_t=np.array([C.old_action for C in self.intersections.values()]),
                            logp_t=np.array([C.logp for C in self.intersections.values()]),
                            rewards_t=np.array([reward for _ in self.intersections.values()]),
                            global_obs_t=global_obs,
                            value_t=self.critic.critic(global_obs[None, :], training=False).numpy()[0],
                            done_t=False
                        )
                        C.old_state, C.old_action = current_state, C.action
                        C.old_total_wait, C.old_ped_wait = current_total_wait, ped_wait
                        if reward < 0: C.sum_neg_reward += reward

                    C.dur, C.yellow = C.choose_phase(self._step, C.action, C.old_action, self.tl_names[idx], C.yellow, idx, self.routes, self.map_env, self.sapa)

                # Coleta de métricas de autocarros (evita crash)
                self._collect_bus_speed_impact(idx, self.incoming_roads[idx], C)

            for C in self.intersections.values():
                if C.dur > 0: C.dur -= 1

            traci.simulationStep()
            self._step += 1
            self._minute_step = (self._minute_step + 1) % 60

        self._save_episode_stats()
        traci.close()
        sim_time = round(timeit.default_timer() - start_time, 1)
        
        last_global = np.concatenate([ag.old_state for ag in self.intersections.values()])
        self.buffer.set_last_value(self.critic.critic(last_global[None, :], training=False).numpy()[0])
        train_time = self._train_mappo()
        self.buffer.reset()
        return sim_time, train_time

    def _collect_bus_speed_impact(self, idx, incoming_roads, C):
        if not hasattr(C, 'bus_speeds_step'): C.bus_speeds_step, C.bus_impedance_step = [], []
        step_speeds, step_impedance, bus_count = [], 0, 0
        for edge in incoming_roads:
            for v_id in traci.edge.getLastStepVehicleIDs(edge):
                if traci.vehicle.getTypeID(v_id) == "bus":
                    bus_count += 1
                    speed = traci.vehicle.getSpeed(v_id)
                    step_speeds.append(speed)
                    leader = traci.vehicle.getLeader(v_id, 30.0)
                    if leader and traci.vehicle.getTypeID(leader[0]) != "bus" and speed < (traci.vehicle.getMaxSpeed(v_id) * 0.8):
                        step_impedance += 1
        
        C.bus_speeds_step.append(np.mean(step_speeds) if step_speeds else 0)
        C.bus_impedance_step.append(step_impedance / bus_count if bus_count > 0 else 0)

        if self._minute_step == 59:
            self.bus_speed_data[idx].append(np.mean(C.bus_speeds_step) if C.bus_speeds_step else 0)
            self.bus_impedance_data[idx].append(np.mean(C.bus_impedance_step) if C.bus_impedance_step else 0)
            C.bus_speeds_step.clear()
            C.bus_impedance_step.clear()

    def _save_episode_stats(self):
        for idx, C in self.intersections.items():
            C.reward_episode.append(C.sum_neg_reward)
            C.sum_neg_reward = 0

    def save_models(self):
        self.actor.save(self.save_path)
        self.critic.save(self.save_path)

    def _collect_bus_waiting_time(self, incoming_roads):
        bus_wait = 0
        for edge in incoming_roads:
            veh_ids = traci.edge.getLastStepVehicleIDs(edge)
            for v_id in veh_ids:
                if traci.vehicle.getTypeID(v_id) == "bus":
                    bus_wait += traci.vehicle.getAccumulatedWaitingTime(v_id)
        return bus_wait

    @property
    def reward_stores(self): return {idx: C.reward_episode for idx, C in self.intersections.items()}
    @property
    def model_loss_actor(self): return self.actor_loss_hist
    @property
    def model_loss_critic(self): return self.critic_loss_hist