import traci
import numpy as np
import timeit
import intersection_manager
import sapa
import os

# --- MÁSCARA DE SEGURANÇA PARA O TRACI ---
_original_setPhase = traci.trafficlight.setPhase
def safe_setPhase(tlsID, phaseIndex):
    if phaseIndex > 32:
        _original_setPhase(tlsID, 32) 
    else:
        _original_setPhase(tlsID, phaseIndex)

traci.trafficlight.setPhase = safe_setPhase
# -----------------------------------------

PHASE_TO_ACTION = {
    0: 1, 2: 2, 4: 3, 6: 4, 8: 5, 10: 6, 12: 7, 14: 8, 32: 9
}

Volume_Edges = ['W_TL0', 'N_TL3', 'S_TL4', 'N_TL2', 'S_TL2', 'N_TL7', 'S_TL8', 'E_TL6']

# --- MAPEAMENTO DOS CORREDORES DE TRÁFEGO ---

RADIAL_PREFIXES = ['TL7_N','N_TL7', 'S_TL8', 'TL5_TL7', 'TL7_TL5', 'TL8_TL5', 'TL5_TL8']
CIRCULAR_PREFIXES = ['W_TL0','TL0_W','TL0_TL','TL_TL2', 'TL5_TL6','TL6_TL5', 'TL6_E', 'E_TL6', 'TL2_TL5', 'TL5_TL2']

class Simulation:
    def __init__(self, Agent, TrafficGen, PedestrianGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions, n_agents):
        self._Actor = Agent
        self._TrafficGen = TrafficGen
        self._PedestrianGen = PedestrianGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._n_agents = n_agents
        
        # Mapeamentos e módulos da rede
        self.intersections = intersection_manager.create_intersections(num_states)
        self.routes = intersection_manager.create_routes()
        self.waiting_ped = intersection_manager.create_waiting_zones()
        self.tl_names = intersection_manager.create_tl_names()
        self.incoming_roads = intersection_manager.create_incoming_routes()
        self.lanes_200_400 = intersection_manager.create_200_400_routes()
        self.map_env = intersection_manager.create_map_environment_()
        self.sapa = sapa.sapa_module()
        
        # Variáveis de monitorização de tráfego
        self._veiculos_unicos = {edge_id: set() for edge_id in Volume_Edges}
        self._volume_por_edge = {edge_id: 0 for edge_id in Volume_Edges}
        self._minute_step = 0
        self._bus_vistos_total = set()
        self.volume_vs_speed_data = [] 
        
        # Novas métricas de velocidade por corredor e autocarros
        self._radial_speed_step = []
        self.radial_speed_min = []
        
        self._circular_speed_step = []
        self.circular_speed_min = []
        
        self.completed_vehicles = 0

        self.bus_start_times = {}           # {bus_id: step_de_entrada}
        self.completed_bus_travel_times = [] # [tempo_viagem_bus_1, tempo_viagem_bus_2, ...]

    def reset_state(self):
         #Limpa os contadores internos para permitir múltiplas execuções consecutivas para executar os diferentes cenários de stress
        self._step = 0
        self._minute_step = 0
        self.completed_vehicles = 0
        self.volume_vs_speed_data = []
        self._bus_vistos_total = set()
        self._veiculos_unicos = {edge_id: set() for edge_id in Volume_Edges}
        self._volume_por_edge = {edge_id: 0 for edge_id in Volume_Edges}
        
        self._radial_speed_step = []
        self.radial_speed_min = []
        self._circular_speed_step = []
        self.circular_speed_min = []
        
        self.bus_start_times = {}           # {bus_id: step_de_entrada}
        self.completed_bus_travel_times = [] # [tempo_viagem_bus_1, tempo_viagem_bus_2, ...]
        
        for C in self.intersections.values():
            C.awt_greenArea = []
            C.waitingVeh = []
            C.bus_wait_step = []
            C.bus_wait_min = []
            C.dur = 0
            C.yellow = 0

    def run(self, episode, stress_config=None):
        #Executa a simulação com suporte a injeção de acidentes
        start_time = timeit.default_timer()
        self.reset_state()
        
        self._TrafficGen.generate_routefile(seed=episode)
        self._PedestrianGen.generate_ped_routefile(seed=episode)
        self.eye = np.eye(self._n_agents, dtype=np.float32)
        
        traci.start(self._sumo_cmd)
        accident_injected = False
        
        # --- DETEÇÃO DINÂMICA DE TODOS OS SEGMENTOS/SUB-VIAS DOS CORREDORES ---
        all_edges = traci.edge.getIDList()
        
        self.radial_edges = [
            e for e in all_edges 
            if any(e.startswith(prefix) for prefix in RADIAL_PREFIXES) and not e.startswith(":")
        ]
        
        self.circular_edges = [
            e for e in all_edges 
            if any(e.startswith(prefix) for prefix in CIRCULAR_PREFIXES) and not e.startswith(":")
        ]
        
        print(f" -> Segmentos detetados na Radial ({len(self.radial_edges)}): {self.radial_edges}")
        print(f" -> Segmentos detetados na Circular ({len(self.circular_edges)}): {self.circular_edges}")
        
        while self._step < self._max_steps:
            try:
                self.completed_vehicles += traci.simulation.getArrivedNumber()

                # 1. Rastrear veículos que saíram/chegaram ao destino no último step
                arrived_vehs = traci.simulation.getArrivedIDList()
                for v in arrived_vehs:
                    if v in self.bus_start_times:
                        travel_time = self._step - self.bus_start_times[v]
                        self.completed_bus_travel_times.append(travel_time)
                        del self.bus_start_times[v]

                # 2. Injeção de Falhas / Stress Tests
                if stress_config and not accident_injected:
                    if stress_config.get('fail_type') == 'accident':
                        step_start = stress_config.get('step_start', 900)
                        
                        if self._step >= step_start:
                            target_lane = stress_config.get('target', 'TL5_TL7.176_1')
                            duration = stress_config.get('duration', 300)
                            
                            try:
                                vehs = traci.lane.getLastStepVehicleIDs(target_lane)
                                if vehs:
                                    target_veh = vehs[0]
                                    veh_edge = traci.vehicle.getRoadID(target_veh)
                                    veh_lane_idx = traci.vehicle.getLaneIndex(target_veh)
                                    veh_pos = traci.vehicle.getLanePosition(target_veh) + 15.0
                                    
                                    traci.vehicle.setStop(
                                        vehID=target_veh,
                                        edgeID=veh_edge,
                                        pos=veh_pos,
                                        laneIndex=veh_lane_idx,
                                        duration=float(duration)
                                    )
                                    traci.vehicle.setColor(target_veh, (0, 255, 0, 255)) # Pinta de preto
                                    print(f"\n [STRESS TEST] Acidente injetado no carro {target_veh} na via {veh_edge} (Lane {veh_lane_idx}) durante {duration}s!")
                                    accident_injected = True
                            except Exception:
                                pass

                # 3. Monitorização de Tráfego e Rastreio de Autocarros
                self._veh_volumes()
                all_vehicles = traci.vehicle.getIDList()
                bus_ids = [v for v in all_vehicles if "bus" in traci.vehicle.getTypeID(v)]
                
                for b in bus_ids: 
                    self._bus_vistos_total.add(b)
                    if b not in self.bus_start_times:
                        self.bus_start_times[b] = self._step  # Regista o segundo exato de entrada

                # Recolhe velocidades separadas por corredor
                self._collect_corridor_speeds()
               
               # 4. Atualização dos Agentes nos Semáforos
                for idx, C in self.intersections.items():
                    if C.dur <= 0:
                        if C.yellow == 0:
                            # --- VERIFICAÇÃO DE ALVO E TIPO DE FALHA ---
                            is_target = False
                            fail_type = stress_config.get('fail_type') if stress_config else None
                            
                            if stress_config:
                                target = stress_config.get('target_agent')
                                step_start = stress_config.get('step_start', 900)
                                duration = stress_config.get('duration', 300)
                                step_end = step_start + duration
                                
                                tl_name = self.tl_names[idx] if idx in self.tl_names else None
                                c_id = getattr(C, 'id', None)
                                
                                if (str(idx) == str(target) or str(tl_name) == str(target) or str(c_id) == str(target)) and (step_start <= self._step < step_end):
                                    is_target = True

                            # --- EXECUÇÃO DOS CENÁRIOS ---
                            if is_target and fail_type == 'agent_disconnect':
                                # 1. Agente Desconectado (Preso numa ação fixa)
                                fallback_act = stress_config.get('fallback_action', 0)
                                C.action = fallback_act
                                print(f" [STRESS TEST] Agente '{target}' OFFLINE no step {self._step}! Ação forçada: {fallback_act}")

                            elif is_target and fail_type == 'fixed_time_fallback':
                                # 2. Modo de Contingência em Tempos Fixos (Relógio tradicional)
                                fixed_cycle = stress_config.get('fixed_split', 16)
                                available_actions = stress_config.get('allowed_actions', [0, 1])
                                action_idx = int((self._step // fixed_cycle) % len(available_actions))
                                C.action = available_actions[action_idx]
                                print(f" [STRESS TEST] Agente '{target}' em TEMPOS FIXOS no step {self._step}! Ação alternada: {C.action}")

                            else:
                                # 3. Funcionamento Normal do Agente de RL
                                current_state = C.get_state(idx, self.waiting_ped[idx], self.routes, self.lanes_200_400, C.old_action)
                                C.action = self._choose_action(current_state, idx)
                        
                        C.dur, C.yellow = C.choose_phase(self._step, C.action, C.old_action, self.tl_names[idx], C.yellow, idx, self.routes, self.map_env, self.sapa)
                        
                        C.old_action = C.action
                        if C.dur != 4:
                            if C.action < len(C.phase_duration):
                                C.phase_duration[C.action] += C.dur
                                C.n_times_active[C.action] += 1


                    self._allactions(C.phase_activated, C.id, self.incoming_roads[idx], self.waiting_ped[idx])
                    if C.dur > 0: 
                        C.dur -= 1
                    
                    if idx == 'J2' or idx == 'C2':
                        print(f"[Step {self._step}] C2 Ação: {C.action} | Fase Ativa no SUMO: {traci.trafficlight.getPhase('J2')}")
                    
                    self._get_queue_length(idx, C.queue_length)
                    self._collect_waiting_times(self.incoming_roads[idx], C.waitingVeh, C.awt_greenArea)
                    self._vehicles_med_Speed(self.incoming_roads[idx], C.avgspeed, C.avgspeed_greenArea)
                    self._collect_bus_metrics(self.incoming_roads[idx], C)
                    self._collect_bus_speed_impact(idx, self.incoming_roads[idx], C, bus_ids)
                    
                    if (self._step > 0 and self._step % 300 == 0): 
                        self._time_extension(C)

                # Avançar o passo de simulação
                traci.simulationStep()
                self._step += 1
                self._minute_step = (self._minute_step + 1) % 60

            except traci.exceptions.FatalTraCIError:
                break

        print('\n' + '='*55)
        print(f" VEÍCULOS CONCLUÍDOS (THROUGHPUT): {self.completed_vehicles}")
        print(f" VELOCIDADE MÉDIA RADIAL DIREITA : {np.mean(self.radial_speed_min) if self.radial_speed_min else 0:.2f} km/h")
        print(f" VELOCIDADE MÉDIA CIRCULAR       : {np.mean(self.circular_speed_min) if self.circular_speed_min else 0:.2f} km/h")
        print(f" TEMPO MÉDIO ESPERA BUS (GLOBAL) : {self.global_bus_avg_wait:.2f} s")
        print(f" TOTAL DE AUTOCARROS VISTOS      : {len(self._bus_vistos_total)}")
        print('='*55 + '\n')

        try: 
            traci.close()
        except: 
            pass
            
        self.export_bus_data(f"impacto_radial_ep{episode}.txt")
        return round(timeit.default_timer() - start_time, 1)

    def _collect_corridor_speeds(self):
        #Regista velocidades médias separadas para a Radial Direita e Circular
        # Radial Direita
        rad_speeds = [traci.edge.getLastStepMeanSpeed(e) for e in self.radial_edges if traci.edge.getLastStepVehicleNumber(e) > 0]
        if rad_speeds:
            self._radial_speed_step.append(np.mean(rad_speeds))
            
        # Circular
        circ_speeds = [traci.edge.getLastStepMeanSpeed(e) for e in self.circular_edges if traci.edge.getLastStepVehicleNumber(e) > 0]
        if circ_speeds:
            self._circular_speed_step.append(np.mean(circ_speeds))
            
        # Agregação por minuto
        if self._minute_step == 59:
            if self._radial_speed_step:
                self.radial_speed_min.append(np.mean(self._radial_speed_step) * 3.6)
                self._radial_speed_step.clear()
            else:
                self.radial_speed_min.append(0.0)

            if self._circular_speed_step:
                self.circular_speed_min.append(np.mean(self._circular_speed_step) * 3.6)
                self._circular_speed_step.clear()
            else:
                self.circular_speed_min.append(0.0)

    def _veh_volumes(self):
        for e_id in Volume_Edges:
            try:
                novos = set(traci.edge.getLastStepVehicleIDs(e_id)).difference(self._veiculos_unicos[e_id])
                self._volume_por_edge[e_id] += len(novos)
                self._veiculos_unicos[e_id].update(novos)
            except: 
                continue

    def _collect_bus_speed_impact(self, idx, incoming_roads, C, bus_ids):
        if self._step > (self._max_steps - 300) or str(idx) not in ['N7', '7']: 
            return
        if not hasattr(C, 'bus_speeds_step'): 
            C.bus_speeds_step = []
        
        radial_speeds = []
        for v_id in bus_ids:
            try:
                if traci.vehicle.getRoadID(v_id) in incoming_roads:
                    radial_speeds.append(traci.vehicle.getSpeed(v_id))
            except: 
                continue

        if radial_speeds: 
            C.bus_speeds_step.append(np.mean(radial_speeds))

        if self._minute_step == 59:
            local = ['N_TL7', 'S_TL8']
            current_vol = sum(self._volume_por_edge.get(e, 0) for e in local)
            if C.bus_speeds_step:
                self.volume_vs_speed_data.append((current_vol, np.mean(C.bus_speeds_step)))
                C.bus_speeds_step.clear()
            for e in Volume_Edges:
                self._volume_por_edge[e] = 0
                self._veiculos_unicos[e].clear()

    def export_bus_data(self, n):
        if not os.path.exists("resultados_excel"): 
            os.makedirs("resultados_excel")
        with open(os.path.join("resultados_excel", n), "w") as f:
            f.write("Volume_Minuto;Velocidade_Media_KMH\n")
            for vol, vel in self.volume_vs_speed_data: 
                f.write(f"{vol};{vel*3.6:.2f}\n")

    def _collect_waiting_times(self, incoming, waitingVeh, awt_area):
        total_wt, valid = 0, 0
        for e in incoming:
            try:
                total_wt += traci.edge.getWaitingTime(e) / (1 + traci.edge.getLastStepHaltingNumber(e))
                valid += 1
            except: 
                continue
        if valid > 0: 
            waitingVeh.append(total_wt / valid)
        if self._minute_step == 59:
            awt_area.append(np.mean(waitingVeh) if waitingVeh else 0)
            waitingVeh.clear()

    def _collect_bus_metrics(self, incoming, C):
        """ Recolhe especificamente o tempo de espera dos autocarros """
        if not hasattr(C, 'bus_wait_step'): 
            C.bus_wait_step = []
        if not hasattr(C, 'bus_wait_min'): 
            C.bus_wait_min = []
            
        waits = []
        for e in incoming:
            try:
                for v in traci.edge.getLastStepVehicleIDs(e):
                    if "bus" in traci.vehicle.getTypeID(v): 
                        waits.append(traci.vehicle.getWaitingTime(v))
            except: 
                continue
                
        if waits:
            C.bus_wait_step.append(np.mean(waits))
            
        if self._minute_step == 59:
            if C.bus_wait_step:
                C.bus_wait_min.append(np.mean(C.bus_wait_step))
                C.bus_wait_step.clear()
            else:
                C.bus_wait_min.append(0.0)

    def _choose_action(self, s, aid):
        obs, oh = s.astype(np.float32)[None, :], self.eye[aid][None, :]
        a, _, _ = self._Actor.actor.act(obs=obs, agent_onehot=oh, deterministic=True)
        return int(a[0])

    def _get_queue_length(self, idx, q):
        try: 
            q.append(sum([traci.edge.getLastStepHaltingNumber(r) for r in self.incoming_roads[idx]]))
        except: 
            q.append(0)

    def _vehicles_med_Speed(self, incoming, avg_s, avg_s_area):
        try: 
            avg_s.append(np.mean([traci.edge.getLastStepMeanSpeed(e) for e in incoming]))
        except: 
            avg_s.append(0)
        if self._minute_step == 59:
            avg_s_area.append(np.mean(avg_s))
            avg_s.clear()

    def _allactions(self, lista, tl, inc, ped):
        try:
            a = PHASE_TO_ACTION.get(traci.trafficlight.getPhase(tl))
            if a: 
                lista.append(a)
        except: 
            pass

    def _time_extension(self, C):
        for p_id in range(9):
            if p_id < len(C.phase_duration) and C.n_times_active[p_id] > 0:
                C.phase_extension_1_hour[p_id+1] = C.phase_duration[p_id] / C.n_times_active[p_id]

    # --- PROPRIEDADES DE ACESSO ÀS MÉTRICAS ---
    @property
    def awt_stores(self): 
        return {idx: C.awt_greenArea for idx, C in self.intersections.items()}
        
    @property
    def bus_awt_stores(self): 
        """ Retorna o tempo de espera dos autocarros por cruzamento """
        return {idx: getattr(C, 'bus_wait_min', []) for idx, C in self.intersections.items()}
        
    @property
    def global_bus_avg_wait(self):
        """ Retorna a média global do tempo de espera de autocarros """
        all_bus_waits = [val for data in self.bus_awt_stores.values() for val in data if val > 0]
        return np.mean(all_bus_waits) if all_bus_waits else 0.0

    @property
    def avg_speed_radial_direita(self):
        return np.mean(self.radial_speed_min) if self.radial_speed_min else 0.0

    @property
    def avg_speed_circular(self):
        return np.mean(self.circular_speed_min) if self.circular_speed_min else 0.0
    
    @property
    def avg_bus_travel_time(self):
        """ Retorna o Tempo Médio de Viagem (Travel Time) dos autocarros concluídos em segundos """
        if self.completed_bus_travel_times:
            return np.mean(self.completed_bus_travel_times)
        return 0.0