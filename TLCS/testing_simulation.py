import traci
import numpy as np
import timeit
import intersection_manager
import sapa
import os

# --- MÁSCARA DE SEGURANÇA PARA O TRACI ---
# Esta função substitui a original do TraCI para impedir o erro da fase 33
_original_setPhase = traci.trafficlight.setPhase
def safe_setPhase(tlsID, phaseIndex):
    if phaseIndex > 32:
        # Se o código tentar fase 33 ou superior, força para 32 ou 0
        #print(f" Bloqueada fase inválida {phaseIndex} no semáforo {tlsID}")
        _original_setPhase(tlsID, 32) 
    else:
        _original_setPhase(tlsID, phaseIndex)
# Sobrescrevemos a função globalmente
traci.trafficlight.setPhase = safe_setPhase
# -----------------------------------------

PHASE_TO_ACTION = {
    0: 1, 2: 2, 4: 3, 6: 4, 8: 5, 10: 6, 12: 7, 14: 8, 32: 9
}

Volume_Edges = ['W_TL0', 'N_TL3', 'S_TL4', 'N_TL2', 'S_TL2', 'N_TL7', 'S_TL8', 'E_TL6']

class Simulation:
    def __init__(self, Agent, TrafficGen, PedestrianGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions, n_agents):
        self._Actor = Agent
        self._TrafficGen = TrafficGen
        self._PedestrianGen = PedestrianGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._n_agents = n_agents
        self.intersections = intersection_manager.create_intersections(num_states)
        self.routes = intersection_manager.create_routes()
        self.waiting_ped = intersection_manager.create_waiting_zones()
        self.tl_names = intersection_manager.create_tl_names()
        self.incoming_roads = intersection_manager.create_incoming_routes()
        self.lanes_200_400 = intersection_manager.create_200_400_routes()
        self.map_env = intersection_manager.create_map_environment_()
        self.sapa = sapa.sapa_module()
        
        self._veiculos_unicos = {edge_id: set() for edge_id in Volume_Edges}
        self._volume_por_edge = {edge_id: 0 for edge_id in Volume_Edges}
        self._minute_step = 0
        self._bus_vistos_total = set()
        self.volume_vs_speed_data = [] 

    def run(self, episode):
        
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        self._PedestrianGen.generate_ped_routefile(seed=episode)
        self.eye = np.eye(self._n_agents, dtype=np.float32)
        
        traci.start(self._sumo_cmd)
        
        while self._step < self._max_steps:
            try:
                self._veh_volumes()
                all_vehicles = traci.vehicle.getIDList()
                bus_ids = [v for v in all_vehicles if "bus" in traci.vehicle.getTypeID(v)]
                for b in bus_ids: self._bus_vistos_total.add(b)
                
                for idx, C in self.intersections.items():
                    if C.dur <= 0:
                        if C.yellow == 0:
                            current_state = C.get_state(idx, self.waiting_ped[idx], self.routes, self.lanes_200_400, C.old_action)
                            C.action = self._choose_action(current_state, idx)
                        
                        # A proteção agora está ativa dentro desta chamada via Monkey Patch
                        C.dur, C.yellow = C.choose_phase(self._step, C.action, C.old_action, self.tl_names[idx], C.yellow, idx, self.routes, self.map_env, self.sapa)
                        
                        C.old_action = C.action
                        if C.dur != 4:
                            if C.action < len(C.phase_duration):
                                C.phase_duration[C.action] += C.dur
                                C.n_times_active[C.action] += 1


                    self._allactions(C.phase_activated, C.id, self.incoming_roads[idx], self.waiting_ped[idx])
                    if C.dur > 0: C.dur -= 1
                    
                    self._get_queue_length(idx, C.queue_length)
                    self._collect_waiting_times(self.incoming_roads[idx], C.waitingVeh, C.awt_greenArea)
                    self._vehicles_med_Speed(self.incoming_roads[idx], C.avgspeed, C.avgspeed_greenArea)
                    self._collect_bus_metrics(self.incoming_roads[idx], C)
                    self._collect_bus_speed_impact(idx, self.incoming_roads[idx], C, bus_ids)
                    
                    if (self._step > 0 and self._step % 300 == 0): self._time_extension(C)

                traci.simulationStep()
                self._step += 1
                self._minute_step = (self._minute_step + 1) % 60

            except traci.exceptions.FatalTraCIError:
                break

        print('\n' + '='*45)
        if self.volume_vs_speed_data:
            velocidades_radial = [vel for vol, vel in self.volume_vs_speed_data]
            print(f" VELOCIDADE MÉDIA NA RADIAL: {np.mean(velocidades_radial) * 3.6:.2f} km/h")
            print(f" TOTAL DE AUTOCARROS VISTOS: {len(self._bus_vistos_total)}")
        print('='*45 + '\n')

        try: traci.close()
        except: pass
        self.export_bus_data(f"impacto_radial_ep{episode}.txt")
        return round(timeit.default_timer() - start_time, 1)

    def _veh_volumes(self):
        for e_id in Volume_Edges:
            try:
                novos = set(traci.edge.getLastStepVehicleIDs(e_id)).difference(self._veiculos_unicos[e_id])
                self._volume_por_edge[e_id] += len(novos)
                self._veiculos_unicos[e_id].update(novos)
            except: continue

    def _collect_bus_speed_impact(self, idx, incoming_roads, C, bus_ids):
        if self._step > (self._max_steps - 300) or str(idx) not in ['N7', '7']: return
        if not hasattr(C, 'bus_speeds_step'): C.bus_speeds_step = []
        
        radial_speeds = []
        for v_id in bus_ids:
            try:
                if traci.vehicle.getRoadID(v_id) in incoming_roads:
                    radial_speeds.append(traci.vehicle.getSpeed(v_id))
            except: continue

        if radial_speeds: C.bus_speeds_step.append(np.mean(radial_speeds))

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
        if not os.path.exists("resultados_excel"): os.makedirs("resultados_excel")
        with open(os.path.join("resultados_excel", n), "w") as f:
            f.write("Volume_Minuto;Velocidade_Media_KMH\n")
            for vol, vel in self.volume_vs_speed_data: f.write(f"{vol};{vel*3.6:.2f}\n")

    def _collect_waiting_times(self, incoming, waitingVeh, awt_area):
        total_wt, valid = 0, 0
        for e in incoming:
            try:
                total_wt += traci.edge.getWaitingTime(e) / (1 + traci.edge.getLastStepHaltingNumber(e))
                valid += 1
            except: continue
        if valid > 0: waitingVeh.append(total_wt / valid)
        if self._minute_step == 59:
            awt_area.append(np.mean(waitingVeh) if waitingVeh else 0)
            waitingVeh.clear()

    def _collect_bus_metrics(self, incoming, C):
        if not hasattr(C, 'bus_wait_step'): C.bus_wait_step = []
        if not hasattr(C, 'bus_wait_min'): C.bus_wait_min = []
        waits = []
        for e in incoming:
            try:
                for v in traci.edge.getLastStepVehicleIDs(e):
                    if "bus" in traci.vehicle.getTypeID(v): waits.append(traci.vehicle.getWaitingTime(v))
            except: continue
        C.bus_wait_step.append(np.mean(waits) if waits else 0)
        if self._minute_step == 59:
            C.bus_wait_min.append(np.mean(C.bus_wait_step) if C.bus_wait_step else 0)
            C.bus_wait_step.clear()

    def _choose_action(self, s, aid):
        obs, oh = s.astype(np.float32)[None, :], self.eye[aid][None, :]
        a, _, _ = self._Actor.actor.act(obs=obs, agent_onehot=oh, deterministic=True)
        return int(a[0])

    def _get_queue_length(self, idx, q):
        try: q.append(sum([traci.edge.getLastStepHaltingNumber(r) for r in self.incoming_roads[idx]]))
        except: q.append(0)

    def _vehicles_med_Speed(self, incoming, avg_s, avg_s_area):
        try: avg_s.append(np.mean([traci.edge.getLastStepMeanSpeed(e) for e in incoming]))
        except: avg_s.append(0)
        if self._minute_step == 59:
            avg_s_area.append(np.mean(avg_s))
            avg_s.clear()

    def _allactions(self, lista, tl, inc, ped):
        try:
            a = PHASE_TO_ACTION.get(traci.trafficlight.getPhase(tl))
            if a: lista.append(a)
        except: pass

    def _time_extension(self, C):
        for p_id in range(9):
            if p_id < len(C.phase_duration) and C.n_times_active[p_id] > 0:
                C.phase_extension_1_hour[p_id+1] = C.phase_duration[p_id] / C.n_times_active[p_id]

    @property
    def awt_stores(self): return {idx: C.awt_greenArea for idx, C in self.intersections.items()}
    @property
    def bus_speed_impact_stores(self): return self.volume_vs_speed_data