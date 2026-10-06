import numpy as np
import math

class TrafficGeneratorExact:
    def __init__(self, max_steps, n_cars_generated, scenario):
        self._n_cars_generated = n_cars_generated  # Total de veículos
        self._max_steps = max_steps
        self._scenario = scenario
        self._bus_total = 60

        self._circular_prob, self._in_out_prob = self.get_probs_by_scenario(self._scenario)

    def get_probs_by_scenario(self, scenario):
        scenario_probs = {
            1: (0.50, 0.50),
            2: (0.45, 0.75),
            3: (0.45, 0.25),
            4: (0.25, 0.75),
            5: (0.25, 0.25)
        }
        if scenario not in scenario_probs:
            raise ValueError(f"Cenário inválido: {scenario}. Deve ser um número entre 1 e 5.")
        return scenario_probs[scenario]

    def generate_routefile(self, seed):
        """
        Geração do ficheiro de rotas para um episódio com paragens de autocarro agendadas
        """
        np.random.seed(seed)

        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps

        for value in timings:
            novo_valor = ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new
            car_gen_steps = np.append(car_gen_steps, novo_valor)

        car_gen_steps = np.rint(car_gen_steps)

        step_interval = self._n_cars_generated // self._bus_total
        bus_indices = set(range(0, self._n_cars_generated, step_interval))
        bus_indices = set(list(bus_indices)[:self._bus_total])

        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
                <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="13.9" depart_speed="13.9" sigma="0.5" />
                <vType id="bus" vClass="bus" accel="0.8" decel="3.5" length="12.0" minGap="3.0" maxSpeed="11.11" depart_speed="11.0" sigma="0.5" color="red" latAlignment="right" minGapLat="1.0" lcStrategic="0.1" lcLookahead="50.0" lcAssertive="0.1" lcSpeedGain="0.0"/>

                <route id="W_E" edges="W_TL0 W_TL0.39 TL0_TL TL0_TL.172 TL0_TL.203 TL0_TL.341 TL_TL2 TL_TL2.138 TL2_TL5 TL2_TL5.337 TL5_TL6 TL5_TL6.141 TL6_E"/>
                <route id="W_S4" edges="W_TL0 W_TL0.39 TL0_TL TL0_TL.172 TL0_TL.203 TL0_TL.341 TL_TL4 TL4_S"/>
                <route id="W_S2" edges="W_TL0 W_TL0.39 TL0_TL TL0_TL.172 TL0_TL.203 TL0_TL.341 TL_TL2 TL_TL2.138 TL2_S"/>
                
                <route id="W_S8" edges="W_TL0 W_TL0.39 TL0_TL TL0_TL.172 TL0_TL.203 TL0_TL.341 TL_TL2 TL_TL2.138 TL2_TL5 TL2_TL5.337 TL5_TL8 TL5_TL8.205 TL5_TL8.238 TL5_TL8.341 TL8_S"/>
                
                <route id="W_W3" edges="W_TL0 W_TL0.39 TL0_TL TL0_TL.172 TL0_TL.203 TL0_TL.341 TL_TL3 TL3_W"/>
                <route id="W_N2" edges="W_TL0 W_TL0.39 TL0_TL TL0_TL.172 TL0_TL.203 TL0_TL.341 TL_TL2 TL_TL2.138 TL2_N"/>
                <route id="W_N6" edges="W_TL0 W_TL0.39 TL0_TL TL0_TL.172 TL0_TL.203 TL0_TL.341 TL_TL2 TL_TL2.138 TL2_TL5 TL2_TL5.337 TL5_TL6 TL5_TL6.141 TL6_N"/>
                <route id="E_W" edges="E_TL6 E_TL6.43 TL6_TL5 TL6_TL5.136 TL5_TL2 TL5_TL2.192 TL5_TL2.222 TL5_TL2.338 TL2_TL TL2_TL.138 TL_TL0 TL_TL0.342 TL0_W"/>
                <route id="E_N7" edges="E_TL6 E_TL6.43 TL6_TL5 TL6_TL5.136 TL5_TL7 TL5_TL7.81 TL5_TL7.113 TL5_TL7.176 TL7_N"/>
                <route id="E_N2" edges="E_TL6 E_TL6.43 TL6_TL5 TL6_TL5.136 TL5_TL2 TL5_TL2.192 TL5_TL2.222 TL5_TL2.338 TL2_N"/>
                <route id="E_N3" edges="E_TL6 E_TL6.43 TL6_TL5 TL6_TL5.136 TL5_TL2 TL5_TL2.192 TL5_TL2.222 TL5_TL2.338 TL2_TL TL2_TL.138 TL_TL3 TL3_N"/>
                
                <route id="E_E8" edges="E_TL6 E_TL6.43 TL6_TL5 TL6_TL5.136 TL5_TL8 TL5_TL8.205 TL5_TL8.238 TL5_TL8.341 TL8_E"/>
                
                <route id="E_S2" edges="E_TL6 E_TL6.43 TL6_TL5 TL6_TL5.136 TL5_TL2 TL5_TL2.192 TL5_TL2.222 TL5_TL2.338 TL2_S"/>
                <route id="E_S0" edges="E_TL6 E_TL6.43 TL6_TL5 TL6_TL5.136 TL5_TL2 TL5_TL2.192 TL5_TL2.222 TL5_TL2.338 TL2_TL TL2_TL.138 TL_TL0 TL_TL0.342 TL0_S"/>
                <route id="N3_S4" edges="N_TL3 TL3_TL TL_TL4 TL4_S"/>
                <route id="N3_W3" edges="N_TL3 TL3_W"/>
                <route id="N3_E4" edges="N_TL3 TL3_TL TL_TL4 TL4_E"/>
                <route id="S4_N3" edges="S_TL4 TL4_TL TL_TL3 TL3_N"/>
                <route id="S4_E3" edges="S_TL4 TL4_TL TL_TL3 TL3_E"/>
                <route id="S4_W0" edges="S_TL4 TL4_TL TL_TL0 TL_TL0.342 TL0_W"/>
                
                <route id="N7_E6" edges="N_TL7 N_TL7.40 TL7_TL5 TL7_TL5.140 TL5_TL6 TL6_E"/>
                
                <route id="N7_E8" edges="N_TL7 N_TL7.40 TL7_TL5 TL7_TL5.140 TL5_TL8 TL5_TL8.205 TL5_TL8.238 TL5_TL8.341 TL8_E"/>
                <route id="N7_S8" edges="N_TL7 N_TL7.40 TL7_TL5 TL7_TL5.140 TL5_TL8 TL5_TL8.205 TL5_TL8.238 TL5_TL8.341 TL8_S"/>
                
                <route id="S8_N7" edges="S_TL8 S_TL8.70 TL8_TL5 TL8_TL5.375 TL5_TL7 TL5_TL7.81 TL5_TL7.113 TL5_TL7.176 TL7_N"/>
                <route id="S8_E7" edges="S_TL8 S_TL8.70 TL8_TL5 TL8_TL5.375 TL5_TL7 TL5_TL7.81 TL5_TL7.113 TL5_TL7.176 TL7_E"/>
                
                <route id="S8_N2" edges="S_TL8 S_TL8.70 TL8_TL5 TL8_TL5.375 TL5_TL2 TL5_TL2.192 TL5_TL2.222 TL5_TL2.338 TL2_N"/>
                <route id="N0_S0" edges="N_TL0 TL0_S"/>
                <route id="S0_N0" edges="S_TL0 TL0_N"/>
                <route id="N6_S6" edges="N_TL6 TL6_S"/>
                <route id="S6_N6" edges="S_TL6 TL6_N"/>
                <route id="N2_S2" edges="N_TL2 TL2_S"/>
                <route id="S2_N2" edges="S_TL2 TL2_N"/>
                <route id="W3_E3" edges="W_TL3 TL3_E"/>
                <route id="E3_W3" edges="E_TL3 TL3_W"/>
                <route id="W4_E4" edges="W_TL4 TL4_E"/>
                <route id="E4_W4" edges="E_TL4 TL4_W"/>
                <route id="W7_E7" edges="W_TL7 TL7_E"/>
                <route id="E7_W7" edges="E_TL7 TL7_W"/>
                <route id="W8_E8" edges="W_TL8 TL8_E"/>
                <route id="E8_W8" edges="E_TL8 TL8_W"/>
                <route id="N0_L" edges="N_TL0 TL0_TL TL0_TL.172 TL0_TL.203 TL0_TL.341 TL_TL3 TL3_W"/>
                <route id="W0_L" edges="W_TL0 W_TL0.39 TL0_N"/>
                <route id="S0_L" edges="S_TL0 TL0_W"/>
                
                <route id="S6_L" edges="S_TL6 TL6_TL5 TL6_TL5.136 TL5_TL8 TL5_TL8.205 TL5_TL8.238 TL5_TL8.341 TL8_E"/>
                
                <route id="N6_L" edges="N_TL6 TL6_E"/>
                <route id="E6_L" edges="E_TL6 E_TL6.43  TL6_S"/>
                <route id="W3_L" edges="W_TL3 TL3_N"/>
                <route id="E3_L" edges="E_TL3 TL3_TL TL_TL2 TL_TL2.138 TL2_N"/>
                <route id="N3_L" edges="N_TL3 TL3_E"/>
                <route id="S4_L" edges="S_TL4 TL4_W"/>
                <route id="E4_L" edges="E_TL4 TL4_S"/>
                <route id="W4_L" edges="W_TL4 TL4_TL TL_TL0 TL_TL0.342 TL0_S"/>
                <route id="S8_L" edges="S_TL8 S_TL8.70 TL8_W"/>
                <route id="E8_L" edges="E_TL8 TL8_S"/>
                <route id="W8_L" edges="W_TL8 TL8_TL5 TL8_TL5.375 TL5_TL2 TL5_TL2.192 TL5_TL2.222 TL5_TL2.338 TL2_S"/>
                <route id="S2_L" edges="S_TL2 TL2_TL TL2_TL.138 TL_TL4 TL4_E"/>
                <route id="N2_L" edges="N_TL2 TL2_TL5 TL2_TL5.337 TL5_TL7 TL5_TL7.81 TL5_TL7.113 TL5_TL7.176 TL7_W"/>
            """ , file=routes)
            
            # Contador para alternar os sentidos
            bus_count = 0
            for car_counter, step in enumerate(car_gen_steps):
                if car_counter in bus_indices:
                        # Decide aleatoriamente se este autocarro vai parar (60% de probabilidade)
                        should_stop = np.random.uniform() < 0.6
                        
                    # A lógica de rotação permanece nos 4 sentidos (bus_count % 4)
                        direction = bus_count % 4

                        if direction == 0:
                            # SENTIDO SUL-NORTE (S8 -> N7)
                            if should_stop:
                                print('    <vehicle id="bus_stop_SN_%i" type="bus" route="S8_N7" depart="%s" departLane="1">' % (car_counter, step), file=routes)
                                print('        <stop busStop="paragem_reentrancia" duration="20"/>', file=routes)
                                # Lane de suporte para a paragem: deve ser a mesma lane definida no busStop
                                print('        <stop lane="TL5_TL7.81_1" startPos="2.0" endPos="18.0" duration="0"/>', file=routes)
                                print('    </vehicle>', file=routes)
                            else:
                                print('    <vehicle id="bus_express_SN_%i" type="bus" route="S8_N7" depart="%s" departLane="1" laneChangeMode="0" color="blue"/>' % (car_counter, step), file=routes)

                        elif direction == 1:
                            # SENTIDO NORTE-SUL (N7 -> S8)
                            if should_stop:
                                print('    <vehicle id="bus_stop_NS_%i" type="bus" route="N7_S8" depart="%s" departLane="1">' % (car_counter, step), file=routes)
                                print('        <stop busStop="paragem_reentrancia_NS" duration="20"/>', file=routes)
                                print('        <stop lane="TL5_TL8.205_1" startPos="2.0" endPos="18.0" duration="0"/>', file=routes)
                                print('    </vehicle>', file=routes)
                            else:
                                print('    <vehicle id="bus_express_NS_%i" type="bus" route="N7_S8" depart="%s" departLane="1" laneChangeMode="0" color="blue"/>' % (car_counter, step), file=routes)

                        elif direction == 2:
                            # SENTIDO OESTE-ESTE (W -> E) - Usa a paragem paragem_reentrancia_OE
                            if should_stop:
                                print('    <vehicle id="bus_stop_EO_%i" type="bus" route="W_E" depart="%s" departLane="1">' % (car_counter, step), file=routes)
                                print('        <stop busStop="paragem_reentrancia_OE" duration="20"/>', file=routes)
                                print('        <stop lane="TL0_TL.172_1" startPos="2.0" endPos="18.0" duration="0"/>', file=routes)
                                print('    </vehicle>', file=routes)
                            else:
                                print('    <vehicle id="bus_express_EO_%i" type="bus" route="W_E" depart="%s" departLane="1" laneChangeMode="0" color="blue"/>' % (car_counter, step), file=routes)

                        else: # direction == 3
                            # SENTIDO ESTE-OESTE (E -> W) - Usa a paragem paragem_reentrancia_EO
                            if should_stop:
                                print('    <vehicle id="bus_stop_OE_%i" type="bus" route="E_W" depart="%s" departLane="1">' % (car_counter, step), file=routes)
                                print('        <stop busStop="paragem_reentrancia_EO" duration="20"/>', file=routes)
                                print('        <stop lane="TL5_TL2.192_1" startPos="2.0" endPos="18.0" duration="0"/>', file=routes)
                                print('    </vehicle>', file=routes)
                            else:
                                print('    <vehicle id="bus_express_OE_%i" type="bus" route="E_W" depart="%s" departLane="1" laneChangeMode="0" color="blue"/>' % (car_counter, step), file=routes)

                        bus_count += 1
                else:
                    # APLICAÇÃO DA LÓGICA DE PERCENTAGENS PARA OS CARROS REGULARES
                    straight_or_turn = np.random.uniform()
                    
                    if straight_or_turn < 0.75:  # 75% das vezes o carro segue o fluxo estruturado
                        arterial_routes = np.random.uniform()
                        
                        if arterial_routes < self._circular_prob:
                            west_east_routes = np.random.uniform()
                            if west_east_routes < 0.50:
                                route_straight = np.random.randint(1, 8)
                                if route_straight == 1:
                                    print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 2:
                                    print('    <vehicle id="W_S4_%i" type="standard_car" route="W_S4" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 3:
                                    print('    <vehicle id="W_S2_%i" type="standard_car" route="W_S2" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 4:
                                    print('    <vehicle id="W_N2_%i" type="standard_car" route="W_N2" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 5:
                                    print('    <vehicle id="W_N6_%i" type="standard_car" route="W_N6" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="W_S8_%i" type="standard_car" route="W_S8" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                            else:
                                route_straight = np.random.randint(1, 8)
                                if route_straight == 1:
                                    print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 2:
                                    print('    <vehicle id="E_N7_%i" type="standard_car" route="E_N7" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 3:
                                    print('    <vehicle id="E_N2_%i" type="standard_car" route="E_N2" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 4:
                                    print('    <vehicle id="E_E8_%i" type="standard_car" route="E_E8" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 5:
                                    print('    <vehicle id="E_S2_%i" type="standard_car" route="E_S2" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                elif route_straight == 6:
                                    print('    <vehicle id="E_S0_%i" type="standard_car" route="E_S0" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="E_N3_%i" type="standard_car" route="E_N3" depart="%s" departLane="best" />' % (car_counter, step), file=routes)

                        else:  # Artéria radial
                            cell = np.random.uniform()
                            if cell < 0.80: # Divisão de tráfego para ambas as radiais
                                in_out_priority = np.random.uniform()
                                if in_out_priority < self._in_out_prob:
                                    route_straight = np.random.randint(1, 4)
                                    if route_straight == 1:
                                        print('    <vehicle id="S4_N3_%i" type="standard_car" route="S4_N3" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                    elif route_straight == 2:
                                        print('    <vehicle id="S4_E3_%i" type="standard_car" route="S4_E3" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                    elif route_straight == 3:
                                        print('    <vehicle id="S4_W0_%i" type="standard_car" route="S4_W0" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                else:
                                    route_straight = np.random.randint(1, 4)
                                    if route_straight == 1:
                                        print('    <vehicle id="N3_S4_%i" type="standard_car" route="N3_S4" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                    elif route_straight == 2:
                                        print('    <vehicle id="N3_E4_%i" type="standard_car" route="N3_E4" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                    elif route_straight == 3:
                                        print('    <vehicle id="N3_W3_%i" type="standard_car" route="N3_W3" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                            else:
                                in_out_priority = np.random.uniform()
                                if in_out_priority < self._in_out_prob:
                                    route_straight = np.random.randint(1, 3)
                                    if route_straight == 1:
                                        print('    <vehicle id="S8_N7_%i" type="standard_car" route="S8_N7" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                    elif route_straight == 2:
                                        print('    <vehicle id="S8_E7_%i" type="standard_car" route="S8_E7" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                else:
                                    route_straight = np.random.randint(1, 3)
                                    if route_straight == 1:
                                        print('    <vehicle id="N7_S8_%i" type="standard_car" route="N7_S8" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                                    elif route_straight == 2:
                                        print('    <vehicle id="N7_E8_%i" type="standard_car" route="N7_E8" depart="%s" departLane="best" />' % (car_counter, step), file=routes)

                    else:  # 25% restantes para escolher uma rota de viragem completamente aleatória (route_turn)
                        route_turn = np.random.randint(1, 35)
                        if route_turn == 1:
                            print('    <vehicle id="N0_S0_%i" type="standard_car" route="N0_S0" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                        elif route_turn == 2:
                            print('    <vehicle id="S0_N0_%i" type="standard_car" route="S0_N0" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 3:
                            print('    <vehicle id="N6_S6_%i" type="standard_car" route="N6_S6" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 4:
                            print('    <vehicle id="S6_N6_%i" type="standard_car" route="S6_N6" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 5:
                            print('    <vehicle id="N2_S2_%i" type="standard_car" route="N2_S2" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 6:
                            print('    <vehicle id="S2_N2_%i" type="standard_car" route="S2_N2" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 7:
                            print('    <vehicle id="W3_E3_%i" type="standard_car" route="W3_E3" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 8:
                            print('    <vehicle id="E3_W3_%i" type="standard_car" route="E3_W3" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 9:
                            print('    <vehicle id="W4_E4_%i" type="standard_car" route="W4_E4" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 10:
                            print('    <vehicle id="E4_W4_%i" type="standard_car" route="E4_W4" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 11:
                            print('    <vehicle id="W7_E7_%i" type="standard_car" route="W7_E7" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 12:
                            print('    <vehicle id="E7_W7_%i" type="standard_car" route="E7_W7" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 13:
                            print('    <vehicle id="W8_E8_%i" type="standard_car" route="W8_E8" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 14:
                            print('    <vehicle id="E8_W8_%i" type="standard_car" route="E8_W8" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 15:
                            print('    <vehicle id="N0_L_%i" type="standard_car" route="N0_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 16:
                            print('    <vehicle id="S0_L_%i" type="standard_car" route="S0_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 17:
                            print('    <vehicle id="W0_L_%i" type="standard_car" route="W0_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 18:
                            print('    <vehicle id="S6_L_%i" type="standard_car" route="S6_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 19:
                            print('    <vehicle id="N6_L_%i" type="standard_car" route="N6_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 20:
                            print('    <vehicle id="E6_L_%i" type="standard_car" route="E6_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 21:
                            print('    <vehicle id="W3_L_%i" type="standard_car" route="W3_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 22:
                            print('    <vehicle id="E3_L_%i" type="standard_car" route="E3_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 23:
                            print('    <vehicle id="N3_L_%i" type="standard_car" route="N3_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 24:
                            print('    <vehicle id="S4_L_%i" type="standard_car" route="S4_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 25:
                            print('    <vehicle id="E4_L_%i" type="standard_car" route="E4_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 26:
                            print('    <vehicle id="W4_L_%i" type="standard_car" route="W4_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 27:
                            print('    <vehicle id="S8_L_%i" type="standard_car" route="S8_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 28:
                            print('    <vehicle id="E8_L_%i" type="standard_car" route="E8_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 29:
                            print('    <vehicle id="W8_L_%i" type="standard_car" route="W8_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 30:
                            print('    <vehicle id="S2_L_%i" type="standard_car" route="S2_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)
                        elif route_turn == 31:
                            print('    <vehicle id="N3_E4_%i" type="standard_car" route="N3_E4" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                        elif route_turn == 32:
                            print('    <vehicle id="S8_N2_%i" type="standard_car" route="S8_N2" depart="%s" departLane="best" />' % (car_counter, step), file=routes)
                        else:
                            print('    <vehicle id="N2_L_%i" type="standard_car" route="N2_L" depart="%s" departLane="best"/>' % (car_counter, step), file=routes)

            # Escreve a tag final quando o loop de veículos acabar
            print("</routes>", file=routes)