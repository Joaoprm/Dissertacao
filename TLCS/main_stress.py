import os
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import xml.etree.ElementTree as ET

from simulation_stress import Simulation
from generator_exact import TrafficGeneratorExact
from ped_generator import PedestrianGenerator
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path
from model_MAPPO import MAPPOAgent

if __name__ == "__main__":
    # --- Configurações Iniciais ---
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'], config['episode_seed'])

    # --- Inicialização de Componentes ---
    Actor = MAPPOAgent(obs_dim=config['num_states'], agent_id_dim=config['num_agents'], n_actions=config['num_actions'], lr=0.0)
    Actor.load(model_path)

    TrafficGen = TrafficGeneratorExact(config['max_steps'], config['n_cars_generated'], config['scenario'])
    PedestrianGen = PedestrianGenerator(config['max_steps'], config['n_peds_generated'])
    Visualizer = Visualization(plot_path, dpi=96)

    Sim = Simulation(
        Actor, TrafficGen, PedestrianGen, sumo_cmd, 
        config['max_steps'], config['green_duration'], config['yellow_duration'], 
        config['num_states'], config['num_actions'], config['num_agents']
    )

    # --- DEFINIÇÃO DOS CENÁRIOS DE STRESS TEST ---
    stress_scenarios = [
        {
            "id": "baseline",
            "name": "1. Operação Normal (Baseline)",
            "stress_config": None
        },
        {
            "id": "acidente_tl5_tl7",
            "name": "2. Acidente em Via Crítica (TL5_TL7)",
            "stress_config": {
                "fail_type": "accident",
                "tar": "TL5_TL7.176_1",
                "step_start": 900,
                "duration": 300
            }
        },
        {
            "id": "desconexao_agente_j5", #Bloqueio da interseção no caso do agente se desconectar
            "name": "3. Desconexão do Agente (J5)",
            "stress_config": {
                "fail_type": "agent_disconnect",
                "target_agent": "J5",
                "step_start": 900,
                "duration": 600,
                "fallback_action": 8
            }
        },
        {
            "id": "tempos_fixos_contingencia_j5",
            "name": "4. Modo de Tempos Fixos (J5)",
            "stress_config": {
                "fail_type": "fixed_time_fallback",
                "target_agent": "J5",
                "step_start": 900,
                "duration": 600,
                "fixed_split": 16,          # Alterna a cada 16 segundos
                "allowed_actions": [0, 1]   # Alterna entre as duas ações/vias principais
            }
        }
        
    ]

    summary_table_rows = []

    # ==============================================================================
    # --- LOOP PRINCIPAL DE EXECUÇÃO DOS CENÁRIOS ---
    # ==============================================================================
    for scen in stress_scenarios:
        scen_id = scen["id"]
        scen_name = scen["name"]
        
        print(f"\n=====================================================")
        print(f"  A INICIAR: {scen_name}")
        print(f"=====================================================")

        execution_time = Sim.run(config['episode_seed'], stress_config=scen["stress_config"])

        # --- Gravação de Gráficos de Espera Global vs Autocarro ---
        for idx, data in Sim.awt_stores.items():
            Visualizer.save_data_and_plot(data, f'AWT_C{idx}_{scen_id}', 'Minutos', 'Tempo de Espera Global (s)')
            
        for idx, data in Sim.bus_awt_stores.items():
            if data:
                Visualizer.save_data_and_plot(data, f'BUS_AWT_C{idx}_{scen_id}', 'Minutos', 'Tempo de Espera Bus (s)')

        # --- Gráfico do Impacto de Velocidade na Radial Direita ---
        impact_data = Sim.volume_vs_speed_data 
        if impact_data:
            vols, speeds = zip(*impact_data)
            vols = np.array(vols)
            speeds_kmh = np.array(speeds) * 3.6

            mask = (vols > 5) & (speeds_kmh > 2) & (speeds_kmh < 60)
            vols_f = vols[mask]
            speeds_f = speeds_kmh[mask]

            plt.figure(figsize=(10, 6))
            if len(vols_f) > 5:
                try:
                    z = np.polyfit(np.log(vols_f), speeds_f, 1)
                    p = np.poly1d(z)
                    v_range = np.linspace(vols_f.min(), vols_f.max(), 100)
                    plt.plot(v_range, p(np.log(v_range)), "r-", linewidth=3, label="Impact Trend (Log)")
                except Exception as e:
                    print(f"Aviso: Não foi possível calcular a curva de tendência: {e}")

            plt.grid(True, linestyle='--', alpha=0.6)
            plt.title(f'Vehicle Volume Impact on Bus Speed ({scen_name})')
            plt.xlabel('Vehicles per Minute (Radial N7/S8)')
            plt.ylabel('Bus Mean Speed (km/h)')
            
            if len(speeds_f) > 0:
                plt.ylim(max(0, speeds_f.min() - 5), speeds_f.max() + 10)
            
            plt.legend()
            impact_plot_path = os.path.join(plot_path, f'Bus_Speed_Impact_{scen_id}_S{config["scenario"]}.png')
            plt.savefig(impact_plot_path)
            plt.close()

        # Heatmap Global
        medias_finais = {idx: np.mean(data) if data else 0 for idx, data in Sim.awt_stores.items()}
        Visualizer.save_heatmap_3x3(medias_finais, config['scenario'], f"S8_Frente_{scen_id}")

        # ==============================================================================
        # --- CÁLCULO DE MÉTRICAS AVANÇADAS DO CENÁRIO ---
        # ==============================================================================
        report_metrics = {}

        # 1. Espera Geral e Espera dos Autocarros
        all_awt_values = [val for data in Sim.awt_stores.values() for val in data]
        global_avg_wait = np.mean(all_awt_values) if all_awt_values else 0
        global_max_wait = np.max([np.max(d) for d in Sim.awt_stores.values() if d]) if any(Sim.awt_stores.values()) else 0

        bus_avg_wait = Sim.global_bus_avg_wait
        radial_speed = Sim.avg_speed_radial_direita
        circular_speed = Sim.avg_speed_circular

        report_metrics['Global Avg Waiting Time (s)'] = global_avg_wait
        report_metrics['Global Max Waiting Time (s)'] = global_max_wait
        report_metrics['Bus Avg Waiting Time (s)'] = bus_avg_wait
        report_metrics['Velocidade Média Radial Direita (km/h)'] = radial_speed
        report_metrics['Velocidade Média Circular (km/h)'] = circular_speed

        # 2. Throughput, Tempo de Viagem e Time Loss
        completed_vehicles = getattr(Sim, 'completed_vehicles', None)
        avg_travel_time = 'N/A'
        avg_time_loss = 'N/A'

        tripinfo_file = 'intersection/tripinfo.out.xml'
        if os.path.exists(tripinfo_file):
            try:
                tree = ET.parse(tripinfo_file)
                root = tree.getroot()
                trips = root.findall('tripinfo')
                if trips:
                    completed_vehicles = len(trips)
                    avg_travel_time = round(np.mean([float(t.attrib['duration']) for t in trips]), 2)
                    avg_time_loss = round(np.mean([float(t.attrib['timeLoss']) for t in trips]), 2)
            except Exception as e:
                print(f"   Aviso: Erro ao ler tripinfo.xml: {e}")

        report_metrics['Throughput (Veículos Concluídos)'] = completed_vehicles if completed_vehicles is not None else 'N/A'
        report_metrics['Average Travel Time (s)'] = avg_travel_time
        report_metrics['Average Time Loss (s)'] = avg_time_loss

        # 3. Exportação do Relatório em Texto por Cenário
        summary_txt_path = os.path.join(plot_path, f'metrics_summary_{scen_id}_S{config["scenario"]}.txt')
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write("=====================================================\n")
            f.write(f"     RELATÓRIO DE DESEMPENHO - {scen_name.upper()}\n")
            f.write(f"     Cenário: {config['scenario']} | Semente: {config['episode_seed']}\n")
            f.write("=====================================================\n\n")
            f.write("--- 1. MÉTRICAS DE ESPERA ---\n")
            f.write(f"Tempo Médio de Espera Global: {report_metrics['Global Avg Waiting Time (s)']:.2f} s\n")
            f.write(f"Tempo Máximo de Espera Registado: {report_metrics['Global Max Waiting Time (s)']:.2f} s\n")
            f.write(f"Tempo Médio de Espera dos AUTOCARROS: {report_metrics['Bus Avg Waiting Time (s)']:.2f} s\n\n")
            f.write("--- 2. VELOCIDADES POR CORREDOR ---\n")
            f.write(f"Velocidade Média Radial Direita: {report_metrics['Velocidade Média Radial Direita (km/h)']:.2f} km/h\n")
            f.write(f"Velocidade Média Circular: {report_metrics['Velocidade Média Circular (km/h)']:.2f} km/h\n\n")
            f.write("--- 3. EFICIÊNCIA DE TRÁFEGO E VAZÃO ---\n")
            f.write(f"Vazão / Veículos Concluídos: {report_metrics['Throughput (Veículos Concluídos)']}\n")
            f.write(f"Tempo Médio de Viagem: {report_metrics['Average Travel Time (s)']}\n")
            f.write(f"Perda de Tempo Média (Time Loss): {report_metrics['Average Time Loss (s)']}\n\n")

        print(f" -> Relatório do cenário guardado em: {summary_txt_path}")

        # Guardar dados para a Tabela Comparativa Final
        summary_table_rows.append({
            "Cenário": scen_name,
            "AWT Global (s)": round(global_avg_wait, 2),
            "AWT Bus (s)": round(bus_avg_wait, 2),
            "Vel. Radial (km/h)": round(radial_speed, 2),
            "Vel. Circular (km/h)": round(circular_speed, 2),
            "Throughput": completed_vehicles
        })

    # ==============================================================================
    # --- RELATÓRIO COMPARATIVO FINAL ---
    # ==============================================================================
    print('\n' + '='*105)
    print("      TABELA COMPARATIVA DE STRESS TEST, VELOCIDADES E AUTOCARROS")
    print('='*105)
    
    final_report_path = os.path.join(plot_path, f'STRESS_TEST_COMPARISON_S{config["scenario"]}.txt')
    with open(final_report_path, 'w', encoding='utf-8') as f:
        f.write("===========================================================================================================\n")
        f.write(f"              TABELA COMPARATIVA COMPLETA DE RESILIÊNCIA E CORREDORES (CENÁRIO {config['scenario']})\n")
        f.write("===========================================================================================================\n\n")
        
        header = f"{'Cenário de Teste':<35} | {'AWT Global':<10} | {'AWT Bus':<9} | {'Vel. Radial':<11} | {'Vel. Circ.':<10} | {'Throughput':<10}\n"
        f.write(header)
        print(header)
        f.write("-" * 107 + "\n")
        print("-" * 107)
        
        for row in summary_table_rows:
            line = f"{row['Cenário']:<35} | {row['AWT Global (s)']:<10.2f} | {row['AWT Bus (s)']:<9.2f} | {row['Vel. Radial (km/h)']:<11.2f} | {row['Vel. Circular (km/h)']:<10.2f} | {str(row['Throughput']):<10}"
            print(line)
            f.write(line + "\n")
            
        f.write("\n" + "="*107 + "\n")

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    print(f"\n----- Processo Concluído com Sucesso!")
    print(f"----- Relatório comparativo guardado em: {final_report_path}")