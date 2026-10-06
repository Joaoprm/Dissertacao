import os
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from testing_simulation import Simulation
from generator_exact import TrafficGeneratorExact
from ped_generator import PedestrianGenerator
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path
from model_MAPPO import MAPPOAgent
import xml.etree.ElementTree as ET

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

    # --- Execução da Simulação ---
    print('\n----- A iniciar Simulação')
    Sim.run(config['episode_seed'])

    # --- Gravação de Gráficos de Espera (AWT) ---
    for idx, data in Sim.awt_stores.items():
        Visualizer.save_data_and_plot(data, f'AWT_C{idx}', 'Minutos', 'Tempo de Espera (s)')

    # --- PROCESSAMENTO DO IMPACTO DE VELOCIDADE (RADIAL DIREITA) ---
    impact_data = Sim.volume_vs_speed_data 
    
    if impact_data:
        vols, speeds = zip(*impact_data)
        vols = np.array(vols)
        speeds_kmh = np.array(speeds) * 3.6

        # FILTRO: Removemos ruído (volumes baixos ou velocidades de erro)
        # Ignora minutos com menos de 5 carros para focar no impacto do tráfego real
        mask = (vols > 5) & (speeds_kmh > 2) & (speeds_kmh < 60)
        vols_f = vols[mask]
        speeds_f = speeds_kmh[mask]

        plt.figure(figsize=(10, 6))
        

        if len(vols_f) > 5:
            try:
                # Ajuste logarítmico: reflete como a velocidade cai e estabiliza
                z = np.polyfit(np.log(vols_f), speeds_f, 1)
                p = np.poly1d(z)
                
                v_range = np.linspace(vols_f.min(), vols_f.max(), 100)
                plt.plot(v_range, p(np.log(v_range)), "r-", linewidth=3, label="Impact Trend (Log)")
            except Exception as e:
                print(f"Aviso: Não foi possível calcular a curva de tendência: {e}")

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f'Vehicle Volume Impact on Bus Speed (Scenario {config["scenario"]})')
        plt.xlabel('Vehicles per Minute (Radial N7/S8)')
        plt.ylabel('Bus Mean Speed (km/h)')
        
        # Ajuste dinâmico de eixos
        if len(speeds_f) > 0:
            plt.ylim(max(0, speeds_f.min() - 5), speeds_f.max() + 10)
        
        plt.legend()
        
        # Guardar gráfico de impacto
        impact_plot_path = os.path.join(plot_path, f'Bus_Speed_Impact_S{config["scenario"]}.png')
        plt.savefig(impact_plot_path)
        plt.show()
        print(f"-> Gráfico de impacto guardado em: {impact_plot_path}")

    # --- Heatmap de Performance ---
    medias_finais = {idx: np.mean(data) if data else 0 for idx, data in Sim.awt_stores.items()}
    Visualizer.save_heatmap_3x3(medias_finais, config['scenario'], "S8_Frente")

    # --- Backup das Definições ---
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
    
    print(f"\n----- Processo Concluído!")
    print(f"----- Resultados em: {plot_path}")

   

#Acumulador para o relatório completo
report_metrics = {}

# 1. ANÁLISE DE TEMPO DE ESPERA (Máximos e Desvio Padrão)
max_waits = {}
std_waits = {}

for idx, data in Sim.awt_stores.items():
    if data:
        max_waits[idx] = np.max(data)
        std_waits[idx] = np.std(data)
    else:
        max_waits[idx] = 0
        std_waits[idx] = 0

all_awt_values = [val for data in Sim.awt_stores.values() for val in data]
global_avg_wait = np.mean(all_awt_values) if all_awt_values else 0
global_max_wait = np.max([m for m in max_waits.values()]) if max_waits else 0
global_std_wait = np.std(all_awt_values) if all_awt_values else 0

report_metrics['Global Avg Waiting Time (s)'] = global_avg_wait
report_metrics['Global Max Waiting Time (s)'] = global_max_wait
report_metrics['Global Std Waiting Time (Equitability)'] = global_std_wait

# Gráfico de Barras: Comparação Médio vs Máximo por Cruzamento (Fairness)
plt.figure(figsize=(12, 6))
intersections = list(Sim.awt_stores.keys())
x_indices = np.arange(len(intersections))
bar_width = 0.35

plt.bar(x_indices - bar_width/2, [medias_finais[i] for i in intersections], width=bar_width, label='Tempo Médio (s)', color='skyblue')
plt.bar(x_indices + bar_width/2, [max_waits[i] for i in intersections], width=bar_width, label='Tempo Máximo (s)', color='salmon')

plt.xlabel('Cruzamento (C_idx)')
plt.ylabel('Tempo de Espera (s)')
plt.title(f'Média e Picos de Espera por Cruzamento (Cenário {config["scenario"]})')
plt.xticks(x_indices, [f'C{i}' for i in intersections])
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

fairness_plot_path = os.path.join(plot_path, f'Waiting_Time_Fairness_S{config["scenario"]}.png')
plt.savefig(fairness_plot_path)
plt.close()


# 2. VAZÃO (THROUGHPUT), TEMPO DE VIAGEM E PERDA DE TEMPO (TRIPINFO)
print(' -> A extrair Throughput e Perda de Tempo (TripInfo)...')
completed_vehicles = getattr(Sim, 'completed_vehicles', None)
avg_travel_time = getattr(Sim, 'avg_travel_time', None)
avg_time_loss = getattr(Sim, 'avg_time_loss', None)

# Fallback: Tenta ler o ficheiro XML 'tripinfo.xml' se o SUMO o tiver gerado automaticamente
tripinfo_file = 'intersection/tripinfo.out.xml'
if completed_vehicles is None and os.path.exists(tripinfo_file):
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        trips = root.findall('tripinfo')
        completed_vehicles = len(trips)
        if completed_vehicles > 0:
            avg_travel_time = np.mean([float(t.attrib['duration']) for t in trips])
            avg_time_loss = np.mean([float(t.attrib['timeLoss']) for t in trips])
    except Exception as e:
        print(f"   Aviso: Erro ao ler tripinfo.xml: {e}")

report_metrics['Throughput (Veículos Concluídos)'] = completed_vehicles if completed_vehicles is not None else 'N/A'
report_metrics['Average Travel Time (s)'] = avg_travel_time if avg_travel_time is not None else 'N/A'
report_metrics['Average Time Loss (s)'] = avg_time_loss if avg_time_loss is not None else 'N/A'


# 3. SUSTENTABILIDADE E QUALIDADE DE CONDUÇÃO (EMISSÕES E PARAGENS)
print(' -> A calcular Métricas de Sustentabilidade e Paragens...')
total_co2_kg = getattr(Sim, 'total_co2_kg', None)
avg_stops_per_veh = getattr(Sim, 'avg_stops_per_veh', None)

# Fallback: Tenta ler o ficheiro XML 'summary.xml' do SUMO se existir
summary_file = 'intersection/summary.out.xml'
if total_co2_kg is None and os.path.exists(summary_file):
    try:
        tree = ET.parse(summary_file)
        root = tree.getroot()
        steps = root.findall('step')
        if steps:
            # Soma emissões instantâneas em mg e converte para kg
            co2_mg = sum([float(s.attrib.get('co2', 0)) for s in steps if 'co2' in s.attrib])
            total_co2_kg = co2_mg / 1e6
    except Exception as e:
        print(f"   Aviso: Erro ao ler summary.xml: {e}")

report_metrics['Emissões Totais de CO2 (kg)'] = total_co2_kg if total_co2_kg is not None else 'N/A'
report_metrics['Média de Paragens por Veículo'] = avg_stops_per_veh if avg_stops_per_veh is not None else 'N/A'


# 4. ESTABILIDADE DO CONTROLO (TROCAS DE FASE DOS SEMÁFOROS)
phase_switches = getattr(Sim, 'phase_switches', None)
if phase_switches:
    print(' -> A gerar gráfico de Estabilidade dos Semáforos...')
    plt.figure(figsize=(10, 5))
    plt.bar([f'C{k}' for k in phase_switches.keys()], list(phase_switches.values()), color='mediumpurple')
    plt.xlabel('Cruzamento')
    plt.ylabel('Número Total de Trocas de Fase')
    plt.title(f'Estabilidade do Controlo: Trocas de Fase por Cruzamento (S{config["scenario"]})')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    stability_plot_path = os.path.join(plot_path, f'Signal_Stability_S{config["scenario"]}.png')
    plt.savefig(stability_plot_path)
    plt.close()
    
    report_metrics['Total Phase Switches (Rede)'] = sum(phase_switches.values())


# 5. EXPORTAÇÃO DO RELATÓRIO COMPLETO EM TEXTO (SUMMARY REPORT)
summary_txt_path = os.path.join(plot_path, f'metrics_summary_S{config["scenario"]}.txt')
with open(summary_txt_path, 'w', encoding='utf-8') as f:

    f.write("--- 1. MÉTRICA DE ESPERA ---\n")
    f.write(f"Tempo Média de Espera Global: {report_metrics['Global Avg Waiting Time (s)']:.2f} s\n")
    f.write(f"Tempo Máximo de Espera Registado: {report_metrics['Global Max Waiting Time (s)']:.2f} s\n")
    f.write(f"Desvio Padrão da Espera: {report_metrics['Global Std Waiting Time']:.2f}\n\n")
    
    f.write("--- 2. EFICIÊNCIA DE TRÁFEGO E VAZÃO ---\n")
    f.write(f"Vazão / Veículos Concluídos: {report_metrics['Throughput (Veículos Concluídos)']}\n")
    f.write(f"Tempo Médio de Viagem: {report_metrics['Average Travel Time (s)']}\n")
    f.write(f"Perda de Tempo Média (Time Loss): {report_metrics['Average Time Loss (s)']}\n\n")
    
print(f"-> Relatório de métricas globais guardado em: {summary_txt_path}")


# --- Backup das Definições ---
copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

print(f"\n----- Processo Concluído!")
print(f"----- Resultados em: {plot_path}")