from __future__ import absolute_import, print_function
import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
import gc
from shutil import copyfile
from training_simulation import Simulation 
from generator_exact import TrafficGeneratorExact
from ped_generator import PedestrianGenerator
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path

warnings.filterwarnings("ignore")

# GPU mem growth (igual ao teu)
gpus = tf.config.list_physical_devices('GPU')
"""
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurada corretamente!")
    except RuntimeError as e:
        print(e)
else:
    print("Nenhuma GPU encontrada. Usando apenas a CPU.")
"""
if __name__ == "__main__":
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    TrafficGen = TrafficGeneratorExact(config['max_steps'], config['n_cars_generated'], config['scenario'])
    PedestrianGen = PedestrianGenerator(config['max_steps'], config['n_peds_generated'])
    Visualization = Visualization(path, dpi=96)

    sim = Simulation(
        traffic_gen=TrafficGen, ped_gen=PedestrianGen, sumo_cmd=sumo_cmd, save_path=path,
        n_agents=config.get('n_agents', 9), num_states=config['num_states'], num_actions=config['num_actions'],
        max_steps=config['max_steps'], green_duration=config['green_duration'], yellow_duration=config['yellow_duration']
    )


    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        print(f'\n----- Episode {episode+1} of {config["total_episodes"]}')
        sim_time, train_time = sim.run(episode)
        gc.collect()
        tf.keras.backend.clear_session()
        print(f'Sim: {sim_time}s - Train: {train_time}s')
        episode += 1

    sim.save_models()
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    # --- GRAVAÇÃO DE DADOS (IMAGENS + TXT) ---
    print(f"logs em: {path}")
    for idx, rewards in sim.reward_stores.items():
        # Salva TXT explicitly
        np.savetxt(os.path.join(path, f"Reward_C{idx}.txt"), rewards, fmt="%.2f")
        # Salva Imagem
        Visualization.save_data_and_plot(data=rewards, filename=f"Reward_C{idx}", xlabel='Episode', ylabel=f'Cumulative negative reward C{idx}')

    # Salva Perdas e métricas extras
    np.savetxt(os.path.join(path, "Actor_Loss.txt"), sim.actor_loss_hist, fmt="%.6f")
    np.savetxt(os.path.join(path, "Critic_Loss.txt"), sim.critic_loss_hist, fmt="%.6f")

    # Salva Velocidade dos Autocarros (TXT)
    for idx in range(sim.n_agents):
        if sim.bus_speed_data[idx]:
            np.savetxt(os.path.join(path, f"Bus_Speed_C{idx}.txt"), sim.bus_speed_data[idx], fmt="%.2f")

    print(f"Sessão finalizada. Início: {timestamp_start} - Fim: {datetime.datetime.now()}")