import configparser
from sumolib import checkBinary
import os
import sys

def import_train_configuration(config_file):
    """
    Lê o INI de treino. Retrocompatível com a tua versão antiga.
    Se existir a secção [ppo] e/ou n_agents em [agent], são usados.
    Caso contrário, caímos em defaults adequados ao MAPPO.
    """
    content = configparser.ConfigParser()
    content.read(config_file)

    cfg = {}

    # -------- [simulation] (igual ao teu) --------
    sim = content['simulation']
    cfg['gui']              = sim.getboolean('gui')
    cfg['total_episodes']   = sim.getint('total_episodes')
    cfg['max_steps']        = sim.getint('max_steps')
    cfg['n_cars_generated'] = sim.getint('n_cars_generated')
    cfg['n_peds_generated'] = sim.getint('n_peds_generated')
    cfg['green_duration']   = sim.getint('green_duration')
    # mantemos leitura do yellow_duration por compatibilidade (não é usado no treino MAPPO básico)
    cfg['yellow_duration']  = sim.getint('yellow_duration', fallback=4)
    cfg['scenario']         = sim.getint('scenario')

    # -------- [agent] --------
    agent = content['agent']
    cfg['num_states'] = agent.getint('num_states')
    cfg['num_actions'] = agent.getint('num_actions')
    # novo: n_agents (por omissão 5 para J0..J4)
    cfg['n_agents'] = agent.getint('n_agents', fallback=5)

    # -------- [ppo] (nova) com fallbacks se não existir --------
    if content.has_section('ppo'):
        ppo = content['ppo']
        cfg['gamma']         = ppo.getfloat('gamma',       fallback=0.99)
        cfg['gae_lambda']    = ppo.getfloat('gae_lambda',  fallback=0.95)
        cfg['actor_lr']      = ppo.getfloat('actor_lr',    fallback=3e-4)
        cfg['critic_lr']     = ppo.getfloat('critic_lr',   fallback=3e-4)
        cfg['horizon']       = ppo.getint('horizon',       fallback=4096)
        cfg['ppo_epochs']    = ppo.getint('ppo_epochs',    fallback=4)
        cfg['minibatch_size']= ppo.getint('minibatch_size',fallback=4096)
        cfg['clip_eps']      = ppo.getfloat('clip_eps',    fallback=0.2)
        cfg['ent_coef']      = ppo.getfloat('ent_coef',    fallback=0.01)
        cfg['vf_coef']       = ppo.getfloat('vf_coef',     fallback=0.5)
    else:
        # compat: se não existir [ppo], usa defaults MAPPO
        cfg['gamma']          = agent.getfloat('gamma', fallback=0.99)
        cfg['gae_lambda']     = 0.95
        cfg['actor_lr']       = 3e-4
        cfg['critic_lr']      = 3e-4
        cfg['horizon']        = 4096
        cfg['ppo_epochs']     = 4
        cfg['minibatch_size'] = 4096
        cfg['clip_eps']       = 0.2
        cfg['ent_coef']       = 0.01
        cfg['vf_coef']        = 0.5

    # -------- [dir] (igual ao teu) --------
    d = content['dir']
    cfg['models_path_name'] = d['models_path_name']
    cfg['sumocfg_file_name']= d['sumocfg_file_name']

    # chaves antigas que deixaram de ser usadas (mantidas para não partir nada)
    # se existirem no ini, são lidas; caso contrário, ignoradas com fallback
    cfg['num_layers']      = content.getint('model', 'num_layers', fallback=2)
    cfg['width_layers']    = content.getint('model', 'width_layers', fallback=128)
    cfg['batch_size']      = content.getint('model', 'batch_size', fallback=64)
    cfg['learning_rate']   = content.getfloat('model', 'learning_rate', fallback=3e-4)
    cfg['training_epochs'] = content.getint('model', 'training_epochs', fallback=1)

    cfg['memory_size_min'] = content.getint('memory', 'memory_size_min', fallback=1)
    cfg['memory_size_max'] = content.getint('memory', 'memory_size_max', fallback=1)

    # chave antiga do teu trabalho DQN (não usada em MAPPO)
    cfg['pneigh'] = sim.getfloat('pneigh', fallback=0.0)

    return cfg


def import_test_configuration(config_file):
    """
    Sem mudanças — mantido como no teu ficheiro.
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['n_peds_generated'] = content['simulation'].getint('n_peds_generated')
    config['episode_seed'] = content['simulation'].getint('episode_seed')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['scenario'] = content['simulation'].getint('scenario')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['num_agents'] = content['agent'].getint('num_agents')
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['models_path_name'] = content['dir']['models_path_name']
    config['model_to_test'] = content['dir'].getint('model_to_test')
    return config


def set_sumo(gui, sumocfg_file_name, max_steps):
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    sumoBinary = checkBinary('sumo-gui' if gui else 'sumo')
    sumo_cmd = [
        sumoBinary,
        "-c", os.path.join('intersection', sumocfg_file_name),
        "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps),
        '--tripinfo-output', 'intersection/tripinfo.out.xml'
    ]
    return sumo_cmd


def set_train_path(models_path_name):
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(models_path_name, model_n, episode_seed):
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_'+str(model_n), '')
    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test_'+str(episode_seed), '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit('The model number specified does not exist in the models folder')
