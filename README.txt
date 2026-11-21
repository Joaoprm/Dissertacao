# TLCS — Traffic Light Control System (Dissertação)

> Repositório desenvolvido no âmbito de dissertação de mestrado no ISEL — Instituto Superior de Engenharia de Lisboa.

## Visão Geral
Esta pasta TLCS contém todos os algoritmos responsáveis pelo treino e teste da rede em diferentes cenários com a inserção do BUS sobre o gerador de tráfego. Contém o layout físico da rede modificado para acomodar a infraestrutura BUS e todos os respetivos algoritmos/alterações realizadas para o seu funcionamento.

## Filiação Académica
- Autor: João Matos Github Username:(`Joaoprm`)
- Instituição: ISEL — Instituto Superior de Engenharia de Lisboa (Instituto Politécnico de Lisboa)
- Orientação:Doutora Maria Manuela Almeida Carvalho Vieira 
	     Doutor Mário Pereira Véstias 
	     Doutora Paula Maria Garcia Louro 

# Estrutura do Diretório (`TLCS/`)
- `intersection/` — dados, mapas ou configurações associadas à interseção.
- `intersection.py` — definição e lógica da interseção.
- `intersection_manager.py` — gestão e coordenação da interseção / agentes.
- `model_MAPPO.py` — arquitetura e lógica do modelo MAPPO.
- `memory.py` — armazenamento e gestão de experiências/memória durante o treino.
- `sapa.py` — implementação do mecanismo/algoritmo SAPA.
- `generator.py` e `generator_exact.py` — geração de fluxos/cenários de tráfego (incluindo geração exata).
- `ped_generator.py` — geração de tráfego pedonal.
- `data_processing.py` — pré-processamento e tratamento de dados.
- `phases_analasys.py` — análise de fases semafóricas.
- `training_main.py`, `training_simulation.py` e `training_settings.ini` — arranque, simulação e parâmetros de treino.
- `testing_main.py`, `testing_simulation.py` e `testing_settings.ini` — arranque, simulação e parâmetros de teste/avaliação.
- `main_stress.py` e `simulation_stress.py` — execução e simulação de cenários de stress.
- `graphics_comparation.py` — geração e comparação de gráficos de desempenho.
- `utils.py` — funções auxiliares e utilitários gerais.
 
#Requisitos e Instalação
```bash
# Clonar o repositório
git clone [https://github.com/Joaoprm/Dissertacao.git](https://github.com/Joaoprm/Dissertacao.git)
cd Dissertacao/TLCS

#Execução:
 Após configuração de "training_settings.ini", executar training_main.py para treinar a rede com as configurações específicas.
Para executar a rede recém treinada, configura-se "testing_settings.ini" e executa-se "testing_main.py"
Os resultados surgem sobre a pasta "model_XX" com os resultados do treino prévio e respetivo teste.