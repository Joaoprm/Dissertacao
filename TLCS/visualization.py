import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

class Visualization:
    def __init__(self, path, dpi):
        self._path = path
        self._dpi = dpi

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        if not data: return
        plt.figure(figsize=(20, 11.25))
        plt.rcParams.update({'font.size': 24})
        plt.plot(data)
        plt.ylabel(ylabel); plt.xlabel(xlabel)
        plt.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close()

    def save_data_and_barchart(self, data, filename, xlabel, ylabel):
        if not data: return
        plt.figure(figsize=(20, 11.25))
        plt.bar(range(len(data)), data)
        plt.ylabel(ylabel); plt.xlabel(xlabel)
        plt.savefig(os.path.join(self._path, 'bar_'+filename+'.png'), dpi=self._dpi)
        plt.close()

    def save_heatmap_3x3(self, waiting_dict, car_strat, bus_strat):
        """Gera o Heatmap para o Word"""
        try:
            grid = np.array([
                [waiting_dict.get(0,0), waiting_dict.get(1,0), waiting_dict.get(2,0)],
                [waiting_dict.get(3,0), waiting_dict.get(4,0), waiting_dict.get(5,0)],
                [waiting_dict.get(6,0), waiting_dict.get(7,0), waiting_dict.get(8,0)]
            ])
            plt.figure(figsize=(10, 8))
            sns.heatmap(grid, annot=True, fmt=".1f", cmap="YlOrRd", square=True)
            plt.title(f"Waiting Time (s)\nCarros: {car_strat} | Bus: {bus_strat}")
            
            f_name = f'Heatmap_C{car_strat}_B{bus_strat}.png'
            plt.savefig(os.path.join(self._path, f_name), dpi=self._dpi, bbox_inches='tight')
            plt.close()
            print(f"SUCESSO: Heatmap guardado em {self._path}")
        except Exception as e:
            print(f"ERRO Heatmap: {e}")