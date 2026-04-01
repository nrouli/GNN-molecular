import os
import pandas as pd
import matplotlib.pyplot as plt


class MetricsTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics = {
            'epoch': [],
            'train_mae': [],
            'val_mae': [],
        }
    
    def update(self, epoch, train_mae, val_mae):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_mae'].append(train_mae)
        self.metrics['val_mae'].append(val_mae)
    
    def save(self, csv_dir='results/csv', fig_dir='results/figures'):
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        # save csv
        csv_path = os.path.join(csv_dir, f'{self.model_name}_metrics.csv')
        pd.DataFrame(self.metrics).to_csv(csv_path, index=False)

        # save figure
        fig, _ = self.plot()
        fig_path = os.path.join(fig_dir, f'{self.model_name}_mae.pdf')
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

        print(f'Metrics saved: {csv_path}')
        print(f'Figure saved:  {fig_path}')
    
    @classmethod
    def load(cls, filepath, model_name=None):
        df = pd.read_csv(filepath)
        if model_name is None:
            model_name = os.path.basename(filepath).replace('_metrics.csv', '')
        tracker = cls(model_name)
        tracker.metrics = df.to_dict(orient='list')
        return tracker
    
    def plot(self, ax=None):
        plt.rcParams['lines.linewidth'] = 0.8
        plt.rcParams['font.size'] = 9
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        
        epochs = self.metrics['epoch']
        ax.plot(epochs, self.metrics['train_mae'], label=f'{self.model_name} train')
        ax.plot(epochs, self.metrics['val_mae'], label=f'{self.model_name} val')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        
        return fig, ax