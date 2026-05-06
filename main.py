import torch, time
from Simulator import MonteCarloSimulator
from Model import DynamicNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Vizualization import vizualization

def my_network(init_type):
        return DynamicNN(input_dim=30,init_type=init_type, hidden_dim=[32,32,32,16],act_type='relu')


if __name__ == "__main__":
    """ main orchesters all parts together"""

    begin_time = time.perf_counter()

    #X, y = make_classification(n_samples=1000, n_features=300)
    X, y = make_classification(n_samples=10000, n_features=30)
    scaler = StandardScaler()
    X = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    train_loader = [(X, y)]

    sim = MonteCarloSimulator(
        model_builder = my_network,
        train_loader=train_loader,
        num_runs=30,
    )

    rezultati_kaiming = sim.run_simulation(init_type='kaiming')
    rezultati_xavier = sim.run_simulation(init_type='xavier')
    rezultati_orthogonal = sim.run_simulation(init_type='orthogonal')

    viz = vizualization(rezultati_kaiming,rezultati_xavier,rezultati_orthogonal, activation_name='relu')

    end_time = time.perf_counter()

    print(f"\nVreme izvrsavanje je: {end_time - begin_time:.4f} sekundi\n")

    viz.plotBoxPlots()

    
    #print(f"\nVreme izvrsavanje je: {end_time - begin_time:.4f} sekundi\n")
    #viz.plotDiagnostics()