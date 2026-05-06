import torch
import pytest
from model import DynamicNN
from Simulator import MonteCarloSimulator

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def test_simulator_initialization():
    """Test MonteCarloSimulator initialization"""
    def model_builder(init_type):
        return DynamicNN(input_dim=10, init_type=init_type, hidden_dim=[5], output_dim=1).to(DEVICE)
    
    train_loader = [(torch.randn(5, 10).to(DEVICE), torch.randn(5, 1).to(DEVICE))]
    
    sim = MonteCarloSimulator(model_builder, train_loader, num_runs=3, device=DEVICE)
    assert sim.model_builder == model_builder
    assert sim.train_loader == train_loader
    assert sim.num_runs == 3

def test_run_simulation():
    """Test running simulation for one initialization type"""
    def model_builder(init_type):
        return DynamicNN(input_dim=10, init_type=init_type, hidden_dim=[5], output_dim=1).to(DEVICE)
    
    train_loader = [(torch.randn(5, 10).to(DEVICE), torch.randn(5, 1).to(DEVICE))]
    
    sim = MonteCarloSimulator(model_builder, train_loader, num_runs=2, device=DEVICE)
    results = sim.run_simulation('kaiming')
    
    # Check results structure
    assert 'gradients' in results
    assert 'variances' in results
    assert len(results['gradients']) == 2  # num_runs
    assert len(results['variances']) == 2
    assert len(results['gradients'][0]) > 0  # gradient norms for layers
    assert len(results['variances'][0]) > 0  # variance ratios for layers