import torch
import pytest
from model import DynamicNN
from MetricsClasses import MetricsCollector

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def test_metrics_collector_initialization():
    """Test MetricsCollector initialization"""
    model = DynamicNN(input_dim=10, hidden_dim=[5], output_dim=1).to(DEVICE)
    train_loader = [(torch.randn(10, 10).to(DEVICE), torch.randn(10, 1).to(DEVICE))]
    
    collector = MetricsCollector(model, train_loader, device=DEVICE)
    assert collector.model == model
    assert collector.train_loader == train_loader

def test_gradient_norms():
    """Test gradient norms calculation"""
    model = DynamicNN(input_dim=10, hidden_dim=[5], output_dim=1).to(DEVICE)
    x = torch.randn(5, 10).to(DEVICE)
    y = torch.randn(5, 1).to(DEVICE)
    train_loader = [(x, y)]
    
    collector = MetricsCollector(model, train_loader, device=DEVICE)
    grad_norms = collector.get_gradient_norms()
    
    # Should return list of gradient norms for each Linear layer
    assert isinstance(grad_norms, list)
    # Count Linear layers in model
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(grad_norms) == len(linear_layers)
    assert all(isinstance(norm, float) for norm in grad_norms)

def test_variance_ratios():
    """Test variance ratios calculation"""
    model = DynamicNN(input_dim=10, hidden_dim=[5], output_dim=1).to(DEVICE)
    x = torch.randn(5, 10).to(DEVICE)
    train_loader = [(x, torch.randn(5, 1).to(DEVICE))]
    
    collector = MetricsCollector(model, train_loader, device=DEVICE)
    var_ratios = collector.get_variance_ratios()
    
    # Should return list of variance ratios for each layer
    assert isinstance(var_ratios, list)
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(var_ratios) == len(linear_layers)
    assert all(isinstance(ratio, float) for ratio in var_ratios)

def test_activation_histograms():
    """Test activation histograms collection"""
    model = DynamicNN(input_dim=10, hidden_dim=[5], output_dim=1).to(DEVICE)
    x = torch.randn(5, 10).to(DEVICE)
    train_loader = [(x, torch.randn(5, 1).to(DEVICE))]
    
    collector = MetricsCollector(model, train_loader, device=DEVICE)
    activations = collector.get_activation_histograms()
    
    # Should return dict with activation data
    assert isinstance(activations, dict)
    # Check that we have data for layers
    assert len(activations) > 0
    for key, data in activations.items():
        assert isinstance(data, list)
        assert len(data) > 0