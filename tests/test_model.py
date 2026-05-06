import torch
import pytest
from model import DynamicNN

def test_dynamic_nn_initialization():
    """Test basic model creation and forward pass"""
    model = DynamicNN(input_dim=10, hidden_dim=[5, 3], output_dim=1)
    
    # Check model structure
    assert len(model.layers) == 2
    assert model.layers[0].in_features == 10
    assert model.layers[0].out_features == 5
    assert model.layers[1].in_features == 5
    assert model.layers[1].out_features == 3
    assert model.output_layer.in_features == 3
    assert model.output_layer.out_features == 1
    
    # Test forward pass
    x = torch.randn(2, 10)
    output = model(x)
    assert output.shape == (2, 1)

def test_weight_initialization():
    """Test different weight initialization methods"""
    init_types = ['kaiming', 'xavier', 'orthogonal']
    
    for init_type in init_types:
        model = DynamicNN(input_dim=10, init_type=init_type, hidden_dim=[5], output_dim=1)
        
        # Check that weights are initialized (not all zeros)
        for layer in model.layers:
            assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert not torch.allclose(model.output_layer.weight, torch.zeros_like(model.output_layer.weight))

def test_activation_functions():
    """Test different activation functions"""
    activations = ['relu', 'sigmoid', 'tanh']
    
    for act in activations:
        model = DynamicNN(input_dim=5, act_type=act, hidden_dim=[3], output_dim=1)
        x = torch.randn(2, 5)
        output = model(x)
        assert output.shape == (2, 1)

def test_invalid_activation():
    """Test that invalid activation raises error"""
    with pytest.raises(ValueError):
        DynamicNN(input_dim=5, act_type='invalid', hidden_dim=[3], output_dim=1)