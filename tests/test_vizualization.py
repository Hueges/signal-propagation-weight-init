import torch
import pandas as pd
import pytest
from Vizuelizacija import vizualization

def test_visualization_initialization():
    """Test visualization class initialization and data processing"""
    # Create mock data similar to what simulator produces
    kaiming_dict = {
        'gradients': [[0.1, 0.2], [0.15, 0.25]],
        'variances': [[0.8, 0.9], [0.85, 0.95]]
    }
    xavier_dict = {
        'gradients': [[0.12, 0.22], [0.17, 0.27]],
        'variances': [[0.82, 0.92], [0.87, 0.97]]
    }
    orthogonal_dict = {
        'gradients': [[0.14, 0.24], [0.19, 0.29]],
        'variances': [[0.84, 0.94], [0.89, 0.99]]
    }
    
    viz = vizualization(kaiming_dict, xavier_dict, orthogonal_dict, 'relu')
    
    # Check that DataFrame is created
    assert isinstance(viz.records, pd.DataFrame)
    assert not viz.records.empty
    
    # Check columns
    expected_columns = ['Init', 'Layer', 'Type', 'Value']
    assert list(viz.records.columns) == expected_columns
    
    # Check that we have data for all init types and types
    init_types = viz.records['Init'].unique()
    assert set(init_types) == {'kaiming', 'xavier', 'orthogonal'}
    
    types = viz.records['Type'].unique()
    assert set(types) == {'Grad Norm', 'Var Ratio'}