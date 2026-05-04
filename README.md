# Weight Initialization Analysis

This project analyzes the impact of different weight initialization techniques (Kaiming, Xavier, Orthogonal) on neural network training dynamics using Monte Carlo simulations.

## Features

- **Dynamic Neural Network**: Flexible feed-forward network architecture
- **Metrics Collection**: Gradient norms and variance ratios during training
- **Monte Carlo Simulation**: Statistical analysis over multiple runs
- **Visualization**: Box plots comparing initialization methods

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- seaborn
- matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Testing

Run tests to verify the code works correctly:
```bash
pytest
```

Tests cover:
- Model initialization and forward passes
- Metrics collection (gradients, variances, activations)
- Monte Carlo simulation
- Data visualization preparation

## Usage

Run the main analysis:
```bash
python main.py
```

This will:
1. Generate synthetic classification data
2. Run Monte Carlo simulations for each initialization method
3. Display box plots comparing gradient norms and variance ratios

## Project Structure

- `model.py`: Neural network architecture with weight initialization
- `MetricsClasses.py`: Metrics collection utilities
- `Simulator.py`: Monte Carlo simulation framework
- `Vizuelizacija.py`: Visualization classes
- `main.py`: Main execution script

## Results

The analysis shows how different initialization schemes affect:
- Gradient flow during backpropagation
- Variance propagation through network layers
- Training stability and convergence

## Contributing

Feel free to open issues or submit pull requests for improvements.