from MetricsClasses import MetricsCollector
import torch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class MonteCarloSimulator:
    """Initialization of simulator, it receives model builder so that it can call method run_simulation for specific weight init"""
    def __init__(self,model_builder,train_loader:list[tuple[torch.Tensor, torch.Tensor]], num_runs=100,device=DEVICE):
        self.model_builder = model_builder
        self.train_loader = train_loader
        self.num_runs = num_runs
        self.device = device

        self.results ={}

    def run_simulation(self,init_type:str):
        """Runs simulation of weight initializations for a given initialization type """

        print(f"Starting simulation of ({self.num_runs} iterations ) for {init_type}")

        self.results[init_type] = {
            'gradients': [],
            'variances': []
        }
        
        for i in range(self.num_runs):

            model = self.model_builder(init_type=init_type).to(self.device)

            collector = MetricsCollector(model,self.train_loader,self.device)

            grads = collector.get_gradient_norms()
            var_ratios = collector.get_variance_ratios()

            self.results[init_type]['gradients'].append(grads)
            self.results[init_type]['variances'].append(var_ratios)

        print(f"Done for {init_type}")

        return self.results[init_type]