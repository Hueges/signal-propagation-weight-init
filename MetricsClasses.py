import torch 
import torch.nn as nn
from collections import defaultdict
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MetricsCollector:
    """Class that receives network model and train loader"""
    acceptable_layers = (nn.Linear)

    def __init__(self,model,train_loader,device=DEVICE):
        self.model = model
        self.train_loader = train_loader
        self.device = device


    def get_activation_histograms(self):
        """Method that collects outputs  after actiavtion function in a forward pass and stores them in dictionary"""
        activations = defaultdict(list)
        hooks = []

        def get_activations(name):
            def hook(module,input,output):
                activations[name]=output.detach().cpu().numpy().flatten()
            return hook
        
        count =1 
        for name, layer in self.model.named_modules():
            if isinstance(layer, self.acceptable_layers):
                hooks.append(layer.register_forward_hook(get_activations(f"Layer {count} ({name})")))
                count+=1
        Xb, _ = next(iter(self.train_loader))

        with torch.no_grad():
            self.model(Xb.to(self.device))
        
        for h in hooks: h.remove()
        return activations
    
    
    def get_gradient_norms(self):
        """Method that saves gradient norms of a network during backpropagation"""
        self.model.train()
        self.model.zero_grad()

        Xb,yb = next(iter(self.train_loader))
        Xb,yb = Xb.to(self.device), yb.to(self.device)

        if yb.dim() == 1: yb = yb.unsqueeze(1)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(self.model(Xb),yb)
        loss.backward()

        grad_norms = []

        for layer in self.model.modules():
            if isinstance(layer, self.acceptable_layers):
                grad_norms.append(layer.weight.grad.norm(2).item())
        
        self.model.zero_grad()
        return grad_norms
    
    
    def get_variance_ratios(self):
        """Method that catches variance of inputs and variance of outputs and saves their ratio in a list"""
        self.model.eval()
        ratios = []
        hooks = []

        def ratio_hook(module,input,output):
            x = input[0].detach()
            y = output.detach()
            r = y.var()/(x.var() + 1e-8)
            ratios.append(r.item())
        
        for layer in self.model.modules():
            if isinstance(layer, self.acceptable_layers):
                hooks.append(layer.register_forward_hook(ratio_hook))

        Xb,_ = next(iter(self.train_loader))

        with torch.no_grad():
            self.model(Xb.to(self.device))
        
        for h in hooks: h.remove()
        return ratios