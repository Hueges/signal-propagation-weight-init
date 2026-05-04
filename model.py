import torch 
import torch.nn as nn
from _collections_abc import Sequence

class DynamicNN(nn.Module):
    """Class that creates dynamical feed forward network architecture"""
    def __init__(self, input_dim:int, init_type = None, hidden_dim:Sequence[int] = (32,32,32,32,16),
                 output_dim:int = 1,act_type:str ='relu'):
        super().__init__()
        
        self.activation_name = act_type

        activations = {
            'relu' : nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh':nn.Tanh()
        }

        if act_type not in activations:
            raise ValueError(f"There is no such activation function {act_type}\n")

        self.act = activations[act_type]

        self.layers = nn.ModuleList()

        current_dim = input_dim

        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(in_features=current_dim,out_features=h_dim))
            current_dim = h_dim

        self.output_layer = nn.Linear(current_dim,output_dim)

        if init_type is not None:
            self._init_weights(init_type)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        
        x = self.output_layer(x)
        return x
    
    def _init_weights(self,init_type:str):
        """This method initializes weights for a given architecture, one of: Kaiming, Xavier, Orthogonal """
        for m in self.modules():
            if isinstance(m,nn.Linear):

                # if m.out_features == 1:
                #     nn.init.xavier_normal_(m.weight)
                #     if m.bias is not None:
                #         nn.init.constant_(m.bias, 0)
                #     continue
                
                #calculating gain for choosen activation function
                try:
                    gain = nn.init.calculate_gain(self.activation_name)
                except:
                    gain=1.0
                
                if init_type == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity=self.activation_name)
                if init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight,gain=gain)
                if init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight,gain=gain)
                
                #setting bias to be zero always
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)