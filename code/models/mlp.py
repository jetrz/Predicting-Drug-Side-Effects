import torch
import torch.nn.functional as F
from torch.nn import Linear, ParameterList, ReLU

class MLP(torch.nn.Module):
    """Standard MLP.

    Parameters:
    layers (list<tuple>): List of tuples containing the input and output size of each consecutive linear layer. E.g. [(2048, 1500), (1500, 1000), (1000, 1385)]
    
    """
    
    def __init__(self, layers, activation="sigmoid"):
        super().__init__()
        torch.manual_seed(777)

        l = []
        for i in layers:
            l.append(Linear(i[0], i[1]))
            l.append(ReLU())
        del l[-1]
        
        self.layers = ParameterList(l)
        
        if activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "softmax":
            self.activation = torch.nn.Softmax()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None
        

    def forward(self, x):
        for ind, l in enumerate(self.layers):
            x = l(x)
            if (ind+1) % 4 == 0:
               x = F.dropout(x, p=0.5, training=self.training)
        
        if self.activation:
            x = self.activation(x)

        return x