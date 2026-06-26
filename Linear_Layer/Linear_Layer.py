import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, d_model,vokab_size):
        super(LinearLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(vokab_size, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(vokab_size))

        self.cache = {
            "input": []
        }

    def forward(self, input):
        #Flip the weights to make the matrixes the same size
        self.cache["input"] = input
        return input @ self.W.T + self.b

    


        