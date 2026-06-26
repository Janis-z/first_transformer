import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, Layer_count):
        super(FeedForward, self).__init__()

        self.inputLayer_size = d_model
        self.outputLayer_size = d_model
        self.layer_count = Layer_count
        self.hiddenLayer_size = d_model * 4

        #Weights and Biases
        self.weightList = nn.ParameterList()
        self.biasList = nn.ParameterList()

        #store calculations in cache for backpropagation
        self.cache = {
            "inputLayer": [],
            "afterReLU": [],
        }

        #first layer is smaller so sepratly
        self.W1 = nn.Parameter(torch.randn(self.inputLayer_size, self.hiddenLayer_size) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(self.hiddenLayer_size))

        self.weightList.append(self.W1)
        self.biasList.append(self.b1)

        self.W_output = nn.Parameter(torch.randn(self.hiddenLayer_size, self.outputLayer_size) * 0.02)
        self.b_output = nn.Parameter(torch.zeros(self.outputLayer_size))

        #for layercount create weights and biases
        for i in range(2, Layer_count):
            weights_name = f"W{i}"
            bias_name = f"b{i}"

            weights = nn.Parameter(torch.randn(self.hiddenLayer_size, self.hiddenLayer_size) * 0.02)
            bias = nn.Parameter(torch.zeros(self.hiddenLayer_size))

            setattr(self, weights_name, weights)
            setattr(self, bias_name, bias)
            self.weightList.append(getattr(self, weights_name))
            self.biasList.append(getattr(self, bias_name))
        
        self.weightList.append(self.W_output)
        self.biasList.append(self.b_output)

    
    def calculate(self, input):
        # delete cache on next calculation because backpropagation has finished
        self.cache["inputLayer"] = []
        self.cache["afterReLU"] = []

        #First Layer sepratly because smaller
        L1 = input @ self.weightList[0] + self.biasList[0]
        
        self.cache["inputLayer"].append(L1)
        self.cache["afterReLU"].append(torch.relu(L1))
        
        #Hidden Layers
        #Start at i because first was seperatly and remove last too
        for i in range(1, len(self.weightList) - 1):
            prev_activation = self.cache["afterReLU"][-1]
            layer = prev_activation @ self.weightList[i] + self.biasList[i]
            
            self.cache["inputLayer"].append(layer)
            self.cache["afterReLU"].append(torch.relu(layer))
        
        #Last Layer with last weights
        output = self.cache["afterReLU"][-1] @ self.weightList[-1] + self.biasList[-1]

        return output

