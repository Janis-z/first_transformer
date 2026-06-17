import torch
import torch.nn as nn


class Norm(nn.Module):

    def __init__(self,d_model):
        super(Norm,self).__init__()
        self.Gamma=nn.Parameter(torch.ones(d_model))
        self.Beta=nn.Parameter(torch.zeros(d_model))

        self.cache={
            "normalized_input":[],
            "variance":[]
        }
        
    
    def apply_norm(self,input):
    #norm every token seperatly, not per batch

        #calculate the mean of every token
        #all values + then / num of tokens
        mean=torch.mean(input,dim=-1, keepdim=True)
        
        #calculate the variance
        #(each value - mean)^2 / num of tokens
        variance=torch.var(input,dim=-1, keepdim=True,correction=0)
        
        self.cache["variance"]=variance

        #now apply the formula
        normalized_input=(input-mean)/torch.sqrt(variance + 1e-6)

        self.cache["normalized_input"]=normalized_input
        
        #now add the learnable parameters Gamma and Beta
        output = normalized_input * self.Gamma + self.Beta
        
        return output
    