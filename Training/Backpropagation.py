import torch
import torch.nn as nn

class Backpropagation(nn.Module):
    
    @staticmethod
    def calculate(Decoder_list, Encoder_list, linear_layer, output_probabilities, targets, Encoder_Ids,Decoder_Ids, learningRate, context_size, batch_size):

#Look at the picture of the transformer for better understanding of the build
#just go backwards from the output 
#first calculate all decoders(right) then add the outputs oft the mha togethter to get the input for the encoders(left)
#for better understandings of the math functions look in the folders unbder Training variable names may not fit correctly to the pictures
#transpose is applied to swap values -1 and -2 means last 2 values - starts from the back
#sum on dim means summing an that dimension here used to calculate every batch seperatly (dim=0)
#also look in the folders of the seperate parts to understand the math better like the split of the mha
#-----------------------------------------------------------------------------------------------------------------------------
        #1. Softmax
#----------------------------------------------------------------------------------------------------------------------------
        # 1. Ensure targets has the shape (batch_size, context_size)
        targets = targets.view(batch_size, context_size)
        
        # 2. Copy the probabilities (P)
        # For all incorrect classes (Y = 0), the value remains P (since P - 0 = P)
        loss = output_probabilities.clone()
        
        # 3. Subtract 1 at the correct target ID positions (P - 1)
        for b in range(batch_size):
            loss[b, torch.arange(context_size), targets[b]] -= 1.0
            
        # 4. Mask out the loss for positions that correspond to [PAD] (id: 50259)
        # This prevents the model from learning to predict padding tokens
        pad_mask = (targets == 50259)  # Shape: (batch_size, context_size)
        loss[pad_mask] = 0.0

        # 'loss' now has the shape (batch_size, context_size, vocab_size)
        # and contains:
        # - at the correct positions: (probability - 1)
        # - at all other positions: (probability)
        # - at [PAD] positions: 0.0

#----------------------------------------------------------------------------------------------------------------------------
        #2. Linear Layer
#-----------------------------------------------------------------------------------------------------------------------------
        vocab_size = linear_layer.W.shape[0]
        d_model = linear_layer.W.shape[1]

        # Flatten inputs and gradients to 2D
        input_flat = linear_layer.cache["input"].view(-1, d_model)
        d_flat = loss.view(-1, vocab_size)

        # Average gradient for weights (dW): Shape (vocab_size, d_model)
        d_Weights = (d_flat.T @ input_flat)

        # Average gradient for bias (db): Shape (vocab_size)
        d_bias = torch.sum(d_flat, dim=0)

        # Average gradient for input to pass back to the decoder (dX): Shape (batch_size, context_size, d_model)
        d_input = (loss @ linear_layer.W)


        #add derivative
        #must be done in no_grad because otherwise it is not a tensor
        with torch.no_grad():
            linear_layer.W -= d_Weights * learningRate
            linear_layer.b -= d_bias * learningRate




#Decoders
#-----------------------------------------------------------------------------------------------------------------------------
        #3. go through the decoders
#----------------------------------------------------------------------------------------------------------------------------
        #input for the encoder
        d_K_combine = torch.zeros(batch_size, context_size, d_model, device=d_input.device)
        d_V_combine = torch.zeros(batch_size, context_size, d_model, device=d_input.device)

        for Decoder in Decoder_list[::-1]:
            
            #1. add split
            d_into_ff=d_input
            d_bypass=d_input

        #--------------------------------------------------------------------------------------------------------            
            #2. Feed-Forward
        #--------------------------------------------------------------------------------------------------------
            FF=Decoder.feedforward

            # 1. Backpropagate last Feed-Forward layer (weightList[-1])
            d_Weights = FF.cache["afterReLU"][-1].transpose(-1, -2) @ d_into_ff
            d_Weights = torch.sum(d_Weights, dim=0)
            d_Biases = torch.sum(d_into_ff, dim=(0, 1))

            d_input = d_into_ff @ FF.weightList[-1].T

            with torch.no_grad():
                FF.weightList[-1] -= d_Weights * learningRate
                FF.biasList[-1] -= d_Biases * learningRate

            d_into_ff = d_input

            # 2. Backpropagate remaining Feed-Forward layers
            for layer_number in reversed(range(FF.layer_count-1)):

                #relu derivative 
                #all values who were over 1 go through rest doesnt
                #see mask in mmha folder
                relu_mask=(FF.cache["inputLayer"][layer_number-1]>0).float()

                d_into_ff=d_input * relu_mask
                


                #calculate gradients (calculations in FF_math)
                #d_into_ff = d_into_ff * relu_mask
                #when 0 take norm because thats the input layer
                if layer_number>0:
                    d_Weights=torch.sum(FF.cache["afterReLU"][layer_number-1].transpose(-1,-2) @ (d_into_ff),dim=0)   
                else:
                    d_Weights=torch.sum(Decoder.cache["norm_output_3"].transpose(-1,-2) @ (d_into_ff),dim=0)

                d_Biases=torch.sum(d_into_ff, dim=(0,1))

                d_input=d_into_ff @ FF.weightList[layer_number].transpose(-1,-2)


                #add derivative 
                with torch.no_grad():
                    FF.weightList[layer_number] -= d_Weights * learningRate
                    FF.biasList[layer_number] -= d_Biases * learningRate
        

                
                
                
        #-----------------------------------------------------------------------------------------------------------------
            #3. norm
        #-----------------------------------------------------------------------------------------------------------------
            d_Beta=torch.sum(d_input,dim=(0,1))

            d_Gamma=torch.sum(d_input * Decoder.norm3.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Decoder.norm3.Gamma

            #add derivative 
            with torch.no_grad():
                Decoder.norm3.Gamma -= d_Gamma * learningRate
                Decoder.norm3.Beta -= d_Beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Decoder.norm3.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Decoder.norm3.cache["normalized_input"] * torch.sum((d_x_hat*Decoder.norm3.cache["normalized_input"]),dim=-1,keepdim=True))))
            
        #-----------------------------------------------------------------------------------------------------------------
            #4. add
        #-----------------------------------------------------------------------------------------------------------------
            d_input=d_x + d_bypass

            #5.add split
            d_bypass=d_input

        #-----------------------------------------------------------------------------------------------------------------
            #6. mha
        #-----------------------------------------------------------------------------------------------------------------
            #all math functions under mha_math
            d_WO=torch.sum(Decoder.mha.cache["H"].transpose(-1,-2) @ d_input, dim=0)

            d_H= d_input @ Decoder.mha.Wo.weight.T

            #add derivative
            with torch.no_grad():
                Decoder.mha.Wo.weight -= d_WO * learningRate
            

            #split d_h into num_heads
            d_H_split= d_H.view(batch_size,context_size,Decoder.mha.num_heads,Decoder.mha.head_size).permute(0,2,1,3)

            d_Q_K_Soft = d_H_split @ Decoder.mha.cache["Vw_split"].transpose(-1, -2)

            d_Vw_split = Decoder.mha.cache["Q_K_Soft"].transpose(-1, -2) @ d_H_split

            d_S= Decoder.mha.cache["Q_K_Soft"] * (d_Q_K_Soft - (torch.sum(Decoder.mha.cache["Q_K_Soft"] * d_Q_K_Soft, dim=-1, keepdim=True)))

            d_Q_K= d_S / (Decoder.mha.head_size ** 0.5)

            d_Qw_split= d_Q_K @ Decoder.mha.cache["Kw_split"]

            d_Kw_split= d_Q_K.transpose(-2,-1) @ Decoder.mha.cache["Qw_split"]

            #put all together
            d_Vw= d_Vw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            d_Qw= d_Qw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            d_Kw= d_Kw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            #weights
            d_Wq= torch.sum(Decoder.mha.cache["Q"].transpose(-1,-2) @ d_Qw, dim=0)

            d_Wk= torch.sum(Decoder.mha.cache["K"].transpose(-1,-2) @ d_Kw, dim=0)

            d_Wv= torch.sum(Decoder.mha.cache["V"].transpose(-1,-2) @ d_Vw, dim=0)


            #add derivative
            with torch.no_grad():
                Decoder.mha.Wq.weight -= d_Wq * learningRate
                Decoder.mha.Wk.weight -= d_Wk * learningRate
                Decoder.mha.Wv.weight -= d_Wv * learningRate


            #inputs
            d_Q= d_Qw @ Decoder.mha.Wq.weight.T

            d_K= d_Kw @ Decoder.mha.Wk.weight.T

            d_V= d_Vw @ Decoder.mha.Wv.weight.T

            #combine K and V because they go to the Encoder
            d_K_combine= d_K_combine + d_K

            d_V_combine= d_V_combine + d_V


            #d_Q goes through the decoder
            d_input=d_Q

        #-----------------------------------------------------------------------------------------------------------------
            #7. norm
        #-----------------------------------------------------------------------------------------------------------------
            d_Beta=torch.sum(d_input,dim=(0,1))

            d_Gamma=torch.sum(d_input * Decoder.norm2.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Decoder.norm2.Gamma

            #add derivative 
            with torch.no_grad():
                Decoder.norm2.Gamma -= d_Gamma * learningRate
                Decoder.norm2.Beta -= d_Beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Decoder.norm2.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Decoder.norm2.cache["normalized_input"] * torch.sum((d_x_hat*Decoder.norm2.cache["normalized_input"]),dim=-1,keepdim=True))))
            
        #-----------------------------------------------------------------------------------------------------------------
            #7.add
        #-----------------------------------------------------------------------------------------------------------------
            d_input=d_x + d_bypass

        #-----------------------------------------------------------------------------------------------------------------
            #8.mmha
        #-----------------------------------------------------------------------------------------------------------------
            #same ass mha but tiny change on d_Q_K where a mask is added and end does not split up
            #all math functions under mha_math
            d_WO=torch.sum(Decoder.mmha.cache["H"].transpose(-1,-2) @ d_input, dim=0)

            d_H= d_input @ Decoder.mmha.Wo.weight.T

            #add derivative
            with torch.no_grad():
                Decoder.mmha.Wo.weight -= d_WO * learningRate
            

            #split d_h into num_heads
            d_H_split= d_H.view(batch_size,context_size,Decoder.mmha.num_heads,Decoder.mmha.head_size).permute(0,2,1,3)

            d_Q_K_Soft = d_H_split @ Decoder.mmha.cache["Vw_split"].transpose(-1, -2)

            d_Vw_split = Decoder.mmha.cache["Q_K_Soft"].transpose(-1, -2) @ d_H_split

            d_S= Decoder.mmha.cache["Q_K_Soft"] * (d_Q_K_Soft - (torch.sum(Decoder.mmha.cache["Q_K_Soft"] * d_Q_K_Soft, dim=-1, keepdim=True)))

            d_Q_K= d_S / (Decoder.mmha.head_size ** 0.5)

            #mask 
            mask = torch.tril(torch.ones(context_size, context_size, device=d_Q_K.device))
            d_Q_K_Masked = d_Q_K.masked_fill(mask == 0, 0)

            d_Qw_split= d_Q_K_Masked @ Decoder.mmha.cache["Kw_split"]

            d_Kw_split= d_Q_K_Masked.transpose(-2,-1) @ Decoder.mmha.cache["Qw_split"]

            #put all together
            d_Vw= d_Vw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            d_Qw= d_Qw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            d_Kw= d_Kw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            #weights
            d_Wq= torch.sum(Decoder.cache["norm_output_1"].transpose(-1,-2) @ d_Qw, dim=0)

            d_Wk= torch.sum(Decoder.cache["norm_output_1"].transpose(-1,-2) @ d_Kw, dim=0)

            d_Wv= torch.sum(Decoder.cache["norm_output_1"].transpose(-1,-2) @ d_Vw, dim=0)


            #add derivative
            with torch.no_grad():
                Decoder.mmha.Wq.weight -= d_Wq * learningRate
                Decoder.mmha.Wk.weight -= d_Wk * learningRate
                Decoder.mmha.Wv.weight -= d_Wv * learningRate


            #inputs
            d_Q= d_Qw @ Decoder.mmha.Wq.weight.T

            d_K= d_Kw @ Decoder.mmha.Wk.weight.T

            d_V= d_Vw @ Decoder.mmha.Wv.weight.T

            #all Values go to the output
            d_input=d_Q + d_K + d_V

        #-----------------------------------------------------------------------------------------------------------------
            #9. norm
        #-----------------------------------------------------------------------------------------------------------------
            d_Beta=torch.sum(d_input,dim=(0,1))

            d_Gamma=torch.sum(d_input * Decoder.norm1.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Decoder.norm1.Gamma

            #add derivative
            with torch.no_grad():
                Decoder.norm1.Gamma -= d_Gamma * learningRate
                Decoder.norm1.Beta -= d_Beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Decoder.norm1.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Decoder.norm1.cache["normalized_input"] * torch.sum((d_x_hat*Decoder.norm1.cache["normalized_input"]),dim=-1,keepdim=True))))

        #-----------------------------------------------------------------------------------------------------------------
            #10. add
        #-----------------------------------------------------------------------------------------------------------------
            d_input= d_bypass + d_x

            #this input goes into the next decoder till it reaches the end

        #update the token embedings

        Token_Embeddings= torch.load("Token_Embeddings.pt")

        d_Token_Embeddings = torch.zeros_like(Token_Embeddings)

        #tensor with ids of the input tokens
        ids_tensor = torch.tensor(Decoder_Ids, device=d_input.device).view(-1)
        d_input_flat = d_input.view(-1, d_model)

        #set values of d_input on the Ids they come from
        d_Token_Embeddings.index_add_(0, ids_tensor, d_input_flat)

        #add the gradient to the Token_Embedings
        with torch.no_grad():
            Token_Embeddings -= d_Token_Embeddings * learningRate

        #save change
        torch.save(Token_Embeddings, "Token_Embeddings.pt")




#Encoder
#--------------------------------------------------------------------------------------------------------------------------------



        #add d_K_combine and d_V_combine to get the Encoder Input
        d_input = d_K_combine+ d_V_combine
        
#-----------------------------------------------------------------------------------------------------------------
        #4. Go through Encoder
#-----------------------------------------------------------------------------------------------------------------
        #same as decoder but without the mmha
        for Encoder in Encoder_list[::-1]:
            
            #1. add split
            d_into_ff=d_input
            d_bypass=d_input

        #-----------------------------------------------------------------------------------------------------------------
            #2. Feed-Forward
        #-----------------------------------------------------------------------------------------------------------------
            FF=Encoder.feedforward

            # 1. Backpropagate last Feed-Forward layer (weightList[-1])
            d_Weights = FF.cache["afterReLU"][-1].transpose(-1, -2) @ d_into_ff
            d_Weights = torch.sum(d_Weights, dim=0)
            d_Biases = torch.sum(d_into_ff, dim=(0, 1))

            d_input = d_into_ff @ FF.weightList[-1].T

            with torch.no_grad():
                FF.weightList[-1] -= d_Weights * learningRate
                FF.biasList[-1] -= d_Biases * learningRate

            d_into_ff = d_input

            # 2. Backpropagate remaining Feed-Forward layers
            for layer_number in reversed(range(FF.layer_count-1)):

                #relu derivative 
                #all values who were over 1 go through rest doesnt
                relu_mask=(FF.cache["inputLayer"][layer_number-1]>0).float()
                d_into_ff=d_input * relu_mask

                #calculate gradients (calculations in FF_math)
                #when 0 take norm because thats the input layer
                if layer_number>0:
                    d_Weights=torch.sum(FF.cache["afterReLU"][layer_number-1].transpose(-1,-2) @ (d_into_ff),dim=0)   
                else:
                    d_Weights=torch.sum(Encoder.cache["norm_output_2"].transpose(-1,-2) @ (d_into_ff),dim=0)

                d_Biases=torch.sum(d_into_ff, dim=(0,1))

                d_input=d_into_ff @ FF.weightList[layer_number].transpose(-1,-2)

                #add derivative 
                with torch.no_grad():
                    FF.weightList[layer_number] -= d_Weights * learningRate
                    FF.biasList[layer_number] -= d_Biases * learningRate
                
        #-----------------------------------------------------------------------------------------------------------------
            #3. norm
        #-----------------------------------------------------------------------------------------------------------------
            d_Beta=torch.sum(d_input,dim=(0,1))

            d_Gamma=torch.sum(d_input * Encoder.norm2.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Encoder.norm2.Gamma

            #add derivative 
            with torch.no_grad():
                Encoder.norm2.Gamma -= d_Gamma * learningRate
                Encoder.norm2.Beta -= d_Beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Encoder.norm2.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Encoder.norm2.cache["normalized_input"] * torch.sum((d_x_hat*Encoder.norm2.cache["normalized_input"]),dim=-1,keepdim=True))))
            
        #-----------------------------------------------------------------------------------------------------------------
            #4. add
        #-----------------------------------------------------------------------------------------------------------------
            d_input=d_x + d_bypass

            #5.add split
            d_bypass=d_input

        #-----------------------------------------------------------------------------------------------------------------
            #6. mha
        #-----------------------------------------------------------------------------------------------------------------
            #all math functions under mha_math
            d_WO=torch.sum(Encoder.mha.cache["H"].transpose(-1,-2) @ d_input, dim=0)

            d_H= d_input @ Encoder.mha.Wo.weight.T

            #add derivative
            with torch.no_grad():
                Encoder.mha.Wo.weight -= d_WO * learningRate
            

            #split d_h into num_heads
            d_H_split= d_H.view(batch_size,context_size,Encoder.mha.num_heads,Encoder.mha.head_size).permute(0,2,1,3)

            d_Q_K_Soft = d_H_split @ Encoder.mha.cache["Vw_split"].transpose(-1, -2)

            d_Vw_split = Encoder.mha.cache["Q_K_Soft"].transpose(-1, -2) @ d_H_split

            d_S= Encoder.mha.cache["Q_K_Soft"] * (d_Q_K_Soft - (torch.sum(Encoder.mha.cache["Q_K_Soft"] * d_Q_K_Soft, dim=-1, keepdim=True)))

            d_Q_K= d_S / (Encoder.mha.head_size ** 0.5)

            d_Qw_split= d_Q_K @ Encoder.mha.cache["Kw_split"]

            d_Kw_split= d_Q_K.transpose(-2,-1) @ Encoder.mha.cache["Qw_split"]

            #put all together
            d_Vw= d_Vw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            d_Qw= d_Qw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            d_Kw= d_Kw_split.permute(0,2,1,3).contiguous().view(batch_size, context_size, d_model)

            #weights
            d_Wq= torch.sum(Encoder.mha.cache["Q"].transpose(-1,-2) @ d_Qw, dim=0)

            d_Wk= torch.sum(Encoder.mha.cache["K"].transpose(-1,-2) @ d_Kw, dim=0)

            d_Wv= torch.sum(Encoder.mha.cache["V"].transpose(-1,-2) @ d_Vw, dim=0)


            #add derivative
            with torch.no_grad():
                Encoder.mha.Wq.weight -= d_Wq * learningRate
                Encoder.mha.Wk.weight -= d_Wk * learningRate
                Encoder.mha.Wv.weight -= d_Wv * learningRate


            #inputs
            d_Q= d_Qw @ Encoder.mha.Wq.weight.T

            d_K= d_Kw @ Encoder.mha.Wk.weight.T

            d_V= d_Vw @ Encoder.mha.Wv.weight.T

            #combine K and V because they go to the Encoder
            d_K_combine= d_K_combine + d_K

            d_V_combine= d_V_combine + d_V


            #d_Q goes through the decoder
            d_input=d_Q

        #-----------------------------------------------------------------------------------------------------------------
            #7. norm
        #-----------------------------------------------------------------------------------------------------------------
            d_Beta=torch.sum(d_input,dim=(0,1))

            d_Gamma=torch.sum(d_input * Encoder.norm1.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Encoder.norm1.Gamma

            #add derivative 
            with torch.no_grad():
                Encoder.norm1.Gamma -= d_Gamma * learningRate
                Encoder.norm1.Beta -= d_Beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Encoder.norm1.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Encoder.norm1.cache["normalized_input"] * torch.sum((d_x_hat*Encoder.norm1.cache["normalized_input"]),dim=-1,keepdim=True))))
            
        #-----------------------------------------------------------------------------------------------------------------
            #7.add
        #-----------------------------------------------------------------------------------------------------------------
            d_input=d_x + d_bypass


        #update the token embedings

        Token_Embeddings= torch.load("Token_Embeddings.pt")

        d_Token_Embeddings = torch.zeros_like(Token_Embeddings)

        #tensor with ids of the input tokens
        ids_tensor = torch.tensor(Encoder_Ids, device=d_input.device).view(-1)
        d_input_flat = d_input.view(-1, d_model)

        #set values of d_input on the Ids they come from
        d_Token_Embeddings.index_add_(0, ids_tensor, d_input_flat)

        #add the gradient to the Token_Embedings
        Token_Embeddings = Token_Embeddings - d_Token_Embeddings * learningRate

        #save change
        torch.save(Token_Embeddings, "Token_Embeddings.pt")
