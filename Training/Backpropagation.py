from Input import Token_Embedings
import torch
import torch.nn as nn

class Backpropagation(nn.Module):
    
    @staticmethod
    def calculate(Decoder_list, Encoder_list, linear_layer, output_probabilities, targets, Ids, learningRate, context_size, batch_size):

        #1. Softmax

        # 1. Ensure targets has the shape (batch_size, context_size)
        targets = targets.view(batch_size, context_size)
        
        # 2. Copy the probabilities (P)
        # For all incorrect classes (Y = 0), the value remains P (since P - 0 = P)
        loss = output_probabilities.clone()
        
        # 3. Subtract 1 at the correct target ID positions (P - 1)
        for b in range(batch_size):
            loss[b, torch.arange(context_size), targets[b]] -= 1.0
            
        # 'loss' now has the shape (batch_size, context_size, vocab_size)
        # and contains:
        # - at the correct positions: (probability - 1)
        # - at all other positions: (probability)


        #2. Linear Layer
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
        linear_layer.W = linear_layer.W - d_Weights * learningRate
        linear_layer.b = linear_layer.b - d_bias * learningRate
        
        #3. go through the decoders

        for Decoder in Decoder_list[::-1]:
            
            #1. add split
            d_into_ff=d_input
            d_bypass=d_input
            
            #2. Feed-Forward
            FF=Decoder.feedforward

            #count layers backwards
            #-1 because output not needed
            for layer_number in reversed(range(FF.layer_count-1)):

                #calculate gradients (calculations in FF_math)
                #d_into_ff = d_into_ff * relu_mask
                d_Weights=FF.cache["inputLayer"][layer_number].T @ (d_into_ff)   

                d_Biases=torch.sum(d_into_ff, dim=0)

                d_input=d_into_ff @ FF.weightList[layer_number].T


                #add derivative 
                FF.weightList[layer_number] = FF.weightList[layer_number] - d_Weights * learningRate
                FF.biasList[layer_number] = FF.biasList[layer_number] - d_Biases * learningRate
        

                
                #relu derivative 
                #all values who were over 1 go through rest doesnt

                if layer_number>0:
                    relu_mask=(FF.cache["inputLayer"][layer_number-1]>0).float()

                    d_into_ff=d_input @ relu_mask
                

            #3. norm
            d_beta=torch.sum(d_input,dim=(0,1))

            d_gamma=torch.sum(d_input * Decoder.norm3.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Decoder.norm3.Gamma

            #add derivative 
            Decoder.norm3.Gamma = Decoder.norm3.Gamma - d_gamma * learningRate
            Decoder.norm3.beta = Decoder.norm3.beta - d_beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Decoder.norm3.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Decoder.norm3.cache["normalized_input"] * torch.sum((d_x_hat*Decoder.norm3.cache["normalized_input"]),dim=-1,keepdim=True))))
            

            #4. add
            d_input=d_x + d_bypass

            #5.add split
            d_bypass=d_input

            #6. mha
            #all math functions under mha_math
            d_WO=Decoder.mha.cache["H"].T @ d_input 

            d_H= d_input @ Decoder.mha.Wo.T

            #add derivative
            Decoder.mha.Wo = Decoder.mha.Wo - d_WO * learningRate
            

            #split d_h into num_heads
            d_H_split= d_H.view(batch_size,context_size,Decoder.mha.num_heads,Decoder.mha.head_size).permute(0,2,1,3)

            d_Vw_split= Decoder.mha.cache["H_Split"].transpose(-1,-2) @ d_H_split

            d_Q_K_Soft= (Decoder.mha.Vw_split.transpose(-1,-2)) @ d_H_split

            d_S= Decoder.mha.Q_K_Soft * (d_Q_K_Soft - (torch.sum(Decoder.mha.Q_K_Soft * d_Q_K_Soft, dim=-1, keepdim=True)))

            d_Q_K= d_S / torch.sqrt(Decoder.mha.head_size)

            d_Qw_split= d_Q_K @ Decoder.mha.Kw_split

            d_Kw_split= d_Q_K.transpose(-2,-1) @ Decoder.mha.Qw_split

            #put all together
            d_Vw= d_Vw_split.permute(0,2,1,3).contiguous().view(batch_size, Decoder.mha.seq_len, d_model)

            d_Qw= d_Qw_split.permute(0,2,1,3).contiguous().view(batch_size, Decoder.mha.seq_len, d_model)

            d_Kw= d_Kw_split.permute(0,2,1,3).contiguous().view(batch_size, Decoder.mha.seq_len, d_model)

            #weights
            d_Wq= Decoder.mha.cache["Q"].T @ d_Qw

            d_Wk= Decoder.mha.cache["K"].T @ d_Kw

            d_Wv= Decoder.mha.cache["V"].T @ d_Vw


            #add derivative
            Decoder.mha.Wq = Decoder.mha.Wq - d_Wq * learningRate
            Decoder.mha.Wk = Decoder.mha.Wk - d_Wk * learningRate
            Decoder.mha.Wv = Decoder.mha.Wv - d_Wv * learningRate


            #inputs
            d_Q= d_Qw @ Decoder.mha.Wq.T

            d_K= d_Kw @ Decoder.mha.Wk.T

            d_V= d_Vw @ Decoder.mha.Wv.T

            #combine K and V because they go to the Encoder
            d_K_combine= d_K_combine + d_K

            d_V_combine= d_V_combine + d_V


            #d_Q goes through the decoder
            d_input=d_Q


            #7. norm
            d_beta=torch.sum(d_input,dim=(0,1))

            d_gamma=torch.sum(d_input * Decoder.norm2.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Decoder.norm2.Gamma

            #add derivative 
            Decoder.norm2.Gamma = Decoder.norm2.Gamma - d_gamma * learningRate
            Decoder.norm2.beta = Decoder.norm2.beta - d_beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Decoder.norm2.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Decoder.norm2.cache["normalized_input"] * torch.sum((d_x_hat*Decoder.norm2.cache["normalized_input"]),dim=-1,keepdim=True))))
            

            #7.add
            d_input=d_x + d_bypass


            #8.mmha
            #same ass mha but tiny change on d_Q_K where a mask is added and end does not split up
            #all math functions under mha_math
            d_WO=Decoder.mmha.cache["H"].T @ d_input 

            d_H= d_input @ Decoder.mmha.Wo.T

            #add derivative
            Decoder.mmha.Wo = Decoder.mmha.Wo - d_WO * learningRate
            

            #split d_h into num_heads
            d_H_split= d_H.view(batch_size,context_size,Decoder.mmha.num_heads,Decoder.mmha.head_size).permute(0,2,1,3)

            d_Vw_split= Decoder.mmha.cache["H_Split"].transpose(-1,-2) @ d_H_split

            d_Q_K_Soft= (Decoder.mmha.Vw_split.transpose(-1,-2)) @ d_H_split

            d_S= Decoder.mmha.Q_K_Soft * (d_Q_K_Soft - (torch.sum(Decoder.mmha.Q_K_Soft * d_Q_K_Soft, dim=-1, keepdim=True)))

            d_Q_K= d_S / torch.sqrt(Decoder.mmha.head_size)

            #mask 
            mask = torch.tril(torch.ones(Decoder.mmha.seq_len, Decoder.mmha.seq_len, device=d_Q_K.device))
            d_Q_K_Masked = d_Q_K.masked_fill(mask == 0, 0)

            d_Qw_split= d_Q_K_Masked @ Decoder.mmha.Kw_split

            d_Kw_split= d_Q_K_Masked.transpose(-2,-1) @ Decoder.mmha.Qw_split

            #put all together
            d_Vw= d_Vw_split.permute(0,2,1,3).contiguous().view(batch_size, Decoder.mmha.seq_len, d_model)

            d_Qw= d_Qw_split.permute(0,2,1,3).contiguous().view(batch_size, Decoder.mmha.seq_len, d_model)

            d_Kw= d_Kw_split.permute(0,2,1,3).contiguous().view(batch_size, Decoder.mmha.seq_len, d_model)

            #weights
            d_Wq= Decoder.mmha.cache["Q"].T @ d_Qw

            d_Wk= Decoder.mmha.cache["K"].T @ d_Kw

            d_Wv= Decoder.mmha.cache["V"].T @ d_Vw


            #add derivative
            Decoder.mmha.Wq = Decoder.mmha.Wq - d_Wq * learningRate
            Decoder.mmha.Wk = Decoder.mmha.Wk - d_Wk * learningRate
            Decoder.mmha.Wv = Decoder.mmha.Wv - d_Wv * learningRate


            #inputs
            d_Q= d_Qw @ Decoder.mmha.Wq.T

            d_K= d_Kw @ Decoder.mmha.Wk.T

            d_V= d_Vw @ Decoder.mmha.Wv.T

            #all Values go to the output
            d_input=d_Q + d_K + d_V

            
            #9. norm
            d_beta=torch.sum(d_input,dim=(0,1))

            d_gamma=torch.sum(d_input * Decoder.norm1.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Decoder.norm1.Gamma

            #add derivative
            Decoder.norm1.Gamma = Decoder.norm1.Gamma - d_gamma * learningRate
            Decoder.norm1.beta = Decoder.norm1.beta - d_beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Decoder.norm1.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Decoder.norm1.cache["normalized_input"] * torch.sum((d_x_hat*Decoder.norm1.cache["normalized_input"]),dim=-1,keepdim=True))))


            #10. add
            d_input= d_bypass + d_x

            #this input goes into the next decoder till it reaches the end

        #update the token embedings

        Token_Embedings= torch.load("Token_Embeddings.pt")

        d_Token_Embeddings = torch.zeros_like(Token_Embedings)

        #tensor with ids of the input tokens
        ids_tensor = torch.tensor(Ids, device=d_input.device).view(-1)
        d_input_flat = d_input.view(-1, d_model)

        #set values of d_input on the Ids they come from
        d_Token_Embeddings.index_add_(0, ids_tensor, d_input_flat)

        #add the gradient to the Token_Embedings
        Token_Embedings = Token_Embedings - d_Token_Embeddings * learningRate

        #save change
        torch.save(Token_Embedings, "Token_Embeddings.pt")







        #add d_K_combine and d_V_combine to get the Encoder Input
        d_input = d_K_combine+ d_V_combine
        

        #4. Go through Encoder
        #same as decoder but without the mmha
        for Encoder in Encoder_list[::-1]:
            
            #1. add split
            d_into_ff=d_input
            d_bypass=d_input
            
            #2. Feed-Forward
            FF=Encoder.feedforward

            #count layers backwards
            #-1 because output not needed
            for layer_number in reversed(range(FF.layer_count-1)):

                #calculate gradients (calculations in FF_math)
                #d_into_ff = d_into_ff * relu_mask
                d_Weights=FF.cache["inputLayer"][layer_number].T @ (d_into_ff)   

                d_Biases=torch.sum(d_into_ff, dim=0)

                d_input=d_into_ff @ FF.weightList[layer_number].T


                #add derivative 
                FF.weightList[layer_number] = FF.weightList[layer_number] - d_Weights * learningRate
                FF.biasList[layer_number] = FF.biasList[layer_number] - d_Biases * learningRate
        

                
                #relu derivative 
                #all values who were over 1 go through rest doesnt

                if layer_number>0:
                    relu_mask=(FF.cache["inputLayer"][layer_number-1]>0).float()

                    d_into_ff=d_input @ relu_mask
                

            #3. norm
            d_beta=torch.sum(d_input,dim=(0,1))

            d_gamma=torch.sum(d_input * Encoder.norm2.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Encoder.norm2.Gamma

            #add derivative 
            Encoder.norm2.Gamma = Encoder.norm2.Gamma - d_gamma * learningRate
            Encoder.norm2.beta = Encoder.norm2.beta - d_beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Encoder.norm2.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Encoder.norm2.cache["normalized_input"] * torch.sum((d_x_hat*Encoder.norm2.cache["normalized_input"]),dim=-1,keepdim=True))))
            

            #4. add
            d_input=d_x + d_bypass

            #5.add split
            d_bypass=d_input

            #6. mha
            #all math functions under mha_math
            d_WO=Encoder.mha.cache["H"].T @ d_input 

            d_H= d_input @ Encoder.mha.Wo.T

            #add derivative
            Encoder.mha.Wo = Encoder.mha.Wo - d_WO * learningRate
            

            #split d_h into num_heads
            d_H_split= d_H.view(batch_size,context_size,Encoder.mha.num_heads,Encoder.mha.head_size).permute(0,2,1,3)

            d_Vw_split= Encoder.mha.cache["H_Split"].transpose(-1,-2) @ d_H_split

            d_Q_K_Soft= (Encoder.mha.Vw_split.transpose(-1,-2)) @ d_H_split

            d_S= Encoder.mha.Q_K_Soft * (d_Q_K_Soft - (torch.sum(Encoder.mha.Q_K_Soft * d_Q_K_Soft, dim=-1, keepdim=True)))

            d_Q_K= d_S / torch.sqrt(Encoder.mha.head_size)

            d_Qw_split= d_Q_K @ Encoder.mha.Kw_split

            d_Kw_split= d_Q_K.transpose(-2,-1) @ Encoder.mha.Qw_split

            #put all together
            d_Vw= d_Vw_split.permute(0,2,1,3).contiguous().view(batch_size, Encoder.mha.seq_len, d_model)

            d_Qw= d_Qw_split.permute(0,2,1,3).contiguous().view(batch_size, Encoder.mha.seq_len, d_model)

            d_Kw= d_Kw_split.permute(0,2,1,3).contiguous().view(batch_size, Encoder.mha.seq_len, d_model)

            #weights
            d_Wq= Encoder.mha.cache["Q"].T @ d_Qw

            d_Wk= Encoder.mha.cache["K"].T @ d_Kw

            d_Wv= Encoder.mha.cache["V"].T @ d_Vw


            #add derivative
            Encoder.mha.Wq = Encoder.mha.Wq - d_Wq * learningRate
            Encoder.mha.Wk = Encoder.mha.Wk - d_Wk * learningRate
            Encoder.mha.Wv = Encoder.mha.Wv - d_Wv * learningRate


            #inputs
            d_Q= d_Qw @ Encoder.mha.Wq.T

            d_K= d_Kw @ Encoder.mha.Wk.T

            d_V= d_Vw @ Encoder.mha.Wv.T

            #combine K and V because they go to the Encoder
            d_K_combine= d_K_combine + d_K

            d_V_combine= d_V_combine + d_V


            #d_Q goes through the decoder
            d_input=d_Q


            #7. norm
            d_beta=torch.sum(d_input,dim=(0,1))

            d_gamma=torch.sum(d_input * Encoder.norm1.cache["normalized_input"],dim=(0,1))

            d_x_hat=d_input * Encoder.norm1.Gamma

            #add derivative 
            Encoder.norm1.Gamma = Encoder.norm1.Gamma - d_gamma * learningRate
            Encoder.norm1.beta = Encoder.norm1.beta - d_beta * learningRate

            #just look in the folder norm_math for the formula
            d_x=(1/(d_model*torch.sqrt(Encoder.norm1.cache["variance"] + 1e-6))) * (
                (d_model * d_x_hat)
                 - (torch.sum(d_x_hat,dim=-1,keepdim=True)
                 -(Encoder.norm1.cache["normalized_input"] * torch.sum((d_x_hat*Encoder.norm1.cache["normalized_input"]),dim=-1,keepdim=True))))
            

            #7.add
            d_input=d_x + d_bypass


        #update the token embedings

        Token_Embedings= torch.load("Token_Embeddings.pt")

        d_Token_Embeddings = torch.zeros_like(Token_Embedings)

        #tensor with ids of the input tokens
        ids_tensor = torch.tensor(Ids, device=d_input.device).view(-1)
        d_input_flat = d_input.view(-1, d_model)

        #set values of d_input on the Ids they come from
        d_Token_Embeddings.index_add_(0, ids_tensor, d_input_flat)

        #add the gradient to the Token_Embedings
        Token_Embedings = Token_Embedings - d_Token_Embeddings * learningRate

        #save change
        torch.save(Token_Embedings, "Token_Embeddings.pt")
