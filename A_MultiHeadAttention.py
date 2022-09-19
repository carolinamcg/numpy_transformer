import numpy as np
import math
#from numpy_attention import softmax
from NNModule import NNModule

class MultiHeadAttention(NNModule):
    def __init__(self, hidden_size, num_heads, d_model, p, layer_name="EncL"):
        super().__init__()
        self.num_heads = num_heads #number of q, k, v per token x
        self.d = hidden_size #dimension of each query, value and key
        self.d_model = d_model #dimension of the input/output embeddings
        self.layer_name = layer_name

        # Weight matrices to compute Q, K, V from input embeddings (emb dim, hidden dim for each head)
        # each attention head has its own Wv, Wq, and Wk transforms.
        self.W_q, _ = self.get_parameters((d_model, self.d * num_heads), layer_name=layer_name +"_Q", bias=False)
        self.W_k, _ = self.get_parameters((d_model, self.d * num_heads), layer_name=layer_name +"_K", bias=False)
        self.W_v, _ = self.get_parameters((d_model, self.d * num_heads), layer_name=layer_name +"_V", bias=False)

        # Outputs of all sub-layers need to be of dimension self.d_model
        self.W_h, self.bias_h = self.get_parameters((self.num_heads * self.d, self.d_model), layer_name=layer_name +"_H")
        #converts each final value vector computed for each input token/query into 1 single vector w/ dimension=d_model=d_model --> weighted sum over all the heads per query/token

    def softmax(self, X):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(X - np.max(X))
        return e_x / e_x.sum()

    def split_heads(self, X):
        """
        Split the last dimension into (num_heads, hidden size): X -> (num_heads, bs, seq_length, self.d)
        Return after transpose to put in shape (bs, num_heads, seq_length, self.d)

        :param X (3D array): Q, K or V w/ shape (bs, seq_length, num_heads*self.d)
        """
        return np.array(np.split(X, self.num_heads, axis=-1)).transpose(1, 0, 2, 3)
    
    def get_attention_weights(self, Q, K, mask=None):
        """Computes the normalized attentions weights between each q and k,
        as a scaled dot product"""
        att_scores = np.matmul(
            Q, K.transpose(0, 1, 3, 2)
        )  #(batch size, num_heads, seq_length, seq_length)
        att_scores = att_scores / math.sqrt(self.d)

        if mask is not None: #for training decoder (not autorregresive)
            #bu only for self attention, not cross
            #mask = np.tril(np.ones(att_scores.shape[2:])) 
            att_scores[:,:, mask==0] = -math.inf #elements above the k-th diagonal = -inf
        
        att_weights = np.apply_along_axis(self.softmax, -1, att_scores) #-inf values -> 0
        return att_weights

    def get_values_attention(self, Q, K, V, mask=None):
        '''Get the linear combination of values for each query'''
        A = self.get_attention_weights(Q, K, mask=mask)
        H = np.matmul(A, V) #(bs, num_heads, seq_length, self.d)
        return H, A
    
    def forward(self, X, mask=None):
        bs, seq_length, _ = X.shape # (bs, number of tokens, emdedding_dimensions=d_model)

        #Compute all q, k and v for each word in X (seq_length)
        Q = np.matmul(X, self.W_q) #(bs, seq_length, self.d * number_heads)
        K = np.matmul(X, self.W_k) #contains every k: each token in each sequence has num_heads keys
        V = np.matmul(X, self.W_v)
        #Split q, k and v per head
        Q = self.split_heads(Q) #(bs, num_heads, seq_length, self.d)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Calculate the attention weights for each of the heads
        H_cat, A = self.get_values_attention(Q, K, V, mask=mask) #H_cat = (bs, num_heads, seq_length, self.d)

        # Merge heads outputs into the last array's dimension
        H_cat = H_cat.transpose(0,2,1,3).reshape(bs, seq_length, -1) #(bs, seq_length, self.d * number_heads)
        
        # Final linear layer  
        H = np.matmul(H_cat, self.W_h) + self.bias_h # (bs, seq_length, self.d_model)

        return H, A
    
    def forward_crossattention(self, Henc, X):
        '''
        :param X: output embeddings of the decoder's self attention + ADD,NORM module
        :param Henc: output embeddings from the encoder's final layer
        '''
        bs, seq_length, _ = X.shape # (number of tokens, emdedding_dimensions)

        #Compute 1 q, k and v for each word in X (seq_length) and for each head
        Q = np.matmul(X, self.W_q) #(num_heads, seq_length, self.d)
        K = np.matmul(Henc, self.W_k)
        V = np.matmul(Henc, self.W_v)
         #Split q, k and v per head
        Q = self.split_heads(Q) #(bs, num_heads, seq_length, self.d)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Calculate the attention weights for each of the heads
        H_cat, A = self.get_values_attention(Q, K, V, mask=None) #H_cat = (num_heads, seq_length, self.d)

        # Merge heads outputs into the last array's dimension
        H_cat = H_cat.transpose(0,2,1,3).reshape(bs, seq_length, -1) #(bs, seq_length, self.d * number_heads)
        
        # Final linear layer  
        H = np.matmul(H_cat, self.W_h) + self.bias_h # (seq_length, self.d_model)

        return H, A

    
if __name__ == '__main__':
    X = np.random.random((2,3,2))
    X[0, :, :] = np.zeros((3,2))
    X[1,2,:] = [0,0]
    temp_mha = MultiHeadAttention(hidden_size=1, num_heads=2, d_model=2, p=0)
    def print_out(X):
        temp_out, temp_attn = temp_mha.forward(X)
        print('Attention weights are:', temp_attn)
        print('Output is:', temp_out)
    print_out(X)