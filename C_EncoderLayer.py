import numpy as np
from A_MultiHeadAttention import MultiHeadAttention
from B_Embeddings import LayerNormalization
from NNModule import NNModule

class CNN(NNModule):
    def __init__(self, d_model, conv_hidden_dim, layer_name="CNN"):
        super().__init__()
        self.k1convL1, self.bias1 = self.get_parameters((d_model, conv_hidden_dim), layer_name=layer_name)
        self.k1convL2, self.bias2 = self.get_parameters((conv_hidden_dim, d_model), layer_name=layer_name)

    def ReLU(self, X):
        return np.maximum(0,X)

    def forward(self, X):
        X = np.matmul(X, self.k1convL1) + self.bias1
        X = self.ReLU(X)
        X = np.matmul(X, self.k1convL2) + self.bias2
        return X

class EncoderLayer(NNModule):
    def __init__(self, hidden_size, num_heads, d_model, conv_hidden_dim, p, eps=1e-6, layer_name="EncL"):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_heads, d_model, p, layer_name=layer_name)
        self.cnn = CNN(d_model, conv_hidden_dim, layer_name=layer_name + "_CNN")

        self.layernorm1 = LayerNormalization(normal_shape=d_model, epsilon=1e-6, layer_name=layer_name + "_LN1")
        self.layernorm2 = LayerNormalization(normal_shape=d_model, epsilon=1e-6, layer_name=layer_name + "_LN2")

    def forward(self, X):
        
        # Multi-head attention 
        attn_output, attn_weights = self.mha.forward(X)  # (seq_length, self.d_model)

        # Layer norm after adding the residual connection 
        out1 = self.layernorm1.forward(X + attn_output)  # (seq_length, self.d_model)
        
        # Feed forward 
        cnn_output = self.cnn.forward(out1)  # (seq_length, self.d_model)
        
        #Second layer norm after adding residual connection 
        out2 = self.layernorm2.forward(out1 + cnn_output)  # (seq_length, self.d_model)

        return out2, attn_weights


if __name__ == '__main__':
    X = np.array([ [[0, 10, 0, 0], [1,0,0,1]], [[0, 0, 1, 0], [0,0,0,1]]])
    enc = EncoderLayer(hidden_size=3, num_heads=2, d_model=4, conv_hidden_dim=8, p=0)

    def print_out(X):
        temp_out = enc.forward(X)
        print('Output is:', temp_out)
    print_out(X)
