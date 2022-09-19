import numpy as np
from A_MultiHeadAttention import MultiHeadAttention
from B_Embeddings import LayerNormalization
from C_EncoderLayer import CNN
from NNModule import NNModule

class DecoderLayer(NNModule):
    def __init__(self, hidden_size, num_heads, d_model, conv_hidden_dim, p, eps=1e-6, layer_name="DecL"):
        super().__init__()
        #self.layer_name = layer_name
        self.mha = MultiHeadAttention(hidden_size, num_heads, d_model, p, layer_name=layer_name)
        self.cnn = CNN(d_model, conv_hidden_dim, layer_name=layer_name + "_CNN")

        self.layernorm1 = LayerNormalization(normal_shape=d_model, epsilon=1e-6, layer_name=layer_name + "_LN1")
        self.layernorm2 = LayerNormalization(normal_shape=d_model, epsilon=1e-6, layer_name=layer_name + "_LN2")
        self.layernorm3 = LayerNormalization(normal_shape=d_model, epsilon=1e-6, layer_name=layer_name + "_LN3")

    def forward(self, Henc, Y, mask=None):
        
        # Multi-head attention 
        attn_output, attn_weights = self.mha.forward(Y, mask)  # (seq_length, self.d_model)
        # Layer norm after adding the residual connection 
        out1 = self.layernorm1.forward(Y + attn_output)  # (seq_length, self.d_model)

        #CROSS-ATTENTION
        attn_output_cross, attn_weights_cross = self.mha.forward_crossattention(Henc, out1)  # (seq_length, self.d_model)
        out2 = self.layernorm2.forward(out1 + attn_output_cross)
        
        # Feed forward 
        cnn_output = self.cnn.forward(out2)  # (seq_length, self.d_model)
        
        #Second layer norm after adding residual connection 
        out3 = self.layernorm2.forward(out2 + cnn_output)  # (seq_length, self.d_model)

        return out3, [attn_weights, attn_weights_cross]


if __name__ == '__main__':
    X = np.array([[0, 10, 0, 0], [1,0,0,1]])
    enc = DecoderLayer(hidden_size=3, num_heads=2, d_model=4, conv_hidden_dim=8, p=0)

    def print_out(X):
        temp_out = enc.forward(X)
        print('Output is:', temp_out)
    print_out(X)
