import numpy as np

from B_Embeddings import Embeddings
from Encoder import Encoder
from D_DecoderLayer import DecoderLayer
from NNModule import NNModule

class Transformer(NNModule):
    def __init__(self, num_layers, ff_hidden_dim, num_heads, d_model, conv_hidden_dim, input_vocab_size,
               maximum_position_encoding, p=0.1, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.ff_hidden_dim = ff_hidden_dim
        self.num_heads = num_heads
        self.conv_hidden_dim = conv_hidden_dim
        self.p = p
        self.eps = eps

        # Full encoder first
        self.encoder = Encoder(num_layers, ff_hidden_dim, num_heads, d_model, conv_hidden_dim, input_vocab_size,
               maximum_position_encoding, p=0.1, eps=1e-6)

        # Embeddings for the decoder's input
        self.embedding = Embeddings(d_model, input_vocab_size, maximum_position_encoding, p)

        #self.enc_layer= EncoderLayer(ff_hidden_dim, num_heads, d_model, conv_hidden_dim, p, eps)
    
    def __createDecLayer(self, i):
        enc_layer = DecoderLayer(self.ff_hidden_dim, self.num_heads, self.d_model, 
            self.conv_hidden_dim, self.p, self.eps, layer_name="DecL%i"%i)
        return enc_layer
        
    def forward(self, X, Y):
        Henc = self.encoder.forward(X)

        Y, _, _ = self.embedding.forward(Y) # Transform to (batch_size, input_seq_length, d_model)

        for i in range(self.num_layers):
            X, A = self.__createDecLayer(i).forward(Henc, Y)
            self.store_attention_weights(A, layername="ATT%i"%i)
            #X = self.enc_layer.forward(X)

        return X  # (batch_size, input_seq_len, d_model)


if __name__ == '__main__':
    vocab_size = 20
    X = np.array([[0, 5, 3, 2, 1, 4, 3, 2, 9, 8]]) #word indexes
    X = np.squeeze(np.eye(vocab_size)[X.reshape(-1)]) #convert words to one-hot
    t_enc = Transformer(num_layers=2, ff_hidden_dim=3, num_heads=2, d_model=4, conv_hidden_dim=8, 
            input_vocab_size=vocab_size, maximum_position_encoding=10, p=0)

    def print_out(X):
        temp_out = t_enc.forward(X)
        print('Output is:', temp_out)
    print_out(X)
