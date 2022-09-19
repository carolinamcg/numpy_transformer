import numpy as np

from B_Embeddings import Embeddings
from C_EncoderLayer import EncoderLayer
from D_DecoderLayer import DecoderLayer
from NNModule import NNModule

class Transformer(NNModule):
    def __init__(self, num_layers, ff_hidden_dim, num_heads, d_model, conv_hidden_dim, input_vocab_size,
               maximum_position_encoding, p=0.1, eps=1e-6, mask_decoder=True, de_embedd=False):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.ff_hidden_dim = ff_hidden_dim
        self.num_heads = num_heads
        self.conv_hidden_dim = conv_hidden_dim
        self.p = p
        self.eps = eps

        self.mask_decoder = mask_decoder
        self.de_embedd = de_embedd #convert final output embedded vectors into the actual one-hot encoded tokens

        # Full encoder first
        #self.encoder = Encoder(num_layers, ff_hidden_dim, num_heads, d_model, conv_hidden_dim, input_vocab_size,
               #maximum_position_encoding, p=0.1, eps=1e-6)

        # Embeddings for the decoder's and encoder's input
        self.embedding = Embeddings(d_model, input_vocab_size, maximum_position_encoding, p)

        #self.enc_layer= EncoderLayer(ff_hidden_dim, num_heads, d_model, conv_hidden_dim, p, eps)
    
    def __createDecLayer(self, i):
        enc_layer = DecoderLayer(self.ff_hidden_dim, self.num_heads, self.d_model, 
            self.conv_hidden_dim, self.p, self.eps, layer_name="DecL%i"%i)
        return enc_layer
    
    def create_subsequent_maks(self, seq_length):
        mask = np.tril(np.ones((seq_length,seq_length))) 
        return mask
    
    def __createEncLayer(self, i):
        enc_layer = EncoderLayer(self.ff_hidden_dim, self.num_heads, self.d_model, 
            self.conv_hidden_dim, self.p, self.eps, layer_name="EncL%i"%i)
        return enc_layer
        
    def forward(self, X, Y):
        # X.shape = (bs, seq_length, input features)
        X, _, _ = self.embedding.forward(X) # Transform to (batch_size, input_seq_length, d_model)

        for i in range(self.num_layers):
            Henc, A = self.__createEncLayer(i).forward(X)
            self.store_attention_weights(A, layername="ATT%i"%i)

        Y, _, _ = self.embedding.forward(Y) #same embedding weights used for the encoder's input

        if self.mask_decoder:
            mask = self.create_subsequent_maks(Y.shape[1])

        for i in range(self.num_layers):
            Y, A = self.__createDecLayer(i).forward(Henc, Y, mask)
            self.store_attention_weights(A, layername="ATT%i"%i)
            #X = self.enc_layer.forward(X)

        if self.de_embedd:
            X = self.embedding.de_embedd(X)

        return Y # (batch_size, input_seq_len, d_model)


if __name__ == '__main__':
    vocab_size = 20
    X = np.array([0, 5, 3, 2, 1, 4, 3, 2, 9, 8]) #word indexes
    #X = np.squeeze(np.eye(vocab_size)[X.reshape(-1)]) #convert words to one-hot
    X = np.array([X, X])
    t = Transformer(num_layers=2, ff_hidden_dim=3, num_heads=2, d_model=4, conv_hidden_dim=8, 
            input_vocab_size=vocab_size, maximum_position_encoding=10, p=0)

    def print_out(X):
        temp_out = t.forward(X)
        print('Output is:', temp_out)
    print_out(X)
