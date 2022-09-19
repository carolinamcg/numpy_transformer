import numpy as np
import matplotlib.pyplot as plt

from NNModule import NNModule

class LayerNormalization(NNModule):
    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-10, layer_name="LayerNorm"):
        """Layer normalization layer
            See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
            https://github.com/CyberZHG/torch-layer-normalization/blob/89f405b60f53f85da6f03fe685c190ef394ce50c/torch_layer_normalization/layer_normalization.py#L8

            :param normal_shape: The shape of the input tensor or the last dimension of the input tensor (dimension over which LayerNorm is applied)
            :param gamma: Add a scale parameter if it is True.
            :param beta: Add an offset parameter if it is True.
            :param epsilon: Epsilon for calculating variance. """
        super().__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)

        self.epsilon = epsilon
        if gamma:
            self.gamma, _ = self.get_parameters(normal_shape, layer_name=layer_name + "_gamma", bias=False) #(hidden_size of this layer's input, 1)
        if beta:
            self.beta, _ = self.get_parameters(normal_shape, layer_name=layer_name + "_beta", bias=False) #(hidden_size, 1)

    def forward(self, X):
        mean = X.mean(axis=-1, keepdims=True) # mean of each columns/dimension/hidden unit
        var = ((X - mean) ** 2).mean(axis=-1, keepdims=True)
        std = np.sqrt(var + self.epsilon)
        y = (X - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y








class Embeddings(NNModule):
    def __init__(self, d_model, vocab_size, max_position_embeddings, p, layer_name="Emb"):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embeddings = self.Embedding(vocab_size, d_model, padding_idx=1, layer_name="W_"+layer_name)
        #self.position_embeddings = self.Embedding(max_position_embeddings, d_model)
        self.position_embeddings = self.create_sinusoidal_embeddings(
            nb_p=max_position_embeddings, dim=d_model #, E=self.position_embeddings
        ) #(max_position_embeddings, d_model)

        self.LayerNorm = LayerNormalization(normal_shape=d_model, epsilon=1e-12)

    def Embedding(self, N, emb_dim, padding_idx=1, layer_name="Emb"):
        """
        Creates a matrix with all possible embeddings: 1 embedding of dim=d_model for each possible word
        :param N: total number of existinng words/tokens/positions in the current task/problem
        :param emb_dim: dimension of each embedding vector (for each word/token)

        :output w_emb (N, emb_dim): embeddings weights array. Transforms each one-hot vector/word into the corresponding embedding, by the index set to 1

        """
        w_emb, _ = self.get_parameters((N, emb_dim), layer_name=layer_name, bias=False)
        #w_emb = np.random.random((N, emb_dim))
        return w_emb
    
    def one_hot(self, num_classes, ids):
        '''
        Converts a vector with all possible indexes/classes in one-hot encoded matrix
        (len(ids), num_classes)
        '''
        return np.squeeze(np.eye(num_classes)[ids.reshape(-1)])

    def create_sinusoidal_embeddings(self, nb_p, dim):
        E = np.random.random((nb_p, dim))
        theta = np.array(
            [
                [p / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for p in range(nb_p)
            ]
        )
        E[:, 0::2] = np.sin(theta[:, 0::2]) #(max_position_embeddings, d_model)
        E[:, 1::2] = np.cos(theta[:, 1::2]) #odd indexes in the emb dimension = cos

        return E

    def softmax(self, X):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(X - np.max(X))
        return e_x / e_x.sum()

    def forward(self, indexed_tokens):
        '''
        :param one_hot_inputs: (seq_length, vocab_size) array. eaxh row corresponds 
        to a token/word and the column idx=1 identifies which word from the total vocabulary/dictionary is
        '''
        bs, seq_length = indexed_tokens.shape #one_hot_inputs.shape[-2]
        one_hot_inputs = np.array([self.one_hot(self.vocab_size, ids) for ids in indexed_tokens])
        position_ids = self.one_hot(seq_length, np.arange(seq_length)) # (max_seq_length, max_seq_length)
                # marks positions inside each sequence

        # Get word embeddings for each input token (one_hot_inputs = ())
        word_embeddings = np.matmul(one_hot_inputs, self.word_embeddings) # (max_seq_length, d_model)

        # Get position embeddings for each position id
        position_embeddings = np.matmul(position_ids, self.position_embeddings) # (max_seq_length, d_model)

        # Add them both
        embeddings = word_embeddings + position_embeddings  # (max_seq_length, d_model)

        # Layer norm
        embeddings = self.LayerNorm.forward(embeddings)  # (max_seq_length, d_model)
        
        return embeddings, word_embeddings, position_embeddings

    def de_embedd(self, X):
        ''' Final linear layer + softmax to convert the embedded vectors to vocabulary words'''
        words_probabilities = np.matmul(X, self.word_embeddings.transpose(1,0))
        one_hot_words = np.apply_along_axis(self.softmax, -1, words_probabilities) 

        return one_hot_words

if __name__ == '__main__':
    vocab_size = 20
    X = np.array([0, 5, 3, 2, 1, 4, 3, 2, 9, 8]) #word indexes
    #X = np.squeeze(np.eye(vocab_size)[X.reshape(-1)]) #convert words to one-hot
    X = np.array([X, X])
    bs = 2
    emb = Embeddings(d_model=64, vocab_size=20, max_position_embeddings=10, p=0)

    def print_out(X, see=True):
        emb_out, emb_w, emb_p = emb.forward(X)
        print('Output is:', emb_out)

        if see:
            _,b= plt.subplots(bs, 3, figsize=(20,30))
            for i in range(bs):
                b[i, 0].set_title("Word Embeddings")
                b[i, 0].set_ylabel("Word/Token Position")
                b[i, 0].set_xlabel("Embedding Dimension")
                b[i, 0].imshow(emb_w[i], interpolation='none')
                b[i, 1].set_title("Position Embeddings")
                b[i, 1].set_xlabel("Embedding Dimension")
                b[i, 1].imshow(emb_p, interpolation='none')
                b[i, 2].set_title("Input Embeddings")
                b[i, 2].set_xlabel("Embedding Dimension")
                b[i, 2].imshow(emb_out[i], interpolation='none')
            plt.show()

    print_out(X)