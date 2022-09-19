# numpy_transformer

Building a Transformer model from scratch with numpy. An easy explanation of this architecture.

- NNModule.py is an inherited class to store the parameters for each layer.
- Encoder builds an encoder with multiple encoder layers and attention heads, calling for that the Embeddings, MultiHeadAttention and EncoderLayer classes.
- Transformer builds a full model, with encoder and decoder blocks.
