import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, units):
    super(Encoder, self).__init__()
    # self.text_processor = text_processor
    self.vocab_size = 20000
    self.units = units

    # The embedding layer converts tokens to vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
                                               mask_zero=True)

    # The RNN layer processes those vectors sequentially.
    self.rnn = tf.keras.layers.Bidirectional(
        merge_mode='sum',
        layer=tf.keras.layers.GRU(units,
                            # Return the sequence and state
                            return_sequences=True,
                            recurrent_initializer='glorot_uniform'))

  def call(self, x):
    # shape_checker = ShapeChecker()
    # shape_checker(x, 'batch s')

    # 2. The embedding layer looks up the embedding vector for each token.
    x = self.embedding(x)
    # shape_checker(x, 'batch s units')

    # 3. The GRU processes the sequence of embeddings.
    x = self.rnn(x)
    # shape_checker(x, 'batch s units')

    # 4. Returns the new sequence of embeddings.
    return x

#   def convert_input(self, texts):
#     texts = tf.convert_to_tensor(texts)
#     if len(texts.shape) == 0:
#       texts = tf.convert_to_tensor(texts)[tf.newaxis]
#     context = self.text_processor(texts).to_tensor()
#     context = self(context)
#     return context