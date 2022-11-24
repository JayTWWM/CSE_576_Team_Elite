import tensorflow as tf
from multihead_attention import *

class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()
    self.mha = MultiHeadAttention(head_size=units, num_heads=1, **kwargs)
    self.layernorm = tf.keras.layers.BatchNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):
    # shape_checker = ShapeChecker()

    # shape_checker(x, 'batch t units')
    # shape_checker(context, 'batch s units')

    attn_output, attn_scores = self.mha(
        query=x,
        value=context,
        return_attention_scores=True)

    # shape_checker(x, 'batch t units')
    # shape_checker(attn_scores, 'batch heads t s')

    # Cache the attention scores for plotting later.
    attn_scores = tf.reduce_mean(attn_scores, axis=1)
    # shape_checker(attn_scores, 'batch t s')
    self.last_attention_weights = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x