#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;
from Transformer import PositionalEncoding, EncoderLayer;

class EmbeddingSimilarity(tf.keras.layers.Layer):

  def __init__(self, **kwargs):

    super(EmbeddingSimilarity, self).__init__(**kwargs);

  def build(self, input_shape):

    self.bias = self.add_weight(
      shape = (int(input_shape[1][0]),), # (vocab_size,)
      initializer = tf.zeros_initializer(),
      regularizer = tf.zeros_initializer(),
      constraint = tf.zeros_initializer()
    );

  def call(self, inputs):

    # embedding.shape = (batch, encode_length, embed_dim)
    # weights.shape = (vocab_size, embed_dim)
    embedding, weights = inputs;
    results = tf.linalg.matmul(embedding, tf.expand_dims(weights, axis = 0), transpose_b = True); # results.shape = (batch, encode_length, vocab_size)
    results = results + self.bias;
    results = tf.keras.layers.Softmax()(results); # results.shape = (batch, encode_length, vocab_size)
    return results;    

def BERT(vocab_size, num_layers = 12, embed_dim = 768, num_heads = 12, code_dim = 3072, dropout_rate = 0.1, training = True):

  # 1) inputs
  # NOTE: mask array element has 1 when the input token is for control purpose
  token = tf.keras.Input((None,), name = 'Token'); # token.shape = (batch, encode_length)
  segment = tf.keras.Input((None,), name = 'Segment'); # segment.shape = (batch, encode_length)
  mask = tf.keras.Input((None,), name = 'mask'); # mask.shape = (batch, encode_length)
  # 2) embedding
  embedding = tf.keras.layers.Embedding(vocab_size, embed_dim);
  token_embedding = embedding(token);
  segment_embedding = tf.keras.layers.Embedding(2, embed_dim)(segment);
  inputs = tf.keras.layers.Add()([token_embedding, segment_embedding]);
  inputs = PositionalEncoding(embed_dim)(inputs);
  # 3) encoder
  results = tf.keras.layers.Dropout(rate = dropout_rate)(inputs);
  results = tf.keras.layers.LayerNormalization()(results);
  reshaped_mask = tf.keras.layers.Reshape((1,1,-1))(mask);
  for i in range(num_layers):
    results = EncoderLayer(embed_dim, num_heads, code_dim, dropout_rate, tfa.layers.GELU())([results, reshaped_mask]); # results.shape = (batch, encode_length, embed_dim)
  # 4) outputs
  if training:
    masked_lm = tf.keras.layers.Dense(units = embed_dim, activation = tfa.layers.GELU())(results);
    masked_lm = tf.keras.layers.LayerNormalization()(masked_lm);                      # masked_lm.shape = (batch, encode_length, embed_dim)
    masked_lm = EmbeddingSimilarity()([masked_lm, embedding.embeddings]);             # masked_lm.shape = (batch, encode_length, vocab_size)
    next_sentence_prediction = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(results);  # next_sentence_prediction.shape = (batch, embed_dim)
    next_sentence_prediction = tf.keras.layers.Dense(units = embed_dim, activation = tf.math.tanh)(next_sentence_prediction);
    next_sentence_prediction = tf.keras.layers.Dense(units = 2, activation = tf.keras.layers.Softmax())(next_sentence_prediction);
    return tf.keras.Model(inputs = (token, segment, mask), outputs = results);
  return tf.keras.Model(inputs = (token, segment, mask), outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  bert = BERT(100);
  bert.save('bert.h5');
