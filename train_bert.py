#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from math import ceil;
import tensorflow as tf;
from BERT import BERTClassifier;
from create_dataset import tokenizer, parse_function_generator;

max_seq_len = 128

def train_AFQMC():

  batch_size = 64;
  trainset_size = 34334;
  bert_classifier = BERTClassifier(len(tokenizer.vocab));
  optimizer = tf.keras.optimizers.Adam(2e-5);
  # load dataset
  trainset = iter(tf.data.TFRecordDataset(join('datasets', 'afqmc_public','trainset.tfrecord')).map(parse_function_generator(max_seq_len)).repeat().batch(batch_size).shuffle(batch_size));
  validateset = iter(tf.data.TFRecordDataset(join('datasets', 'afqmc_public','trainset.tfrecord')).map(parse_function_generator(max_seq_len)).repeat().batch(batch_size).shuffle(batch_size));
  # restore from existing checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = bert_classifier, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoint'));
  # create log
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  while True:
    (input_ids, segment_ids), label_ids = next(trainset);
    with tf.GradientTape() as tape:
      results = bert_classifier([input_ids, segment_ids]);
      loss = tf.keras.losses.SparseCategoricalCrossentrypy(from_logits = False)(label_ids, results);
    avg_loss.update_state(loss);
    # write_log
    if tf.equal(optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
      print('Step #%d Loss: %,6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    grads = tape.gradient(loss, bert_classifer.trainable_variables);
    optimizer.apply_gradients(zip(grads, bert_classifer.trainable_variables));
    # save model
    if tf.equal(optimizer.iterations % 100, 0):
      checkpoint.save(join('checkpoints','ckpt'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  train_AFQMC();
