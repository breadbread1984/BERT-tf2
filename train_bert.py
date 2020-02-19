#!/usr/bin/python3

import tensorflow as tf;
from BERT import BERTTrainer;
from create_dataset import parse_function;

batch_size = 4096;

def main():

  bert_classifier = BERTTrainer(); # TODO vocab_size
  optimizer = tf.keras.optimizers.Adam();
  trainset = iter(tf.data.TFRecordDataset('trainset.tfrecord').map(parse_function).repeat().batch(batch_size).shuffle(batch_size));
  checkpoint = tf.train.Checkpoint(model = bert_classifier, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  log = tf.summary.create_file_writer('checkpoints');
  avg_loss1 = tf.keras.metrics.Mean(name = 'loss1', dtype = tf.float32);
  avg_loss2 = tf.keras.metrics.Mean(name = 'loss2', dtype = tf.float32);
  while True:
    token, segment, mask = next(trainset);
    masked_lm, next_sentence_prediction = bert_classifier([token, segment, mask]);
    # TODO:

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
