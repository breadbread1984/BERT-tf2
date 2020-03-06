#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from math import ceil;
import tensorflow as tf;
from BERT import BERTClassifier;
from create_dataset import tokenizer, parse_function_generator;

max_seq_len = 128

def train_AFQMC():

  batch_size = 256;
  trainset_size = 34334;
  bert_classifier = BERTClassifier(len(tokenizer.vocab));
  bert_classifier.compile(optimizer = tf.keras.optimizers.Adam(2e-5), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')]);
  trainset = tf.data.TFRecordDataset(join('datasets', 'afqmc_public','trainset.tfrecord')).map(parse_function_generator(max_seq_len)).repeat().batch(batch_size).shuffle(batch_size);
  validateset = tf.data.TFRecordDataset(join('datasets', 'afqmc_public','trainset.tfrecord')).map(parse_function_generator(max_seq_len)).repeat().batch(batch_size).shuffle(batch_size);
  bert_classifier.fit(trainset, validation_data = validateset, validation_steps = 1, epochs = 10, steps_per_epoch = ceil(trainset_size / batch_size));
  if False == exists('models'): mkdir('models');
  bert_classifer.save(join('models', 'bert_classifier.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  train_AFQMC();
