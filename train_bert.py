#!/usr/bin/python3

import sys;
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
  bert_classifier.compile(optimizer = tf.keras.optimizers.Adam(2e-5), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')]);
  trainset = tf.data.TFRecordDataset(join('datasets', 'afqmc_public','trainset.tfrecord')).map(parse_function_generator(max_seq_len)).repeat().batch(batch_size).shuffle(batch_size);
  validateset = tf.data.TFRecordDataset(join('datasets', 'afqmc_public','trainset.tfrecord')).map(parse_function_generator(max_seq_len)).repeat().batch(batch_size).shuffle(batch_size);
  bert_classifier.fit(trainset, validation_data = validateset, validation_steps = 1, epochs = 10, steps_per_epoch = ceil(trainset_size / batch_size));
  if False == exists('models'): mkdir('models');
  bert_classifer.save(join('models', 'afqmc.h5'));

def train_TNEWS():

  batch_size = 64;
  trainset_size = 34334;
  bert_classifier = BERTClassifier(len(tokenizer.vocab));
  bert_classifier.compile(optimizer = tf.keras.optimizers.Adam(2e-5), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')]);
  trainset = tf.data.TFRecordDataset(join('datasets', 'tnews_public','trainset.tfrecord')).map(parse_function_generator(max_seq_len)).repeat().batch(batch_size).shuffle(batch_size);
  validateset = tf.data.TFRecordDataset(join('datasets', 'tnews_public','trainset.tfrecord')).map(parse_function_generator(max_seq_len)).repeat().batch(batch_size).shuffle(batch_size);
  bert_classifier.fit(trainset, validation_data = validateset, validation_steps = 1, epochs = 10, steps_per_epoch = ceil(trainset_size / batch_size));
  if False == exists('models'): mkdir('models');
  bert_classifer.save(join('models', 'tnews.h5'));  

def train_IFLYTEK():
  pass;

def train_CMNLI():
  pass;

def train_WSCWinograd():
  pass;

def train_CSL():
  pass;

def train_CMRC():
  pass;

def train_DRCD():
  pass;

def train_CHID():
  pass;

def train_C3():
  pass;

def train_CLUE():
  pass;

if __name__ == "__main__":

  datasets = ['AFQMC','TNEWS','IFLYTEK','CMNLI','WSCWinograd','CSL','CMRC','DRCD','CHID','C3','CLUE'];
  assert tf.executing_eagerly();
  if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + " <dataset name>");
    exit();
  if sys.argv[1] not in datasets:
    print('wrong dataset name, the dataset name should be among:');
    print(datasets);
  if sys.argv[1] == 'AFQMC':
    train_AFQMC();
  elif sys.argv[1] == 'TNEWS':
    train_TNEWS();
  elif sys.argv[1] == 'IFLYTEK':
    train_IFLYTEK();
  elif sys.argv[1] == 'CMNLI':
    train_CMNLI();
  elif sys.argv[1] == 'WSCWinograd':
    train_WSCWinograd();
  elif sys.argv[1] == 'CSL':
    train_CSL();
  elif sys.argv[1] == 'CMRC':
    train_CMRC();
  elif sys.argv[1] == 'DRCD':
    train_DRCD();
  elif sys.argv[1] == 'CHID':
    train_CHID();
  elif sys.argv[1] == 'C3':
    train_C3();
  elif sys.argv[1] == 'CLUE':
    train_CLUE();
