#!/usr/bin/python3

import sys;
from os import mkdir;
from os.path import exists, join;
import wget;
import zipfile;
import json;
import tensorflow as tf;
from tokenization import FullTokenizer;

tokenizer = FullTokenizer(vocab_file = join("bert_chinese", "vocab.txt"), do_lower_case = False);

def preprocess(s1, s2 = '', max_seq_len = 128):
  tokens1 = tokenizer.tokenize(s1);
  tokens2 = tokenizer.tokenize(s2);
  while True:
    total_length = len(tokens1) + len(tokens2);
    if total_length <= max_seq_len - 3: break;
    if len(tokens1) > len(tokens2): tokens1.pop();
    else: tokens2.pop();
  tokens = list();
  segment_ids = list();
  tokens.append('[CLS]');
  segment_ids.append(0);
  for token in tokens1:
    tokens.append(token);
    segment_ids.append(0);
  tokens.append('[SEP]');
  segment_ids.append(0);
  for token in tokens2:
    tokens.append(token);
    segment_ids.append(1);
  tokens.append('[SEP]');
  segment_ids.append(1);
  input_ids = tokenizer.convert_tokens_to_ids(tokens);
  # NOTE: my implementation uses mask = 1 to ignore padding token
  input_mask = [0,] * len(input_ids);
  while len(input_ids) < max_seq_len:
    input_ids.append(0);
    input_mask.append(1);
    segment_ids.append(0);
  assert len(input_ids) == max_seq_len;
  assert len(input_mask) == max_seq_len;
  assert len(segment_ids) == max_seq_len;
  return input_ids, input_mask, segment_ids;

def parse_function_generator(max_seq_len = 128):
  def parse_function(serialized_example):
    feature = tf.io.parse_single_example(
      serialized_example,
      features = {
        'input_ids': tf.io.FixedLenFeature((max_seq_len), dtype = tf.int64),
        'input_mask': tf.io.FixedLenFeature((max_seq_len), dtype = tf.int64),
        'segment_ids': tf.io.FixedLenFeature((max_seq_len), dtype = tf.int64),
        'label_ids': tf.io.FixedLenFeature((), dtype = tf.int64)
      }
    );
    for name in list(feature.keys()):
      feature[name] = tf.cast(feature[name], dtype = tf.int32);
    return (feature['input_ids'], feature['segment_ids']), feature['label_ids'];
  return parse_function;

def create_AFQMC(max_seq_len = 128):
  # 1) download and extract dataset
  if False == exists('tmp'): mkdir('tmp');
  if False == exists(join('tmp','afqmc_public.zip')):
    filename = wget.download('https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip', out = 'tmp');
  else:
    filename = join('tmp','afqmc_public.zip');
  zip_file = zipfile.ZipFile(filename, 'r');
  if False == exists(join('tmp','afqmc_public')): mkdir(join('tmp','afqmc_public'));
  zip_file.extractall(join('tmp','afqmc_public'));
  zip_file.close();
  # 2) create dateset
  if False == exists('datasets'): mkdir('datasets');
  if False == exists(join('datasets', 'afqmc_public')): mkdir(join('datasets', 'afqmc_public'));
  # trainset
  writer = tf.io.TFRecordWriter(join('datasets', 'afqmc_public','trainset.tfrecord'));
  f = open(join('tmp','afqmc_public','train.json'), 'r');
  lines = f.readlines();
  for line in lines:
    sample = json.loads(line);
    s1 = sample['sentence1'];
    s2 = sample['sentence2'];
    label = int(sample['label']);
    input_ids, input_mask, segment_ids = preprocess(s1, s2, max_seq_len);
    tf_example = tf.train.Example(features = tf.train.Features(
      feature = {
        "input_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_ids))),
        "input_mask": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_mask))),
        "segment_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(segment_ids))),
        "label_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = [label,]))
      }
    ));
    writer.write(tf_example.SerializeToString());
  f.close();
  writer.close();
  # validateset
  writer = tf.io.TFRecordWriter(join('datasets', 'afqmc_public', 'validateset.tfrecord'));
  f = open(join('tmp','afqmc_public','dev.json'), 'r');
  lines = f.readlines();
  for line in lines:
    sample = json.loads(line);
    s1 = sample['sentence1'];
    s2 = sample['sentence2'];
    label = int(sample['label']);
    input_ids, input_mask, segment_ids = preprocess(s1, s2, max_seq_len);
    tf_example = tf.train.Example(features = tf.train.Features(
      feature = {
        "input_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_ids))),
        "input_mask": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_mask))),
        "segment_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(segment_ids))),
        "label_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = [label,]))
      }
    ));
    writer.write(tf_example.SerializeToString());
  f.close();
  writer.close();
  # testset
  writer = tf.io.TFRecordWriter(join('datasets', 'afqmc_public', 'testset.tfrecord'));
  f = open(join('tmp','afqmc_public','test.json'), 'r');
  lines = f.readlines();
  for line in lines:
    sample = json.loads(line);
    s1 = sample['sentence1'];
    s2 = sample['sentence2'];
    id = int(sample['id']);
    input_ids, input_mask, segment_ids = preprocess(s1, s2, max_seq_len);
    tf_example = tf.train.Example(features = tf.train.Features(
      feature = {
        "input_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_ids))),
        "input_mask": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_mask))),
        "segment_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(segment_ids))),
        "id": tf.train.Feature(int64_list = tf.train.Int64List(value = [id,]))
      }
    ));
    writer.write(tf_example.SerializeToString());
  f.close();
  writer.close();

def create_TNEWS(max_seq_len = 128):
  labels = {'100': 0, '101': 1, '102': 2, '103': 3, '104': 4, '106': 5, '107': 6, '108': 7, '109': 8, '110': 9, '112': 10, '113': 11, '114': 12, '115': 13, '116': 14};
  # 1) download and extract dataset
  if False == exists('tmp'): mkdir('tmp');
  if False == exists(join('tmp','tnews_public.zip')):
    filename = wget.download('https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip', out = 'tmp');
  else:
    filename = join('tmp','tnews_public.zip');
  zip_file = zipfile.ZipFile(filename, 'r');
  if False == exists(join('tmp','tnews_public')): mkdir(join('tmp','tnews_public'));
  zip_file.extractall(join('tmp','tnews_public'));
  zip_file.close();
  # 2) create dataset
  if False == exists('datasets'): mkdir('datasets');
  if False == exists(join('datasets', 'tnews_public')): mkdir(join('datasets', 'tnews_public'));
  # trainset
  writer = tf.io.TFRecordWriter(join('datasets', 'tnews_public', 'trainset.tfrecord'));
  f = open(join('tmp','tnews_public','train.json'), 'r');
  lines = f.readlines();
  for line in lines:
    sample = json.loads(line);
    s = sample['sentence'];
    label = labels[sample['label']];
    input_ids, input_mask, segment_ids = preprocess(s, '', max_seq_len);
    tf_example = tf.train.Example(features = tf.train.Features(
      feature = {
        "input_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_ids))),
        "input_mask": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_mask))),
        "segment_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(segment_ids))),
        "label_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = [label,]))
      }
    ));
    writer.write(tf_example.SerializeToString());
  f.close();
  writer.close();
  # validateset
  writer = tf.io.TFRecordWriter(join('datasets', 'tnews_public', 'validateset.tfrecord'));
  f = open(join('tmp','tnews_public','dev.json'), 'r');
  lines = f.readlines();
  for line in lines:
    sample = json.loads(line);
    s = sample['sentence'];
    label = labels[sample['label']];
    input_ids, input_mask, segment_ids = preprocess(s, '', max_seq_len);
    tf_example = tf.train.Example(features = tf.train.Features(
      feature = {
        "input_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_ids))),
        "input_mask": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_mask))),
        "segment_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(segment_ids))),
        "label_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = [label,]))
      }
    ));
    writer.write(tf_example.SerializeToString());
  f.close();
  writer.close();
  # testset
  writer = tf.io.TFRecordWriter(join('datasets', 'tnews_public', 'testset.tfrecord'));
  f = open(join('tmp','tnews_public','test.json'), 'r');
  lines = f.readlines();
  for line in lines:
    sample = json.loads(line);
    s = sample['sentence'];
    id = sample['id'];
    input_ids, input_mask, segment_ids = preprocess(s, '', max_seq_len);
    tf_example = tf.train.Example(features = tf.train.Features(
      feature = {
        "input_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_ids))),
        "input_mask": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_mask))),
        "segment_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(segment_ids))),
        "id": tf.train.Feature(int64_list = tf.train.Int64List(value = [id,]))
      }
    ));
    writer.write(tf_example.SerializeToString());
  f.close();
  writer.close();

def create_IFLYTEK(max_seq_len = 128):
  pass;

def create_CMNLI(max_seq_len = 128):
  pass;

def create_WSCWinograd(max_seq_len = 128):
  pass;

def create_CSL(max_seq_len = 128):
  pass;

def create_CMRC(max_seq_len = 128):
  pass;

def create_DRCD(max_seq_len = 128):
  pass;

def create_CHID(max_seq_len = 128):
  pass;

def create_C3(max_seq_len = 128):
  pass;

def create_CLUE(max_seq_len = 128):
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
    create_AFQMC();
  elif sys.argv[1] == 'TNEWS':
    create_TNEWS();
  elif sys.argv[1] == 'IFLYTEK':
    create_IFLYTEK();
  elif sys.argv[1] == 'CMNLI':
    create_CMNLI();
  elif sys.argv[1] == 'WSCWinograd':
    create_WSCWinograd();
  elif sys.argv[1] == 'CSL':
    create_CSL();
  elif sys.argv[1] == 'CMRC':
    create_CMRC();
  elif sys.argv[1] == 'DRCD':
    create_DRCD();
  elif sys.argv[1] == 'CHID':
    create_CHID();
  elif sys.argv[1] == 'C3':
    create_C3();
  elif sys.argv[1] == 'CLUE':
    create_CLUE();
