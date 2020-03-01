#!/usr/bin/python3

import sys;
from os import mkdir;
from os.path import exists;
import wget;
import zipfile;
import tensorflow as tf;

def create_AFQMC():
  # 1) download and extract dataset
  if False == exists('tmp'): mkdir('tmp');
  if False == exists('tmp/afqmc_public.zip'):
    filename = wget.download('https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip', out = 'tmp');
  else:
    filename = 'tmp/afqmc_public.zip';
  zip_file = zipfile.ZipFile(filename, 'r');
  if False == exists('tmp/afqmc_public'): mkdir('tmp/afqmc_public');
  zip_file.extractall('tmp/afqmc_public');
  zip_file.close();
  # 2)

def create_TNEWS():
  pass;

def create_IFLYTEK():
  pass;

def create_CMNLI():
  pass;

def create_WSCWinograd():
  pass;

def create_CSL():
  pass;

def create_CMRC():
  pass;

def create_DRCD():
  pass;

def create_CHID():
  pass;

def create_C3():
  pass;

def create_CLUE():
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
