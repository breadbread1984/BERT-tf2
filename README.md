# BERT-tf2
Thie project implements BERT with tensorflow 2

NOTE: the project adopts GELU activation from tf-addons. tf-addons only support the latest tensorflow stable version. don't use nightly build of tensorflow!!!

## how to download and create dataset
download and create datasets with command

```bash
python3 create_dataset.py <dataset name>
```

dataset should be one among 'AFQMC','TNEWS','IFLYTEK','CMNLI','WSCWinograd','CSL','CMRC','DRCD','CHID','C3' and 'CLUE'.

## how to train bert
