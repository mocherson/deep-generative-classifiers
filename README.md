# deep generative classifiers

Usage
--------
1. Use the run_dgc.py script for training, and validation and test deep generative classifiers.
```
python run_dgc.py <options>
```
```
python run_dgc.py -h
```
```     
usage: run_dgc.py [-h] [--batch-size N] [--path PATH] [--epochs N] [--seed N] [--gpu N] [-e ENCODER] [-w N]
                  [--start-epoch N]

deep generative classifier for chestXray

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 16)
  --path PATH           data path
  --epochs N            number of epochs to train (default: 10)
  --seed N              random seed (default: 1)
  --gpu N               the GPU number (default auto schedule)
  -e ENCODER, --encoder ENCODER
                        the encoder
  -w N, --lossweight N  weight of KL divergence (default: 0)
  --start-epoch N       manual epoch number (useful on restarts)
```

results will be saved to 'models' directory under path

2. Use the run_baseline.py script for training, and validation and test deep generative classifiers.
```
python run_baseline.py <options>
```
```
python run_baseline.py -h
```
```     
usage: run_baseline.py [-h] [--batch-size N] [--path PATH] [--epochs N] [--seed N] [-e ENCODER] [--gpu N]
                       [--start-epoch N]

baseline classifiers for chestXray

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 16)
  --path PATH           data path
  --epochs N            number of epochs to train (default: 10)
  --seed N              random seed (default: 1)
  -e ENCODER, --encoder ENCODER
                        the encoder
  --gpu N               the GPU number (default auto schedule)
  --start-epoch N       manual epoch number (useful on restarts)
```
results will be saved to 'models/baselines' directory under path
