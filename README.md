# MEDIC

Please skip to our new repository [MEDIC++](https://github.com/zzwdx/MEDIC++) for faster execution speed and higher performance.

### 1. Introduction

This repository contains the implementation of the paper **Generalizable Decision Boundaries: Dualistic Meta-Learning for Open Set Domain Generalization**: 

```
# PACS
Known classes: ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
Unknown classes: ['person']
```

### 2. Dataset Construction

The dataset needs to be divided into two folders for training and validation. We provide reference code for automatically dividing data using official split in `data_list/split_kfold.py`.

```python
root_dir = "path/to/PACS"
instr_dir = "path/to/PACS_data_list"
```

### 3. Train

To run the training code, please update the path of the dataset in `ml_open.py`:

```python
if dataset == 'PACS':	
    train_dir = 'path/to/PACS_train' # the folder of training data 
	val_dir = 'path/to/PACS_val' # the folder of validation data 
	test_dir = 'path/to/PACS_all' or ['path/to/PACS_train', 'path/to/PACS_val']
```

then simply run:

```python
python ml_open.py --source-domain ... --target-domain ... --save-name ... --gpu 0
```

### 4. Evalution

To run the evaluation code, please update the path of the dataset in `eval.py`:

```python
if dataset == 'PACS':
        root_dir = 'path/to/PACS_all' or ['path/to/PACS_train', 'path/to/PACS_val']
```

then simply run:

```
python eval.py --save-name ... --gpu 0
```

