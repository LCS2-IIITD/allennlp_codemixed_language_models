# Language Models for Code-Mixed Data using Mulit Task Learning 

In this repo, we explore a few multi task learning for making language models of code-mix data. This code uses
 [AllenNLP](https://github.com/allenai/allennlp).

## Usage

### Environment Setup

A simple `pip install alllennlp` will do the trick.

### Training 

There are two training mechanisms for **multi-task models**:
1. use `loss(lm_en) + loss(lm_es) + loss(lm_cm)` for optimisation 
2. employ individuals optimisers for each loss function

For training using 1., we can simply use AllenNLP train command like: 
```
    allennlp train -f --include-package lm --serialization-dir <path_to_save> <path_to_training_config>
```

For training using 2., we have to utilise in `train.py` the repo. It's own arguments are exactly like `allennlp train`.  
```
     python3 train.py train -f --include-package lm --serialization-dir <path_to_save> <path_to_training_config>
```
In fact, we can use `train.py` to evaluate by using ` python3 train.py evaluate <rest of allennlp commands>`


## Folder Structure
- `/configs`: stores various configuration for various models.
- `/lm`:  is our language modelling code folder
    - `/models`: code for various models used by us
    - `/modules`: code of token embedder that can be used for sentiment analysis and a custom trainer
    - `/readers`: various dataset readers utilised by us
- `/hmtl` is a mini ripoff version of [Hierarchical Multi-Task Learning](https://github.com/huggingface/hmtl)
 for language modelling. It has it own configs, train and modules folder.
 
For training and utilising this code, you will be concerned with the config files only. 

## Models

**Baselines**
- base_lstm.jsonnet: Baseline vanilla language model that uses LSTM
- base_trans.jsonnet:  Baseline vanilla language model that uses Transformer  
    
**Proposed Models**
- catlstm.jsonnet: A multi-task model that uses representation from monolingual LSTM-based language models for Code-Mixed LSTM-based language models
- transtrans.jsonnet:  A multi-task model that uses representation from monolingual Transformer-based language models for Code-Mixed Transformer-based language models  
    
**Others**
- cm_sentiment_analysis.jsonnet: Doing sentiment analysis using the embeddings from the above.
- transtrans_test.jsonnet: For testing new things.

## Data

Tar version of the data is present in `data/` for language modelling and sentiment analysis.

For baselines models, the path to the required corpus can be specified using the following keys "train_data_path", "validation_data_path" and "test_data_path". 
The value of each key must be path to a text file. The text file should have lines.  

For proposed models, the value of each key is json like:
```
{
        "lang1": "path_to_lang1_corpus",
        "lang2": "path_to_lang2_corpus",
        "cm": "path_to_cm_corpus",
    }
```


## Contact

For any question, you can make an issue on the repo or contact Siddharth Yadav(@sedflix).


