# Polarization and political ideology detection using neural models and heteregeneous knowledge :speech_balloon:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

The inference of politically-charged information from text data is a popular research topic in Natural Language Processing (NLP) at both text- and author-level. This repository contains code that enabled experiments upon the afore mentioned topic. Content will be organized and detailed in a future paper.


## Model description
The resulting model, hereby called hybrid model, takes as an input parallel word embeddings, syntactic n-gramsand and LIWC-MRC (psycho) vectors. This model consists of a word-based CNN with two convolution channels (conv1 and conv2) with filters of size 2 and 3 with a mapping of size = 64 followed by a Batch Normalization and a MaxPooling layer with a 50% dropout. LIWC features are fed into a convolution with filter size = 1 andmapping size = 64. This is followed by a Batch Normalization and a MaxPooling layers, and then concatenated with conv1 and conv2. The sngram-psycho layer is concatenated with the two convolutions (conv1 ⊕ conv2 ⊕ liwc layer) with a 50% dropout layer, and finally fed into a softmax output layer.


## Organization

This repository is organized as follows:

> :floppy_disk: data: folder containing all dataset used
> 
> :wrench: models: folder containing all Python scripts and modules used
> 
> ::brain:: psych: folder containing psycholinguistic resources
> 
> :pencil2: sngram: folder containing syntatic dependencies n-grams


## How to cite this work
You may cite this work into yours using the following bibtext:
```
@MASTERSDISSERTATION{silva2022,
  title        = {Polarization and political ideology detection using neural models and heteregeneous knowledge},
  author       = {Samuel Caetano da Silva},
  year         = {2022},
  type         = "Master's Dissertation",
  school       = "University of São Paulo",
  address      = "São Paulo, SP, Brazil",
}
```


## Contributors

- Samuel Caetano da Silva
- Ivandré Paraboni

