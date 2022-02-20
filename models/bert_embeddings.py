"""
This script model is for building the BERT embeddings
"""
import numpy as np
import torch
import re
import tensorflow as tf
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Embeddings_builder:
  def __init__(self, corpus, max_len=300, dim=768, num_classes=2, batch_size=16, lang='pt'):
    self.corpus = corpus
    self.bert_max_len = max_len
    self.bert_dim = dim
    self.num_classes = num_classes
    self.batch_size = batch_size

    # Whether to use BERT base for english texts
    if lang == 'en':
      self.bert_model = BertModel.from_pretrained('bert-base-uncased')
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Whether to use BERT base for portuguese texts
    else:
      self.bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    # Define resource paths
    self.csv_path = f'../data/{self.corpus}'
    self.psych_path = f'../psych_features/{self.corpus}'
    self.sngram_path = f'../sngram_features/{self.corpus}'

  def load_text(self, filename):
    """Loads textual dataset 

    Args:
        filename: the corpus name to be used

    Returns:
        a tf.Dataset
    """
    return tf.data.experimental.make_csv_dataset(
        self.csv_path+filename,
        shuffle=False,
        batch_size=1,
        num_parallel_reads=4,
    )

  def load_liwc(self, filename):
    """Loads psychlinguistic features

    Args:
        filename: the corpus name of the refered psychlinguistic features to use

    Returns:
        a tf.Dataset
    """
    return tf.data.experimental.make_csv_dataset(
        self.psych_path+filename,
        shuffle=False,
        batch_size=1,
        num_parallel_reads=4,
    )
  
  def clean(self, text):
    """Apply cleaning of twitter-mentions and all non-alphanumerical chars

    Args:
        text: the textual content to be cleaned

    Returns:
        text containing only alphanumerical chars, except twitter-mentions
    """
    text = re.sub(r'([@#][\w_-]+)', ' ', text)
    text = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', text)
    text = re.sub(r'\W', ' ', text)
    
    return text

  def _bert_preproc_unit(self, text, label):
    """Applies BERT preprocessing in a mapping fashion

    Args:
      text: textual content from dataset
      label: label from the textual contents

    Returns:
      indexed_text: an int vector indexed by BERT
      label: the one-hot for label
      text: early alphanumerical text
    """

    # Add special tokens [CLS] and [SEP] to sentences
    text = text.numpy()[0].decode('utf-8')
    
    text = '[CLS] ' + text + ' [SEP]'

    # Tokenize sentences
    tokenized_text = self.tokenizer.tokenize(text)
    
    # Clip to the maximum BERT's model sequence's length
    tokenized_text = tokenized_text[:self.bert_max_len]

    # Find the BERT's index for each token in the sentence
    indexed_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)

    # Pad the text to the limit of 300 tokens
    indexed_text = pad_sequences(
      [indexed_text],
      maxlen=self.bert_max_len,
      padding='post'
    )

    # Return padding to the max length
    return indexed_text, label, text

  def _build_bert(self, text, label, text_str):
    """Builds BERT embeddings from input

    Args:
      text: int vector previously indexed by BERT
      label: label from sample
      text_str: alphanumerical text from sample

    Returns:
      bert_embeddings: a 3D matrix, containing word embeddings
      label: the true label corresponding to the word embedding matrix
      text_str: the aplhanumerical text
    """

    # texts is a list of texts already tokenized and in numerical representation
    text = text[0].numpy()

    text_embedding = []

    segments_ids = [1] * len(text)
    text_tensor = torch.tensor([text]).type(torch.LongTensor)
    segment_tensor = torch.tensor([segments_ids]).type(torch.LongTensor)

    with torch.no_grad():
      encoded_layers, _ = self.bert_model(
        text_tensor,
        segment_tensor,
      )

    for token_i in range(len(text)):
      hidden_layers = []

      for layer_i in range(len(encoded_layers)):
        embed = encoded_layers[layer_i][0][token_i] # type : tensor
        hidden_layers.append(embed)
      
      text_embedding.append(hidden_layers)

    # This is for word embedding (number_of_tokens, 768)
    summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0).numpy() \
                            for layer in text_embedding]
    
    # Convert nparray to list (for json-file saving)
    summed_last_4_layers = [_.tolist() for _ in summed_last_4_layers]

    #print(np.array(summed_last_4_layers).shape)
    label = tf.one_hot(label, depth=2)

    return np.array(summed_last_4_layers), label[0], text_str
  
  def efficient_bert_preprocessing(self, sample):
    """Maps a function to each sample in tf.Dataset

    Args:
      sample: a sample from the dataset

    Returns:
      preprocessed_sample: outputs content that comes from _bert_preproc_unit
    """
    
    # Maps a tensorflow function to the content inputed
    preprocessed_sample = tf.py_function(
      self._bert_preproc_unit,
      inp=[
        sample['Text'],
        sample['Class'],
      ],
      Tout=(
        tf.int32,
        tf.int32,
        tf.string,
      )
    )

    return preprocessed_sample

  def efficient_build_bert_embeddings(self, text, label, text_str):
    """Maps the building of BERT embeddings to each sample in a tf.Dataset

    Args:

    Returns:
    """
    BERTed_sample = tf.py_function(
      self._build_bert,
      inp=[
        text,
        label,
        text_str
      ],
      Tout=(
        tf.float32,
        tf.float32,
        tf.string
      ),
    )
    
    BERTed_sample[0].set_shape([self.bert_max_len, self.bert_dim])
    BERTed_sample[1].set_shape([2])

    return BERTed_sample