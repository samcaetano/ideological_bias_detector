from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch
import re
import tensorflow as tf


class Embeddings_builder:
  def __init__(self, lang='en'):
    self.BERT_MAX_LEN = 300
    self.CSV_PATH = '../data/'
    self.LIWC_PATH = '../liwc/'
    self.LIWC_MRC_PATH = '../liwc_mrc/'

    if lang == 'en':
      # If text language is English
      self.bert_model = BertModel.from_pretrained('bert-base-uncased')
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    else:
      # If text language is Portuguese
      self.bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


  def load_text(self, filename):
    '''
    Load the csv file into a tf.Dataset

    :param filename: the path to the text file
    :return: a tf.Dataset
    '''
    return tf.data.experimental.make_csv_dataset(
      self.CSV_PATH+filename,
      batch_size=1
      )


  def load_liwc(self, filename):
    '''
    Load the liwc features file into a tf.Dataset

    :param filename: the path to the liwc features file
    :return: a tf.Dataset
    '''
    return tf.data.experimental.make_csv_dataset(
     self.LIWC_PATH+filename,
     batch_size=1
     )


  def load_mixture(self, filename):
    '''
    Load the liwc+mrc features file into a tf.Dataset

    :param filename: the path to the liwc+mrc file
    :return: a tf.Dataset
    '''
    return tf.data.experimental.make_csv_dataset(
      self.LIWC_MRC_PATH+filename,
      batch_size=1
      )
  

  def clean(self, text):
    '''
    Apply text preprocessing

    :param text: a single text from the corpus
    :return: the text without mentions, urls, html code, numbers, etc.
    '''
    text = re.sub(r'([@#][\w_-]+)', ' ', text)
    text = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', text)
    text = re.sub(r'\W', ' ', text)
    return text


  def efficient_bert_preprocessing(self, sample):
    '''
    Apply preprocessing to the text with the BERT requirements

    :param sample: a row from the tf.Dataset
    :return: a tf.Tensor with the content from the inner function 
    '''

    # This inner function is used in a map() way
    def bert_preproc_unit(text, label):
      '''
      Preprocess the text using BERT from pytorch lib

      :param text: the text content from the row 
      :param label: the label of the text
      :return: a tuple containing 1) the indexed text (numerical)
                           2) the label 
                           3) the original text
      '''

      # Add special tokens [CLS] and [SEP] to sentences    
      text = self.clean(text.numpy()[0].decode('utf-8'))

      text = '[CLS] ' + text + ' [SEP]'

      # Tokenize sentences
      tokenized_text = self.tokenizer.tokenize(text)

      # Clip to the maximum BERT's model sequence's length
      tokenized_text = tokenized_text[:self.BERT_MAX_LEN]

      # Find the BERT's index for each token in the sentence
      indexed_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)

      # Pad the text to the limit of 300 tokens
      indexed_text = tf.keras.preprocessing.sequence.pad_sequences(
		  [indexed_text],
		  maxlen=300,
		  padding='post'
		  )

      # Return padding to the max length
      return indexed_text, label, text

    # Apply a function to the content of the tf.Dataset
    preprocessed_sample = tf.py_function(
      bert_preproc_unit,
      inp=[sample['text'], sample['hyperpartisan']],
      Tout=(tf.int32, tf.int32, tf.string)
    )

    return preprocessed_sample


  def efficient_build_bert_embeddings(self, text, label, text_str):    
    '''
    Build BERT embeddings from the text. Receives a text in numerical representation
    then build the embedding. Heavy BERT computation.

    :param text:  the text in numerical representation 
    :param label: the label of the text
    :param text_str: the original text, keep for latter reference
    :return: a tf.Tensor with the content from the inner function
    '''

    # This inner function is used in a map() way
    def build_bert(text, label, text_str):
      '''
      Heavy computation of the BERT embeddings matrix from the input text.

      :param text: the text in numerical representation
      :param label: the label of the text
      :param text_str: the original text, keep for latter reference
      :return: a tuple containing 1) the BERT embedding numpy matrix
                                             2) the label
                                             3) the original string text 
      '''
      # texts is a list of texts already tokenized and in numerical representation
      text = text[0].numpy()

      text_embedding = []

      segments_ids = [1] * len(text)
      text_tensor = torch.tensor([text]).type(torch.LongTensor)
      segment_tensor = torch.tensor([segments_ids]).type(torch.LongTensor)

      with torch.no_grad():
        encoded_layers, _ = self.bert_model(text_tensor, segment_tensor)

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
      return np.array(summed_last_4_layers), label, text_str
      
    # Apply function to the tf.Dataset
    BERTed_sample = tf.py_function(
      build_bert,
      inp=[text, label, text_str],
      Tout=(tf.float32, tf.int32, tf.string)
      )
   
    BERTed_sample[0].set_shape([300, 768])
    BERTed_sample[1].set_shape([1])
   
    return BERTed_sample