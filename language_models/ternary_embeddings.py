from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch
import re
import tensorflow as tf
from embeddings import Embeddings_builder


class Embeddings_for_ternary_datasets(Embeddings_builder):
	'''
	This is a subclass from Embeddings_builder. This will build embeddings for ternary (or more) classifications
	'''
  def __init__(self, lang='pt'):
		super().__init__(lang)

	
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

			label = tf.one_hot(label, depth=3)

      return np.array(summed_last_4_layers), label[0], text_str
      
		# Apply function to the tf.Dataset
    BERTed_sample = tf.py_function(
			build_bert,
			inp=[text, label, text_str],
			Tout=(tf.float32, tf.int32, tf.string)
		)
    
		BERTed_sample[0].set_shape([300, 768])
    BERTed_sample[1].set_shape([3])
    
		return BERTed_sample