from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np
import torch
import json
import math
from binary_neural_model import NeuralModel


class TernaryNeuralModel(NeuralModel):
  def __init__(self, corpus, mode='bert', num_conv_layers=5, 
               num_liwc_features=None, num_liwc_mrc_features=None,
               idx2word=None, LIWC_train=None, LIWC_test=None):
    # Hyperparameters
    super().__init__(
      corpus, 
      mode, 
      num_conv_layers, 
      num_liwc_features,
      num_liwc_mrc_features,
      idx2word, 
      LIWC_train,
      LIWC_test
    ) 

    self.NUM_SAMPLES = 510


  def build_graph(self):
    ''' 
    This will build the model's architecture for binary classification

    :param: None
    :return: a tf.keras Model with layers to fit
    '''

    # Open the strategy scope
    with self.strategy.scope():

      # This is the text embedding layer
      embedding_layer = tf.keras.layers.Input(shape=(self.NUM_TOKENS, self.BERT_DIM), name='input_embedding')

      if self.mode == 'baseline.bert':
        # Get a baseline.bert model
        concat_layer = Flatten()(embedding_layer)

      else:
        # Get CNN based models

        if self.mode in ['cnn.bert+liwc', 'cnn.bert+liwc+mrc']:
          # This is the psycholinguistic features layer
          input_liwc = tf.keras.layers.Input(shape=(1, self.NUM_PSYCHO_FEATURES), name='input_liwc')
          layer_liwc = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation="relu")(input_liwc)
          layer_liwc = tf.keras.layers.BatchNormalization()(layer_liwc)
          layer_liwc = tf.keras.layers.MaxPooling1D(pool_size=1)(layer_liwc)
          layer_liwc = tf.keras.layers.Flatten()(layer_liwc)

        # Creates CNN layers
        kernel_sizes = [2, 3, 4, 5, 6]
        layers = []
        for i in range(self.NUM_CONV_LAYERS):
          conv_layer = tf.keras.layers.Conv1D(filters=128, kernel_size=kernel_sizes[i], activation="relu")(embedding_layer)
          conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)
          conv_layer = tf.keras.layers.MaxPooling1D(pool_size=self.NUM_TOKENS-kernel_sizes[i])(conv_layer)
          conv_layer = tf.keras.layers.Dropout(0.5)(conv_layer)
          layers.append(conv_layer)
        
        # Concatenate all CNN layers
        query_layer = tf.keras.layers.Concatenate()(layers) 
      
        if self.mode == 'cnn.bert':
          # Keep data-driven approach
          concat_layer = query_layer
        
        elif self.mode in ['cnn.bert+liwc', 'cnn.bert+liwc+mrc']:
          # Add CNN and psycholinguistic layers
          concat_layer = tf.keras.layers.Dense(128)(query_layer)
          concat_layer = tf.keras.layers.Add()([concat_layer, layer_liwc])
      
      concat_layer = tf.keras.layers.Flatten()(concat_layer)
      concat_layer = tf.keras.layers.Dropout(0.5)(concat_layer)
      output_layer = tf.keras.layers.Dense(3, activation='softmax', name='output')(concat_layer)

      if self.mode == 'cnn.bert':
        model = tf.keras.Model(embedding_layer, output_layer)

      elif self.mode in ['cnn.bert+liwc', 'cnn.bert+liwc+mrc']:
        model = tf.keras.Model([embedding_layer, input_liwc], output_layer)

      model.compile(
          loss=tf.keras.losses.CategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(),
          metrics=['accuracy']
      )

      return model


  def predict(self, test, dataset_size=None):
    '''
    Generate predictions from the trained model

    :param test: a tf.Dataset to predict on
    :param dataset_size: a tf.Dataset size
    :return: None
    '''
    # This is for LIWC features dataset
    pack_features = lambda features : tf.stack(list(features.values()), axis=1)

    # Reorder in a tuple format ({x, liwc}, y)
    reorder = lambda sample, liwc : {'input_embedding':sample[0], 'input_liwc':liwc}, sample[1], sample[2]

    # Build model graph
    model = self.build_graph()
    
    # Load trained model weights
    model.load_weights(self.model_path + f'{self.corpus}.{self.mode}.model.hdf5')

    test = test.take(int(dataset_size))

    # Preprocess and buil BERT embeddings
    test = test.map(builder.efficient_bert_preprocessing)
    test = test.map(builder.efficient_build_bert_embeddings)

    steps_per_epoch = math.ceil(dataset_size/32)

    if self.liwc_test is not None:
      test_liwc = self.liwc_test.map(pack_features)

      test = tf.data.Dataset.zip((test, test_liwc))

      test = test.map(reorder)
    
    test = test.batch(1)

    references, predictions, incorrects = [], [], []
    for sample in test:
      x, y, text = sample[0], sample[1], sample[2]
      y_pred = model.predict(x, verbose=1)
      
      # Saves text from incorrect predictions for later analysis
      if int(np.argmax(y_pred)) != int(np.argmax(y)):
        incorrects.append(text.numpy()[0].decode('utf-8'))

      predictions.append(int(np.argmax(y_pred)))
      references.append(int(np.argmax(y)))
    
    with open(self.json_path + f'{self.corpus}.{self.mode}.json', 'w') as f:
      json.dump(predictions, f)
    
    with open(self.json_path + f'{self.corpus}.{self.mode}.json', 'w') as f:
      json.dump(references, f)
    
    with open(self.json_path + f'{self.corpus}.{self.mode}.incorrects', 'w') as f:
      json.dump(incorrects, f)