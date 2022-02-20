''' This is the script model '''
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Conv2D
from tensorflow.keras.layers import Concatenate, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Embedding, BatchNormalization, Add, Attention
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from pytorch_pretrained_bert import BertModel
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from sklearn import metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import math
import tensorflow_addons as tfa 

class NeuralModel:
  def __init__(self, corpus, mode='bert', num_conv_layers=2, 
               num_sngram_features=None, num_samples=510,
               attention=False, idx2word=None, sngram_train=None, sngram_test=None):
    # Hyperparameters
    self.num_folds = 5
    self.batch_size = 8
    self.num_epochs = 15
    self.num_tokens = 300
    self.num_sngram_features = num_sngram_features
    self.BERT_dim = 768
    self.GLOVE_dim = 300
    self.kfold = KFold(n_splits=self.num_folds, shuffle=False)
    self.num_conv_layers = num_conv_layers
    self.attention = attention

    # External knowledge
    self.idx2word = idx2word # vocabulary
    self.sngram_train = sngram_train # psycolinguistic    
    self.sngram_test = sngram_test

    # Meta-data
    self.mode = mode
    self.corpus = corpus

    self.num_samples = num_samples #  408 dev, 102 test

    self.strategy = tf.distribute.MirroredStrategy()
    self.model_path = '/content/drive/My Drive/Colab Notebooks/GovBR/saved_models/'
    self.json_path = '/content/drive/My Drive/Colab Notebooks/GovBR/json/'


  def build_graph(self):
    ''' 
    This will build the model's architecture for classification

    :param: None
    :return: a tf.keras Model with layers to fit
    '''

    # Open the strategy scope
    with self.strategy.scope():

      # This is the text embedding layer
      embedding_layer = tf.keras.layers.Input(shape=(self.num_tokens, self.BERT_dim), name='input_embedding')

      if self.mode == 'bert+sngram+liwc':
        
        input_sngram = tf.keras.layers.Input(shape=(1, self.num_sngram_features), name='input_sngram')
        sngram_layer = tf.keras.layers.Flatten()(input_sngram)

        # Creates CNN layers
        kernel_sizes = [2, 3, 4, 5, 6]
        layers = []
        for i in range(self.num_conv_layers):
          # filters 128 or 16
          conv_layer = tf.keras.layers.Conv1D(filters=128, kernel_size=kernel_sizes[i], activation="relu")(embedding_layer)
          conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)
          conv_layer = tf.keras.layers.MaxPooling1D(pool_size=self.num_tokens-kernel_sizes[i])(conv_layer)
          conv_layer = tf.keras.layers.Dropout(0.5)(conv_layer)
          layers.append(conv_layer)
        
        # Concatenate all CNN layers
        query_layer = tf.keras.layers.Flatten()(
                          tf.keras.layers.Concatenate()(layers))

        #bert_layer = tf.keras.layers.Flatten()(embedding_layer)

        concat_layer = tf.keras.layers.Concatenate()([sngram_layer, query_layer])

      elif self.mode == 'baseline.bert':
        # Get a baseline.bert model
        concat_layer = tf.keras.layers.Flatten()(embedding_layer)

      else:
        # Get CNN based models
        if self.mode in ['cnn.bert+sngram']:
          # This is the Sn-gram features layer
          input_sngram = tf.keras.layers.Input(shape=(1, self.num_sngram_features), name='input_sngram')
          layer_sngram = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation="relu")(input_sngram)
          layer_sngram = tf.keras.layers.BatchNormalization()(layer_sngram)
          layer_sngram = tf.keras.layers.MaxPooling1D(pool_size=1)(layer_sngram)
          layer_sngram = tf.keras.layers.Flatten()(layer_sngram)

        # Creates CNN layers
        kernel_sizes = [2, 3, 4, 5, 6]
        layers = []
        for i in range(self.num_conv_layers):
          # filters 128 or 16
          conv_layer = tf.keras.layers.Conv1D(filters=128, kernel_size=kernel_sizes[i], activation="relu")(embedding_layer)
          conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)
          conv_layer = tf.keras.layers.MaxPooling1D(pool_size=self.num_tokens-kernel_sizes[i])(conv_layer)
          conv_layer = tf.keras.layers.Dropout(0.5)(conv_layer)
          layers.append(conv_layer)
        
        # Concatenate all CNN layers
        query_layer = tf.keras.layers.Concatenate()(layers) 
      
        if self.mode == 'cnn.bert':
          # Keep data-driven approach
          concat_layer = query_layer
        
        elif self.mode in ['cnn.bert+sngram']:
          # Add CNN and psycholinguistic layers
          concat_layer = tf.keras.layers.Dense(128)(query_layer)
          concat_layer = tf.keras.layers.Add()([concat_layer, layer_sngram])
        
        concat_layer = tf.keras.layers.Flatten()(concat_layer)
      

      concat_layer = tf.keras.layers.Dropout(0.5)(concat_layer)
      output_layer = tf.keras.layers.Dense(2, activation='softmax', name='output')(concat_layer)

      if self.mode in ['baseline.bert', 'cnn.bert']:
        model = tf.keras.Model(embedding_layer, output_layer)

      elif self.mode in ['bert+sngram+liwc']:
        model = tf.keras.Model([embedding_layer, input_sngram], output_layer)

      model.compile(
          loss=tf.keras.losses.CategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(),
          metrics=[tfa.metrics.F1Score(2), 'accuracy']
      )

      return model

  def train(self, dataset):
    ''' 
      This will train the builted graph 
    '''
    model = self.build_graph()
    model.summary()

    BATCH_SIZE_PER_REPLICA = self.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync
    DEVELOPMENT_SIZE = math.ceil(self.num_samples * 0.8) 
    
    dataset = dataset.take(DEVELOPMENT_SIZE)

    TRAIN_SIZE = math.ceil(DEVELOPMENT_SIZE * 0.8) # 2612

    train = dataset.take(TRAIN_SIZE) # takes all
    validation = dataset.skip(TRAIN_SIZE)

    if self.sngram_train is not None:
      self.sngram_train = self.sngram_train.take(DEVELOPMENT_SIZE) # takes all
        
    steps_per_epoch = math.ceil(TRAIN_SIZE / BATCH_SIZE) 
    validation_steps = math.ceil((DEVELOPMENT_SIZE - TRAIN_SIZE) / BATCH_SIZE) 

    train = train.map(builder.efficient_bert_preprocessing)
    train = train.map(builder.efficient_build_bert_embeddings)

    validation = validation.map(builder.efficient_bert_preprocessing)
    validation = validation.map(builder.efficient_build_bert_embeddings)

    if self.sngram_train is not None:
      def pack_features(features):
        return tf.stack(list(features.values()), axis=1)

      def reorder(sample, sngram):
        return {'input_embedding':sample[0], 'input_sngram':sngram}, sample[1]
      
      self.sngram_train = self.sngram_train.map(pack_features)

      train_sngram = self.sngram_train.take(TRAIN_SIZE)
      validation_sngram = self.sngram_train.skip(TRAIN_SIZE)

      train = tf.data.Dataset.zip((train, train_sngram))
      validation = tf.data.Dataset.zip((validation, validation_sngram))

      train = train.map(reorder)
      validation = validation.map(reorder)
    
    else:
      def reorder(embedding, label, text):
        return {'input_embedding':embedding}, label

      train = train.map(reorder)
      validation = validation.map(reorder)


    train_dataset = train.batch(BATCH_SIZE).cache().repeat(50)
    validation_dataset = validation.batch(BATCH_SIZE).repeat(1)
    
    checkpoints = ModelCheckpoint(
						filepath=self.model_path+f'{self.corpus}.{self.mode}.model.hdf5',
            verbose=2,
            monitor='val_accuracy',
            save_best_only=True, 
            mode='max'
            )
       
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=25,
        callbacks=[checkpoints]
        )
    
    print(history.history)

  def predict(self, test):

    model = self.build_graph()
    
    model.load_weights(
        self.model_path + f'{self.corpus}.{self.mode}.model.hdf5'
        )

    test = test.take(math.ceil(self.num_samples * 0.2))

    test = test.map(builder.efficient_bert_preprocessing)
    test = test.map(builder.efficient_build_bert_embeddings)

    #steps_per_epoch = math.ceil((self.num_samples * 0.2) / self.batch_size)
    steps_per_epoch = math.ceil(1000 / self.batch_size)

    if self.sngram_test is not None:
      def pack_features(features):
        return tf.stack(list(features.values()), axis=1)

      def reorder(sample, sngram):
        return {'input_embedding':sample[0], 'input_sngram':sngram}, sample[1], sample[2]
      
      test_sngram = self.sngram_test.map(pack_features)

      test = tf.data.Dataset.zip((test, test_sngram))

      test = test.map(reorder)

    else:
      def reorder(embedding, label, text):
        return {'input_embedding':embedding}, label, text

      test = test.map(reorder)
    
    test = test.batch(1)

    references, predictions = [], []


    incorrects = []
    for sample in test:
      x, y, text = sample[0], sample[1], sample[2]
      y_pred = model.predict(x, verbose=1)

      if int(np.argmax(y_pred)) != int(np.argmax(y)):
        text = text.numpy()[0].decode('utf-8')
        incorrects.append(text)
        

      predictions.append(int(np.argmax(y_pred)))
      references.append(int(np.argmax(y)))
      
    
    with open(self.json_path + f'{self.corpus}.{self.mode}.json', 'w') as f:
      json.dump(predictions, f)

    with open(self.json_path + f'{self.corpus}.{self.mode}.TRUE.json', 'w') as f:
      json.dump(references, f)

    with open(self.json_path+f'{self.corpus}.{self.mode}.incorrects', 'w') as f:
      print(incorrects)
      json.dump(incorrects, f)

    print(metrics.confusion_matrix(references, predictions))
    print(metrics.classification_report(references, predictions))  