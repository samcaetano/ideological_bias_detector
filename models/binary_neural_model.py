from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np
import torch
import json
import math


class NeuralModel:
  def __init__(self, corpus, mode='bert', num_conv_layers=5, 
               num_liwc_features=None, num_liwc_mrc_features=None,
               idx2word=None, LIWC_train=None, LIWC_test=None):
    # Hyperparameters
    self.NUM_FOLDS = 5
    self.BATCH_SIZE = 16
    self.NUM_EPOCHS = 15
    self.NUM_TOKENS = 300
    self.NUM_PSYCHO_FEATURES = num_liwc_features
    self.BERT_DIM = 768
    self.NUM_CONV_LAYERS = num_conv_layers

    # External knowledge
    self.idx2word = idx2word # vocabulary
    self.liwc_train = LIWC_train # psycolinguistic
    self.liwc_test = LIWC_test    

    # Meta-data
    self.mode = mode
    self.corpus = corpus

    if self.corpus == 'by_articles':
      self.NUM_SAMPLES = 645
    elif self.corpus == 'by_publisher':
      self.NUM_SAMPLES = 2500

    self.strategy = tf.distribute.MirroredStrategy()
    self.model_path = '../saved_models/'
    self.json_path = '../json/'


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
      output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(concat_layer)

      if self.mode == 'cnn.bert':
        model = tf.keras.Model(embedding_layer, output_layer)

      elif self.mode in ['cnn.bert+liwc', 'cnn.bert+liwc+mrc']:
        model = tf.keras.Model([embedding_layer, input_liwc], output_layer)

      model.compile(
          loss=tf.keras.losses.BinaryCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(),
          metrics=['accuracy']
      )

      return model


  @DeprecationWarning
  def pack_features(features):
      '''
      This is for LIWC features dataset

      :param features: liwc features to get right
      :return: None
      '''
      return tf.stack(list(features.values()), axis=1)


  @DeprecationWarning
  def reorder(sample, liwc):
      '''
      Reorder features before the model's fit. sample, liwc is composed of a tuple in the following format
      ((x, y), liwc). The idea is to reformat to the following output ({'input_embedding': x, 'input_liwc': liwc} y)
      
      :param sample: a tuple in the format (x, y)
      :param liwc: a liwc vector from the text
      :return: a tuple in the format ({x, liwc}, y)
      '''
      return {'input_embedding':sample[0], 'input_liwc':liwc}, sample[1]


  def train(self, dataset):
    ''' 
    Fit the model with the given dataset

    :param dataset: a tf.Dataset to fit the model
    :return: None
    '''
    # This is for LIWC features dataset
    pack_features = lambda features : tf.stack(list(features.values()), axis=1)

    # Reorder in a tuple format ({x, liwc}, y)
    reorder = lambda sample, liwc : {'input_embedding':sample[0], 'input_liwc':liwc}, sample[1]

    model = self.build_graph()
    model.summary()

    BATCH_SIZE_PER_REPLICA = 32
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync
    
    # takes all training size from dataset
    TRAIN_SIZE = math.ceil(self.NUM_SAMPLES * 0.8) 
    
    # takes all
    dataset = dataset.take(self.NUM_SAMPLES) 

    if self.liwc_train is not None:
      # takes all
      self.liwc_train = self.liwc_train.take(self.NUM_SAMPLES) 

    # train size
    train = dataset.take(TRAIN_SIZE) 

    # validation size
    validation = dataset.skip(TRAIN_SIZE) 
        
    # steps per epoch for training
    steps_per_epoch = math.ceil(TRAIN_SIZE/BATCH_SIZE) 

     # steps per epoch for validation
    validation_steps = math.ceil((self.NUM_SAMPLES - TRAIN_SIZE) / BATCH_SIZE)

    # Preprocess data and build its BERT embedding
    train = train.map(builder.efficient_bert_preprocessing)
    train = train.map(builder.efficient_build_bert_embeddings)
    validation = validation.map(builder.efficient_bert_preprocessing)
    validation = validation.map(builder.efficient_build_bert_embeddings)

    if self.liwc_train is not None:
      # Apply features packing to liwc matrix      
      self.liwc_train = self.liwc_train.map(pack_features)

      train_liwc = self.liwc_train.take(TRAIN_SIZE)
      validation_liwc = self.liwc_train.skip(TRAIN_SIZE)

      train = tf.data.Dataset.zip((train, train_liwc))
      validation = tf.data.Dataset.zip((validation, validation_liwc))

      # Reorder train set
      train = train.map(reorder)

      # Reorder validation set
      validation = validation.map(reorder)

    train_dataset = train.batch(BATCH_SIZE).cache().repeat(50)
    validation_dataset = validation.batch(BATCH_SIZE).cache().repeat(50)
    
    checkpoints = ModelCheckpoint(
      filepath=self.model_path + f'{self.corpus}.{self.mode}.model.hdf5',
      verbose=2,
      monitor='val_accuracy',
      save_best_only=True,
      mode='max'
    )
    
    # Fit the model
    history = model.fit(
      train_dataset,
      epochs=50,
      validation_data=validation_dataset,
      steps_per_epoch=steps_per_epoch,
      validation_steps=validation_steps,
      validation_freq=10,
      callbacks=[checkpoints]
    )


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
      
      if round(y_pred[0][0]) != y.numpy()[0][0]:
        # Saves text from incorrect predictions for later analysis
        incorrects.append(text.numpy()[0].decode('utf-8'))

      predictions.append(round(y_pred[0][0]))
      references.append(int(y.numpy()[0][0]))
    
    y_pred = np.asarray(predictions)
    predictions = np.where(y_pred < 0.5, 0, 1)
    predictions = predictions.tolist()

    with open(self.json_path + f'{self.corpus}.{self.mode}.json', 'w') as f:
      json.dump(predictions, f)
    
    with open(self.json_path + f'{self.corpus}.{self.mode}.json', 'w') as f:
      json.dump(references, f)
    
    with open(self.json_path + f'{self.corpus}.{self.mode}.incorrects', 'w') as f:
      json.dump(incorrects, f)
