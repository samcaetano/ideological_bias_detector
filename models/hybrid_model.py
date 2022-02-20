''' This is the script model '''
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
import tensorflow as tf
import numpy as np
import json
import math
import tensorflow_addons as tfa 

class NeuralModel:
  def __init__(self,
    corpus,
    mode,
    num_samples,
    num_conv_layers=5,
    batch_size=8,
    num_epochs=15,
    num_tokens=300,
    bert_dim=768,
    num_comple_features=None,
    comple_features_train=None,
    comple_features_test=None,
  ):
    # Hyperparameters
    self.batch_size = batch_size
    self.num_samples = num_samples
    self.num_epochs = num_epochs
    self.num_tokens = num_tokens
    self.num_comple_features = num_comple_features
    self.bert_dim = bert_dim
    self.num_conv_layers = num_conv_layers

    # Complementary features
    self.comple_features_train = comple_features_train
    self.comple_features_test = comple_features_test

    # Meta-data
    self.mode = mode
    self.corpus = corpus    

    self.strategy = tf.distribute.MirroredStrategy()
    self.model_path = '/content/drive/My Drive/Colab Notebooks/GovBR/saved_models/'
    self.json_path = '/content/drive/My Drive/Colab Notebooks/GovBR/json/'

  def _pack_features(self, features):
    return tf.stack(
      list(
        features.values()
      ),
      axis=1,
    )

  def _reorder_hybrid(self, sample, sngram):
    return {'input_embedding':sample[0], 'input_sngram':sngram}, sample[1]

  def _reorder_for_test(self, sample, sngram):
    return {'input_embedding':sample[0], 'input_sngram':sngram}, sample[1], sample[2]

  def _reorder(self, embedding, label, text):
    return {'input_embedding':embedding}, label

  def build_graph(self):
    """
    This will build the model's architecture for classification

    Args:
      None
    
    Returns:
      a tf.keras Model with layers to fit
    """

    # Open the strategy scope
    with self.strategy.scope():

      # Layer for textual word embeddings
      embedding_layer = tf.keras.layers.Input(shape=(self.num_tokens, self.bert_dim), name='input_embedding')

      # Whether model is meant to be baseline.bert
      query_layer = tf.keras.layers.Flatten()(embedding_layer)

      # Whether model is cnn-based
      if self.mode in ['bert', 'bert+sngram', 'bert+liwc', 'bert+sngram+liwc']:
        kernel_sizes = [2, 3, 4, 5, 6]
        layers = []
        for i in range(self.num_conv_layers):
          # Add a CNN layer
          conv_layer = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=kernel_sizes[i],
            activation="relu"
          )(embedding_layer)
          
          conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)
          
          # Add MaxPooling layer
          conv_layer = tf.keras.layers.MaxPooling1D(
            pool_size=self.num_tokens-kernel_sizes[i]
          )(conv_layer)
          
          # Add dropout layer
          conv_layer = tf.keras.layers.Dropout(0.5)(conv_layer)
          layers.append(conv_layer)
        
        # Concatenate all CNN layers
        conv_layers = tf.keras.layers.Concatenate()(layers)
        conv_layers = tf.keras.layers.Flatten()(conv_layers)

        # Updates previous query_layer, from baseline.bert to CNN.bert
        query_layer = tf.keras.Flatten()(conv_layers)

        # Whether model is hybrid
        if self.mode in ['bert+sngram', 'bert+liwc', 'bert+sngram+liwc']:
          input_extra = tf.keras.layers.Input(
            shape=(1, self.num_comple_features),
            name='input_extra_knowledge'
          )

          # Add CNN to the heterogeneous features
          layer_extra = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=1,
            activation='relu'
          )(input_extra)
          
          layer_extra = tf.keras.layers.BatchNormalization()(layer_extra)
          layer_extra = tf.keras.layers.MaxPooling1D(pool_size=1)(layer_extra)
          layer_extra = tf.keras.layers.Flatten()(layer_extra)

          # Concatenate hybrid cnn with embedding cnn
          concat_layer = tf.keras.layers.Concatenate()([layer_extra, conv_layers])
          
          # Updates previous query_layer, from CNN.bert to any hybrid cnn-based model
          query_layer = tf.keras.Flatten()(concat_layer)
     
      query_layer = tf.keras.layers.Dropout(0.5)(query_layer)
      output_layer = tf.keras.layers.Dense(
        self.num_classes,
        activation='softmax',
        name='output',
      )(query_layer)

      if self.mode in ['baseline.bert', 'bert']:
        model = tf.keras.Model(
          embedding_layer,
          output_layer,
        )

      elif self.mode in ['bert+sngram', 'bert+liwc', 'bert+sngram+liwc']:
        model = tf.keras.Model(
          [embedding_layer, input_extra],
          output_layer,
        )

      model.compile(
          loss=tf.keras.losses.CategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(),
          metrics=[
            tfa.metrics.F1Score(2),
            'accuracy',
          ]
      )

      return model

  def train(self, dataset):
    """This will train the builted graph 
    """

    model = self.build_graph()
    model.summary()

    BATCH_SIZE_PER_REPLICA = self.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync
    DEVELOPMENT_SIZE = math.ceil(self.num_samples) 
    
    # Takes all dataset
    dataset = dataset.take(DEVELOPMENT_SIZE)

    # Split training section
    TRAIN_SIZE = math.ceil(DEVELOPMENT_SIZE * 0.8)

    # Takes training and validation section
    train = dataset.take(TRAIN_SIZE)
    validation = dataset.skip(TRAIN_SIZE)
        
    steps_per_epoch = math.ceil(TRAIN_SIZE / BATCH_SIZE) 
    validation_steps = math.ceil((DEVELOPMENT_SIZE - TRAIN_SIZE) / BATCH_SIZE) 

    train = train.map(builder.efficient_bert_preprocessing)
    train = train.map(builder.efficient_build_bert_embeddings)

    validation = validation.map(builder.efficient_bert_preprocessing)
    validation = validation.map(builder.efficient_build_bert_embeddings)

    if self.comple_features_train is not None:
      self.comple_features_train = self.comple_features_train.take(DEVELOPMENT_SIZE) # takes all
      self.comple_features_train = self.comple_features_train.map(self._pack_features)

      train_sngram = self.comple_features_train.take(TRAIN_SIZE)
      validation_sngram = self.comple_features_train.skip(TRAIN_SIZE)

      train = tf.data.Dataset.zip((train, train_sngram))
      validation = tf.data.Dataset.zip((validation, validation_sngram))

      train = train.map(self._reorder_hybrid)
      validation = validation.map(self._reorder_hybrid)
    
    else:
      train = train.map(self._reorder)
      validation = validation.map(self._reorder)

    train_dataset = train.batch(BATCH_SIZE).cache().repeat(50)
    validation_dataset = validation.batch(BATCH_SIZE).repeat(1)
    
    checkpoints = ModelCheckpoint(
      filepath=self.model_path+f'{self.corpus}.{self.mode}.model.hdf5',
      verbose=2,
      monitor='val_accuracy',
      save_best_only=True, 
      mode='max',
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

  def predict(self, test, test_num_samples):

    model = self.build_graph()
    
    model.load_weights(
      self.model_path + f'{self.corpus}.{self.mode}.model.hdf5'
    )

    test = test.take(test_num_samples)

    test = test.map(builder.efficient_bert_preprocessing)
    test = test.map(builder.efficient_build_bert_embeddings)


    if self.comple_features_test is not None:      
      test_sngram = self.comple_features_test.map(self._pack_features)

      test = tf.data.Dataset.zip((test, test_sngram))

      test = test.map(self._reorder_for_test)

    else:
      test = test.map(self._reorder)
    
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
      json.dump(incorrects, f)

    confusion_matrix = metrics.confusion_matrix(references, predictions)
    class_report = metrics.classification_report(references, predictions)

    print(confusion_matrix)
    print(class_report)