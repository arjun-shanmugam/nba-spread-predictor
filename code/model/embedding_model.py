import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from load_data import load_data_as_numpy, split_test_and_train
from datetime import datetime
from matplotlib import pyplot as plt

class FFModelWithEmbeddings(tf.keras.Model):
    def __init__(self, batch_size):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam()
        self.logdir="logs/FFModelWithEmbeddings/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.batch_size = batch_size
        self.E = tf.keras.layers.Embedding(batch_size, 32)
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(1,activation='linear')
        ])
        self.loss_list = [] #loss_list for loss visualization

    # Inputs is the vector of stats (no date, no team)
    # TeamIds corresponds to a unique season/team combo
    def call(self, inputs, team_ids):
        # team_and_year_embeddings = tf.nn.embedding_lookup(unique_ids)
        self.normalizer.adapt(inputs)
        inputs = self.normalizer(inputs) #normalize non-embedding inputs
        # inputs = tf.keras.layers.concatenate([inputs, team_and_year_embeddings])
        return self.model(inputs)
        
    def loss(self, y_pred, y_true):
        return tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))