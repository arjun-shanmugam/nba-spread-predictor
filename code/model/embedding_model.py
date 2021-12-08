import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from load_data import load_data_as_numpy, split_test_and_train
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from preprocess_for_embeddings import preprocess
from sklearn.decomposition import PCA
from scratch_file import id_to_teamname_and_record

class FFModelWithEmbeddings(tf.keras.Model):
    def __init__(self, batch_size, team_id_max):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.1)
        self.logdir="logs/FFModelWithEmbeddings/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.batch_size = batch_size
        self.E = tf.keras.layers.Embedding(team_id_max, 32, input_length=2)
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.flattern_layer = tf.keras.layers.Flatten()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(32,activation='relu'),
            tf.keras.layers.Dense(16,activation='relu'),
            tf.keras.layers.Dense(1,activation='linear')
        ])
        self.loss_list = [] #loss_list for loss visualization

    # Inputs is the vector of stats (no date, no team)
    # TeamIds corresponds to a unique season/team combo
    def call(self, inputs, team_ids):
        team_and_year_embeddings = self.E(team_ids)
        team_and_year_embeddings = self.flattern_layer(team_and_year_embeddings)
        self.normalizer.adapt(inputs)
        inputs = self.normalizer(inputs) #normalize non-embedding inputs
        inputs = tf.keras.layers.concatenate([inputs, team_and_year_embeddings])
        return self.model(inputs)
        
    def loss(self, y_pred, y_true):
        return tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))

def train(model, train_ds, train_ids, batch_size=32):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: avg loss
    """
    #TODO: Fill in
    total_loss = 0
    #batch inputs and labels
    dataset = train_ds
    idx = 0
    for step, (batch_x, batch_y) in enumerate(dataset):
        # feature_ds = test_ds.map(lambda x, y:tf.gather(x, range(len(x)), axis=0))
        # batch_x = batch.map(lambda x, y: x)
        # batch_y = batch.map(lambda x, y: y)
        with tf.GradientTape() as tape:
            if len(np.hstack(batch_x.values())) == batch_size: #hacky fix to prevent error
                preds = model.call(np.hstack(batch_x.values()), train_ids[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))])
                loss = model.loss(preds, batch_y)
                total_loss += loss
            idx += 1
        if len(np.hstack(batch_x.values())) == batch_size:
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    avg_loss = total_loss / (idx * batch_size)
    print("len")
    print((idx * batch_size))
    return avg_loss

def test(model, test_ds, test_ids, batch_size=32):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: avg loss
    """
    #TODO: Fill in
    total_loss = 0
    #batch inputs and labels
    # dataset = test_ds.batch(256)
    dataset = test_ds
    idx = 0
    for step, (x_batch, y_batch) in enumerate(dataset):
        feature_ds = test_ds.map(lambda x, y: x)
        with tf.GradientTape() as tape:
            if len(np.hstack(x_batch.values())) == batch_size: #hacky fix to prevent error
                preds = model.call(np.hstack(x_batch.values()), train_ids[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))])
                loss = model.loss(preds, y_batch)
                total_loss += loss
            idx += 1
    avg_loss = total_loss / (idx * batch_size)
    model.loss_list.append(avg_loss)
    return avg_loss

def project(v, basis):
  out = np.zeros((300,))
  for b in basis:
    component = np.dot(v, b) * b
    c = component.astype(float)
    out += c
  return out

def plot_embedding(team_ids, labels, model):
    vecs = model.E(np.array(team_ids, dtype=int))
    pca_two_dim = PCA(n_components=2)
    pca_one_dim = PCA(n_components=1)
    basis_vecs_two_dim = pca_two_dim.fit_transform(vecs)
    basis_vecs_one_dim = pca_one_dim.fit_transform(vecs)
    two_d_projections = []
    one_d_projections = []
    for idx in range(len(basis_vecs_two_dim)):
        two_d_projections.append(basis_vecs_two_dim[idx])
        one_d_projections.append(basis_vecs_one_dim[idx])
    x = [x for [x, y] in two_d_projections]
    y = [y for [x, y] in two_d_projections]
    cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=0, vmax=1)  
    print([id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)])  
    print("hello")
    print(len(x))
    print(len(y))
    print(len([id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)]))
    plt.plot([id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)], one_d_projections)
    plt.show()
    plt.scatter(x, y, c=[id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)], cmap=cmap, norm=norm)
    # for idx, point in enumerate(zip(x, y)):
    #     plt.annotate("Team: {}, Win Pct. {}".format(id_to_teamname_and_record[labels[idx]][0], id_to_teamname_and_record[labels[idx]][0]), point)
    plt.show()

if __name__ == "__main__":
    train_ds, test_ds, train_ids, test_ids, team_map = preprocess()
    train_ids, test_ids = np.ndarray.astype(train_ids, int), np.ndarray.astype(test_ids, int)
    model = FFModelWithEmbeddings(32, max(np.max(train_ids), np.max(test_ids)) + 1) #make sure there are enough embeddings
    for epoch in range(50):
        print(epoch)
        print(train(model, train_ds, train_ids))
    # avg_loss = test(model, test_ds, test_ids)
    # print("TEST AVG LOSS")
    # print(avg_loss)
    embedding_ids = []
    for team_id in range(1610612737, 1610612767):
        embedding_ids.append(team_map[(2018, team_id)])
    plot_embedding(embedding_ids, range(1610612737, 1610612767), model)
    
