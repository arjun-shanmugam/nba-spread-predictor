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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        self.logdir="logs/FFModelWithEmbeddings/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.batch_size = batch_size
        self.E = tf.keras.layers.Embedding(team_id_max, 8, input_length=2)
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.reshape_layer = tf.keras.layers.Reshape((16,), input_shape=(2, 8))
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(2048,activation='relu'),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(256,activation='relu'),
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
        # print("inptus!!")
        team_and_year_embeddings = self.reshape_layer(team_and_year_embeddings)
        self.normalizer.adapt(inputs)
        inputs = self.normalizer(inputs) #normalize non-embedding inputs
        inputs = tf.keras.layers.concatenate([inputs, team_and_year_embeddings])
        # print(inputs[0])
        # print(inputs[1])
        return self.model(inputs)
        
    def loss(self, y_pred, y_true):
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)

def train(model, train_features, train_labels, train_ids):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: avg loss
    """
    #TODO: Fill in
    total_loss = 0
    batch_size = 32
    #batch inputs and labels
    idx = 0
    while (idx + 1) * batch_size < len(train_labels):
        # if idx == 0:
        #     model.build(np.hstack(batch_x.values()).shape)
        # feature_ds = test_ds.map(lambda x, y:tf.gather(x, range(len(x)), axis=0))
        # batch_x = batch.map(lambda x, y: x)
        # batch_y = batch.map(lambda x, y: y)
        with tf.GradientTape() as tape:
            # if len(np.hstack(batch_x.values())) == batch_size: #hacky fix to prevent error
            #     preds = model.call(np.hstack(batch_x.values()), train_ids[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))])[:,0]
            #     loss = model.loss(preds, batch_y)
            #     total_loss += loss
            #     idx += 1
            preds = model.call(train_features[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))],
                               train_ids[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))])[:,0]
            loss = model.loss(preds, train_labels[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))])
            total_loss += loss
            idx += 1
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    avg_loss = total_loss / idx
    return avg_loss

def test(model, test_features, test_labels, test_ids, batch_size=32, return_preds_and_labels=False):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: avg loss
    """
    #TODO: Fill in
    total_loss = 0
    total_l1_error = 0
    total_l2_error = 0
    #batch inputs and labels
    # dataset = test_ds.batch(256)
    idx = 0
    seen_examples = 0
    all_preds = []
    all_labels = []
    while (idx + 1) * batch_size < len(test_labels):
        preds = model.call(test_features[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))],
                               test_ids[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))])[:,0]
        all_preds.extend(list(preds))
        all_labels.extend(list(test_labels[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))]))
        loss = model.loss(preds, test_labels[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))])
        seen_examples += len(preds)
        total_l1_error += tf.losses.mean_absolute_error(preds, test_labels[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))])
        total_l2_error += tf.losses.mean_squared_error(preds, test_labels[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))])
        total_loss += loss
        idx += 1
    avg_loss = total_loss / (idx)
    avg_err = total_l1_error / (idx)
    avg_l2_err = total_l2_error / (idx)
    model.loss_list.append(avg_loss)
    if return_preds_and_labels:
        return avg_loss, avg_err, all_preds, all_labels
    return avg_loss, avg_err, avg_l2_err

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
    plt.scatter([id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)], one_d_projections)
    plt.show()
    plt.scatter(x, y, c=[id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)], cmap=cmap, norm=norm)
    plt.show()
    plt.scatter([id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)], [x_i**2 + y_i**2 for (x_i, y_i) in zip(x, y)])
    for idx, point in enumerate(zip(x, y)):
        plt.annotate("{}, {}".format(id_to_teamname_and_record[labels[idx]][0], id_to_teamname_and_record[labels[idx]][1]), point)
    plt.show()

def simulate_betting(preds, labels, spreads, vig=False):
    money = 0
    money_over_time = [0]
    spreads = spreads.tolist()
    for idx, (pred, label) in enumerate(zip(preds, labels)):
        print("Pred, Spread, True")
        print(pred)
        print(label)
        print(spreads[idx])
    for idx, (pred, label) in enumerate(zip(preds, labels)):
        pick = None
        if pred > spreads[idx]:
            pick = "OVER"
        else:
            pick = "UNDER"
        if (pick == "OVER" and label > spreads[idx]) or (pick == "UNDER" and label < spreads[idx]):
            print("made 10")
            print(pred)
            print(spreads[idx])
            print(label)
            money += 10
        elif label == spreads[idx]:
            print("push")
            money += 0
        else:
            print("lost 10")
            print(pred)
            print(spreads[idx])
            print(label)
            money -= 10
            if vig:
                print("VIG!!!")
                money -= 1
        money_over_time.append(money)
    return money, money_over_time

if __name__ == "__main__":
    # train_df, test_df, train_ids, test_ids, team_map = preprocess()
    # train_ids, test_ids = np.ndarray.astype(train_ids, int), np.ndarray.astype(test_ids, int)
    train_features, train_labels, test_features, test_labels, train_ids, test_ids, team_map = preprocess()
    train_ids, test_ids = np.ndarray.astype(train_ids, int), np.ndarray.astype(test_ids, int)
    model = FFModelWithEmbeddings(32, max(np.max(train_ids), np.max(test_ids)) + 1) #make sure there are enough embeddings
    for epoch in range(20):
        print(epoch)
        print(train(model, train_features, train_labels, train_ids))
        print(test(model, test_features, test_labels, test_ids))
    avg_loss, avg_error, avg_l2_error = test(model, test_features, test_labels, test_ids)
    print(avg_loss)
    print(avg_error)
    print(avg_l2_error)
    embedding_ids = []
    for team_id in range(1610612737, 1610612767):
        embedding_ids.append(team_map[(2018, team_id)])
        # for year in range(2007, 2021):
        #     if (year, team_id) in team_map:
        #         embedding_ids.append(team_map[(year, team_id)])
    plot_embedding(embedding_ids, range(1610612737, 1610612767), model)
    test_features, test_labels, ids, team_map, spreads = preprocess(for_testing=True)
    ids = np.ndarray.astype(ids, int)
    loss, err, preds, labels = test(model, test_features, test_labels, ids, return_preds_and_labels=True)
    money, money_over_time = simulate_betting(preds, labels, spreads)
    print("MONEY!")
    print(money_over_time)
    print(money)
    plt.plot(money_over_time)
    plt.show()
    money, money_over_time = simulate_betting(preds, labels, spreads, vig=True)
    print("MONEY!")
    print(money_over_time)
    print(money)
    plt.plot(money_over_time)
    plt.show()

    
