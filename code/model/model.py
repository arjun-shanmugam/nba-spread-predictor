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


    def call(self, inputs):
        team_ids = 0 #get team
        year_ids = 0 #get year (will be complicated)
        unique_ids = team_ids * 21 + year_ids
        # team_and_year_embeddings = tf.nn.embedding_lookup(unique_ids)
        self.normalizer.adapt(inputs)
        inputs = self.normalizer(inputs) #normalize non-embedding inputs
        # inputs = tf.keras.layers.concatenate([inputs, team_and_year_embeddings])
        return self.model(inputs)
        
    def loss(self, y_pred, y_true):
        return tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))

class FFModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.logdir="logs/FFModel/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt(np.zeros((50, 196))) #normalizing data batch by batch
        self.model = tf.keras.Sequential([
            self.normalizer,
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(1,activation='linear')
        ])

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss = tf.keras.losses.MeanSquaredError()
        )

def train2(model,train_inputs,train_labels,batch_size,num_epochs):
    """
    new train model using model.fit
    """
    model.model.fit(train_inputs,
                    train_labels,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    callbacks=[model.tensorboard_callback])


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: avg loss
    """
    #TODO: Fill in
    index = 0
    total_loss = 0
    #batch inputs and labels
    while index * model.batch_size < len(train_inputs):
        with tf.GradientTape() as tape:
            preds = model.call(train_inputs[index*model.batch_size:min(index*model.batch_size+model.batch_size, len(train_inputs))])
            loss = model.loss(preds,
                              train_labels[index*model.batch_size:min(index*model.batch_size+model.batch_size, len(train_inputs))])
            index += 1
            total_loss += loss
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    avg_loss = total_loss / len(train_inputs)
    model.loss_list.append(avg_loss)
    return avg_loss

def test2(model,test_inputs,test_labels,batch_size):
    """
    same as other test using model.evaluate
    """
    return model.evaluate(test_inputs,test_labels,batch_size=batch_size) / len(test_inputs)

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: avg loss
    """
    
    total_loss = 0
    index = 0
    #batch inputs
    while index * model.batch_size < len(test_inputs):
        total_loss += model.loss(model.call(test_inputs[index*model.batch_size:min(index*model.batch_size+model.batch_size, len(test_inputs))]),
                              test_labels[index*model.batch_size:min(index*model.batch_size+model.batch_size, len(test_inputs))])
        index += 1
    return total_loss / len(test_inputs)

def visualize_epoch_loss(losses):
    """
    Creates a simple graph to show the model losses during training.
    :param losses: list of loss data stored during training
    
    :returns: no return type, generates a plot
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()  

def main():
    #get data
    data = load_data_as_numpy()
    train_data, test_data = split_test_and_train(data)
    train_labels = train_data["point_differential"]
    test_labels = test_data["point_differential"]
    #have to
    #train for 50 epochs
    num_epochs = 5
    batch_size = 50
    ff_model = FFModel()
    ff_model_with_embeddings = FFModelWithEmbeddings(batch_size)

    # train2(ff_model, train_data, train_labels, batch_size, num_epochs)
    print(test(ff_model_with_embeddings, test_data, test_labels))
    for epoch in range(num_epochs):
        print("EPOCH: {}".format(epoch))
        print(train(ff_model_with_embeddings, train_data, train_labels))
    print(test(ff_model_with_embeddings, test_data, test_labels))
    visualize_epoch_loss(ff_model_with_embeddings.loss_list)

if __name__ == '__main__':
    main()