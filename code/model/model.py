import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import load_data_as_numpy, split_test_and_train


class FFModelWithEmbeddings(tf.keras.Model):
    def __init__(self, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.E = tf.keras.layers.Embedding(batch_size, 32)
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.model = tf.keras.layers.Sequential([
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(1,activation='linear')
        ])


    def call(self, inputs):
        team_ids = 0 #get team
        year_ids = 0 #get year (will be complicated)
        unique_ids = team_ids * 21 + year_ids
        team_and_year_embeddings = tf.nn.embedding_lookup(unique_ids)
        self.normalizer.adapt(inputs)
        inputs = self.normalizer(inputs) #normalize non-embedding inputs
        inputs = tf.keras.layers.concatenate([inputs, team_and_year_embeddings])
        return self.model(inputs)
        
    def loss(self, y_pred, y_true):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

class FFModel(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt() #normalizing data batch by batch
        self.model = tf.keras.layers.Sequential([
            self.normalizer,
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(1,activation='linear')
        ])

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
            loss = tf.keras.layers.MeanSquaredError()
        )

def train2(model,train_inputs,train_labels,batch_size,num_epochs):
    """
    new train model using model.fit
    """
    model.model.fit(train_inputs,train_labels,batch_size=batch_size,num_epochs=num_epochs)


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #TODO: Fill in
    losses = []
    index = 0

    #batch inputs and labels

    while index * model.batch_size < len(train_inputs):
        with tf.GradientTape() as tape:
            loss = model.loss(train_inputs[index*model.batch_size:min(index*model.batch_size+model.batch_size)],
                              train_labels[index*model.batch_size:min(index*model.batch_size+model.batch_size)])
            index += 1
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return losses

def test2(model,test_inputs,test_labels,batch_size):
    """
    same as other test using model.evaluate
    """
    return model.model.evaluate(test_inputs,test_labels,batch_size=batch_size)

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
        total_loss += model.loss(test_inputs[index*model.batch_size:min(index*model.batch_size+model.batch_size)],
                              test_labels[index*model.batch_size:min(index*model.batch_size+model.batch_size)])
        index += 1
    return total_loss / len(test_inputs)


def main():
    #get data
    data = load_data_as_numpy()
    train_data, test_data = split_test_and_train(data)
    #have to
    #train for 50 epochs
    num_epochs = 5
    batch_size = 50
    ff_model = FFModel()
    ff_model_with_embeddings = FFModelWithEmbeddings(batch_size)
    #have to find a way to split into data and labels


if __name__ == '__main__':
    main()