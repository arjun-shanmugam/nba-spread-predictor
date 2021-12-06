import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import load_data_as_numpy, split_val_and_train


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Model, self).__init__()

    def call(self, inputs):
        return 0
        
    def loss(self, probs, labels):
        return 0

class FFModel(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt() #normalizing data batch by batch
        self.model = tf.keras.layers.Sequential([
            self.normalizer,
            tf.keras.layers.Dense(20,activation='relu')
            tf.keras.layers.Dense(20,activation='relu')
            tf.keras.layers.Dense(1,activation='linear')
        ])

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
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

    while index * model.batch_size * model.window_size < len(train_inputs):
        with tf.GradientTape() as tape:
            loss = 1
        
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
    :returns: perplexity of the test set
    """
    
    #TODO: Fill in
    #NOTE: Ensure a correct perplexity formula (different from raw loss)
    summation = 0
    losses = []
    iterCount = 0

    #batch inputs

    while iterCount * model.batch_size < test_inputs.shape[0]:
        pass

    return tf.math.exp(summation / iterCount).numpy()


def main():
    #get data
    data = load_data_as_numpy()
    train_data, test_data = split_test_and_train(data)
    #have to
    #train for 50 epochs
    num_epochs = 5
    batch_size = 50
    model = FFModel()
    #have to find a way to split into data and labels


if __name__ == '__main__':
    main()