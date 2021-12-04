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
    train_data, validation_data = split_val_and_train(data)
    #train for 50 epochs
    for _ in range(50):
        avg_loss = train(train_data)

if __name__ == '__main__':
    main()