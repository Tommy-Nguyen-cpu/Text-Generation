import tensorflow as tf

import numpy as np
import os
import time

class TextGenModel(tf.keras.Model):
    def __init__(self, text, embedding_dim=256, rnn_units=1024):
        super().__init__()

        self.vocab = set(text)

        self.ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(self.vocab)) # Convert characters into ids.
        self.chars_from_ids = tf.keras.layers.StringLookup(vocabulary=list(self.vocab), invert=True) # Convert ids into characters.

        self.vocab_size = len(self.ids_from_chars.get_vocabulary())

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, embedding_dim) # Map ID to vector of numbers
        self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True) # Learn patterns using gated mechanisms (determines what is remembered and what will be used to update GRU parameters).
        self.dense = tf.keras.layers.Dense(self.vocab_size) # Output layer

    def texts_from_ids(self, predicted_logits):
        ids = tf.random.categorical(predicted_logits, num_samples=1) # Samples from logit distribution predicted by model.
        ids = tf.squeeze(ids, axis=-1) # Removes dimensions of size 1 (i.e. remove unnecessary dimension when it only has 1 element).
        return tf.strings.reduce_join(self.chars_from_ids(ids), axis = -1) # Characters are concatenated via their rows.
    
    def split_data(self, input):
        return input[:-1], input[1:] # Returns input data and target label, respectively.
    
    def create_dataset(self, text, sequence_length = 100, batch_size = 64, buffer_size = 10000):
        ids = self.ids_from_chars(tf.strings.unicode_split(text, "UTF-8")) # String to token, then to id.

        ids_dataset = tf.data.Dataset.from_tensor_slices(ids) # Converts text vector (ids representing text) into an array representing character indices in text. 

        sequences = ids_dataset.batch(sequence_length+1, drop_remainder=True) # Convert data to 1D array with each element being a character.

        dataset = sequences.map(self.split_data) # Creates a dataset of input data and target labels.

        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def call(self, inputs, states=None, return_states=False, training=False):
        x = self.embedding(inputs, training=training) # Pass Ids to embedding, which will return vector.

        # If states is not defined, we have GRU define the initial state.
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, states, training=training) # GRU makes prediction based on input and state.

        x = self.dense(x)

        if return_states:
            return x, states
        else:
            return x
    
    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs

        with tf.GradientTape() as tape:
            predictions = self.call(inputs, training=True) # Have our model predict and training set to true.
            loss = self.loss(labels, predictions) # Calculate loss
            grads = tape.gradient(loss, self.trainable_variables) # Calculate gradients based on loss and trainable variables.
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) # Updated gradients with respects to trainable variables.
        
            return {"loss":loss}

