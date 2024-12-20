import keras
from keras import Model, layers
import numpy as np
import tensorflow as tf


class KerasActorNetwork(Model):
    def __init__(self, min_action, max_action, state_size, action_size):
        self._min_action = min_action
        self._max_action = max_action
        self._action_size = action_size

        last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(state_size,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self._action_size, activation="tanh", kernel_initializer=last_init)(out)

        outputs = outputs * max_action

        super().__init__(inputs, outputs)

    def policy(self, state, noise_object):
        # Ensure the state is a tensor with float32 dtype
        state = tf.convert_to_tensor(state)

        # Remove unnecessary dimensions, ensuring the shape is (1, 5)
        state = tf.squeeze(state)  # This removes dimensions of size 1

        # Reshape the state to be 2D (batch_size, state_size) if it's still 1D
        state = tf.expand_dims(state, axis=0)  # Add batch dimension, making it (1, 5)

        # Predict actions based on the current state
        sampled_actions = self(state)  # Forward pass through the network
        sampled_actions = sampled_actions.numpy()

        # Add noise for exploration
        noise = noise_object()
        noisy_actions = sampled_actions + noise

        # Clip actions to be within bounds
        clipped_actions = np.clip(noisy_actions, self._min_action, self._max_action)

        return clipped_actions
