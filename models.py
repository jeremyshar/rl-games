"""Implementations of various models."""

from __future__ import division

import os
# Avoid some logging messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import keras

import abstract


class TableModel(abstract.Model):
  """A model for TicTacToe that explicitly stores all stats in a table."""

  def __init__(self, board_size):
    """Constructor for a TableModel.

    Args:
      board_size: An int, the size of the board.
    """
    self.board_size = board_size
    self.table = {}

  def transform_policy(self, policy, rotation_amt, flipped, reverse_direction):
    """Transforms a 1D policy vector based on the board's transformation.

    Changes the policy based on the rotation, mirroring of the board. Also can
    be done in the reverse direction.

    Args:
      policy: The input 1D policy array to be transformed.
      rotation_amt: An int between 0 and 3, the number of 90 degree rotations.
      flipped: A boolean, whether or not the policy should be reflected.
      reverse_direction: A boolean, whether or not the transformation should be
        done in reverse/undone.

    Returns:
      The resulting transformed policy as a 1D array.
    """
    reshaped = np.reshape(policy, (self.board_size, self.board_size))
    if reverse_direction:
      mirrored = np.fliplr(reshaped) if flipped else reshaped
      result = np.rot90(mirrored, k=4 - rotation_amt)
    else:
      rotated = np.rot90(reshaped, k=rotation_amt)
      result = np.fliplr(rotated) if flipped else rotated
    return np.reshape(result, (-1))

  def lookup(self, state):
    """Looks up a state in the table.

    Gets the stats for the position if it's in the table, along with the
    transformation parameters required to turn the input state into the state in
    the table. For convenience, returns the existing state in the table. If the
    state doesn't exist in the table, returns None.

    Args:
      state: The input state to look up in the table.

    Returns:
      None if the state isn't in the table, or the state's target value, target
      policy, and weight as a tuple, along with the number of rotations required
      to match the stored state, whether a reflection was required to match the
      stored key, and the actual stored key.
    """
    for flipped in [False, True]:
      for rotation_amt in range(4):
        rotated = np.rot90(state, k=rotation_amt, axes=(0, 1))
        symmetry = np.flip(rotated, axis=1) if flipped else rotated
        if symmetry.tostring() in self.table:
          value, policy, weight = self.table[symmetry.tostring()]
          policy = self.transform_policy(
              policy, rotation_amt, flipped, reverse_direction=True)
          return (value, policy,
                  weight), rotation_amt, flipped, symmetry.tostring()
    return None

  def insert_or_update(self, state, stats):
    """Inserts the state and stats into the table.

    If the state doesn't exist in the table, inserts the stats directly,
    otherwise averages these stats with the stats from the existing entry (using
    the weights).

    Args:
      state: The state to insert into the table.
      stats: The stats associated with the state, a tuple of (value, policy,
        weight).
    """
    lookup_result = self.lookup(state)
    if lookup_result is None:
      self.table[state.tostring()] = stats
    else:
      (v, p, w), rotation_amt, flipped, key = lookup_result
      value, policy, weight = stats
      new_weight = w + weight
      new_value = (v * w + value * weight) / new_weight
      new_policy = (p * w + policy * weight) / new_weight
      new_policy = self.transform_policy(
          new_policy, rotation_amt, flipped, reverse_direction=False)
      self.table[key] = (new_value, new_policy, new_weight)

  def policy(self, state):
    """Returns the policy for the inpout state.

    Args:
      state: the state for which to fetch the policy.

    Returns:
      The policy vector for the input state, or a uniform random policy if the
      state isn't in the table.
    """
    entry = self.lookup(state)
    if entry is not None:
      return entry[0][1]
    else:
      return np.ones(self.board_size * self.board_size) / (
          self.board_size * self.board_size)

  def value(self, state):
    """Returns the value of the input state.

    Args:
      state: The state for which to fetch the value.

    Returns:
      A float, the value for the input state, or 0 if the state isn't in the
      table.
    """
    entry = self.lookup(state)
    if entry is not None:
      return entry[0][0]
    else:
      return 0

  def train(self, data):
    """Updates the table from a list of input data.

    Args:
      data: A list of (state, target_value, target_policy) tuples.
    """
    for datum in data:
      self.insert_or_update(datum[0], (datum[1], datum[2], 1.0))


class TensorFlowModel(abstract.Model):
  """A base class for TensorFlow models."""

  def __init__(self):
    pass

  def policy(self, state):
    """Returns the policy for the inpout state.

    Args:
      state: the state for which to predict the policy.

    Returns:
      The policy vector for the input state.
    """
    return self.session.run(
        self.policy_head, feed_dict={self.input_states: [state]})[0]

  def value(self, state):
    """Returns the value of the input state.

    Args:
      state: The state for which to predict the value.

    Returns:
      A float, the value for the input state.
    """
    return self.session.run(
        self.value_head, feed_dict={self.input_states: [state]})

  def train(self, data):
    """Trains the model from a list of input data.

    Args:
      data: A list of (state, target_value, target_policy) tuples.
    """
    counter = 0
    states = []
    target_values = []
    target_policies = []
    min_loss = float('inf')
    total_loss = 0
    for _ in xrange(self.data_passes):
      np.random.shuffle(data)
      pass_loss = 0
      for state, target_value, target_policy in data:
        states.append(state)
        target_values.append(target_value)
        target_policies.append(target_policy)
        counter += 1
        if counter == self.batch_size:
          loss, _ = self.session.run(
              [self.loss, self.train_op],
              feed_dict={
                  self.input_states: states,
                  self.target_values: target_values,
                  self.target_policies: target_policies
              })
          pass_loss += loss
          counter = 0
          states = []
          target_values = []
          target_policies = []
      total_loss += pass_loss
      min_loss = min(min_loss, pass_loss)
    print 'AVERAGE LOSS: {}, MIN LOSS: {}'.format(total_loss / self.data_passes,
                                                  min_loss)
    if counter != 0:
      self.session.run(
          self.train_op,
          feed_dict={
              self.input_states: states,
              self.target_values: target_values,
              self.target_policies: target_policies
          })


class FullyConnectedModel(TensorFlowModel):
  """A neural net consisting of fully connected layers.

  The model has two 'heads', one of which computes a policy vector and the other
  of which computes a single state value.
  """

  def __init__(self,
               board_size,
               layer_sizes,
               learning_rate=0.001,
               batch_size=20,
               data_passes=1,
               regularization_scale=0.1):
    """Constructor for the fully connected model.

    Args:
      board_size: An int, the size of the board.
      layer_sizes: A list of the sizes of the fully connected layers to
        construct.
      learning_rate: A float, the learning rate to use.
      batch_size: An int, the number of (state, target value, target policy)
        tuples per minibatch.
      data_passes: An int, the number of times to
      regularization_scale: A float, the multiplicative factor used to scale the
        regularization loss.
    """
    self.board_size = board_size
    self.batch_size = batch_size
    self.data_passes = data_passes
    self.graph = tf.get_default_graph()
    self.session = tf.Session(graph=self.graph)
    self.input_states = tf.placeholder(
        dtype=tf.float32, shape=(None, 3, self.board_size, self.board_size))
    regularizer = tf.contrib.layers.l2_regularizer(regularization_scale)
    # Flatten and convert to +1/-1
    self.layers = [
        tf.reshape(self.input_states,
                   [-1, self.board_size * self.board_size * 3]) * 2 - 1
    ]
    for size in layer_sizes:
      self.layers.append(
          tf.layers.dense(
              self.layers[-1],
              size,
              activation=tf.nn.relu,
              kernel_regularizer=regularizer,
              bias_regularizer=regularizer,
              activity_regularizer=regularizer))
    self.value_head = tf.layers.dense(self.layers[-1], 1, activation=tf.nn.relu)
    self.policy_head = tf.nn.softmax(
        tf.layers.dense(
            self.layers[-1],
            self.board_size * self.board_size,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            activity_regularizer=regularizer))
    self.target_values = tf.placeholder(dtype=tf.float32)
    self.target_policies = tf.placeholder(
        dtype=tf.float32, shape=(None, self.board_size * self.board_size))
    self.value_loss = tf.reduce_sum(
        tf.square(self.value_head - self.target_values))
    self.loss = tf.reduce_sum(
        tf.where(
            tf.greater(self.target_policies, -0.5),
            tf.square(self.target_policies - self.policy_head),
            tf.zeros_like(
                self.policy_head))) + tf.losses.get_regularization_loss()
    self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    self.session.run(tf.global_variables_initializer())


class ConvolutionalModel(TensorFlowModel):
  """A neural net consisting of convolutional layers.

  The model has two 'heads', one of which computes a policy vector and the other
  of which computes a single state value.
  """

  def __init__(self,
               board_size,
               layer_sizes,
               learning_rate=0.001,
               batch_size=20,
               data_passes=1,
               use_dropout=False):
    """Constructor for the convolutional model.

    Args:
      board_size: An int, the size of the board.
      layer_sizes: A list of the sizes of the convolutional layers to construct.
      learning_rate: A float, the learning rate to use.
      batch_size: An int, the number of (state, target value, target policy)
        tuples per minibatch.
      data_passes: An int, the number of times to
      use_dropout: A boolean, whether to use dropout regularization on the
        layers.
    """
    self.board_size = board_size
    self.batch_size = batch_size
    self.data_passes = data_passes
    self.graph = tf.get_default_graph()
    self.session = tf.Session(graph=self.graph)
    self.input_states = tf.placeholder(
        dtype=tf.float32, shape=(None, 3, self.board_size, self.board_size))
    self.layers = [
        tf.reshape(
            tf.layers.conv2d(
                inputs=tf.transpose(self.input_states, [0, 2, 3, 1]),
                filters=layer_sizes[0],
                kernel_size=(2, 2),
                activation=tf.nn.relu,
                data_format='channels_last'),
            [
                -1,
                layer_sizes[0] * (self.board_size - 1) * (self.board_size - 1)
            ])
    ]
    for size in layer_sizes[1:]:
      self.layers.append(
          tf.layers.dense(
              inputs=self.layers[-1], units=size, activation=tf.nn.relu))
      if use_dropout:
        self.layers.append(tf.layers.dropout(inputs=self.layers[-1], rate=0.4))
    self.value_head = tf.layers.dense(
        inputs=self.layers[-1], units=1, activation=tf.nn.relu)
    self.policy_head = tf.nn.softmax(
        tf.layers.dense(self.layers[-1], self.board_size * self.board_size))
    self.target_values = tf.placeholder(dtype=tf.float32)
    self.target_policies = tf.placeholder(
        dtype=tf.float32, shape=(None, self.board_size * self.board_size))
    self.value_loss = tf.reduce_sum(
        tf.square(self.value_head - self.target_values))
    self.loss = tf.reduce_sum(
        tf.where(
            tf.greater(self.target_policies, -0.5),
            tf.square(self.target_policies - self.policy_head),
            tf.zeros_like(self.policy_head)))
    self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    self.session.run(tf.global_variables_initializer())


class KerasModel(abstract.Model):
  """A Keras-based model combining convolutional and fully-connected layers."""

  def __init__(self,
               game,
               shared_architecture=(),
               policy_head_architecture=(),
               value_head_architecture=(),
               batch_size=200,
               data_passes=2000):
    """Constructor for the Keras model.

    Consists of shared convolutional layers, with separate policy and value
    'heads' of fully connected layers.

    Args:
      game: The game the model is learning to play.
      shared_architecture: A list of ints, the sizes of the shared convolutional
        intermediate network layers.
      policy_head_architecture: A list of ints, the sizes of the dense
        intermediate network layers for the policy head.
      value_head_architecture: A list of ints, the sizes of the dense
        intermediate network layers for the value head.
      batch_size: An int, the size of the batch to use during training.
      data_passes: An int, the number of times to go over the training data.
    """
    self.game = game
    self.batch_size = batch_size
    self.data_passes = data_passes
    self.input_layer = keras.layers.Input(shape=self.game.state_shape())
    self.shared_layers = [self.input_layer]
    for layer_size in shared_architecture:
      self.shared_layers.append(
          keras.layers.Dense(layer_size,
                             activation='relu')(self.shared_layers[-1]))
    self.shared_layers.append(keras.layers.Flatten()(self.shared_layers[-1]))
    self.policy_head_layers = [self.shared_layers[-1]]
    for layer_size in policy_head_architecture:
      self.policy_head_layers.append(
          keras.layers.Dense(layer_size,
                             activation='relu')(self.policy_head_layers[-1]))
    self.raw_policy = keras.layers.Dense(
        self.game.action_space(), activation='relu')(
            self.policy_head_layers[-1])
    self.normalized_policy = keras.layers.Softmax()(self.raw_policy)
    self.value_head_layers = [self.shared_layers[-1]]
    for layer_size in value_head_architecture:
      self.value_head_layers.append(
          keras.layers.Dense(layer_size,
                             activation='relu')(self.value_head_layers[-1]))
    self.value_prediction = keras.layers.Dense(
        1, activation='tanh')(
            self.value_head_layers[-1])
    self.model = keras.models.Model(
        inputs=self.input_layer,
        outputs=[self.normalized_policy, self.value_prediction])
    self.model.compile(
        optimizer='adam',
        loss=['binary_crossentropy', 'mean_squared_error'],
        loss_weights=[10.0, 1.0])

  def value(self, state):
    """Returns the value/predicted outcome of the current state."""
    return self.model.predict_on_batch(np.expand_dims(state, 0))[1][0]

  def policy(self, state):
    """Returns a policy vector for the input state."""
    res = self.model.predict_on_batch(np.expand_dims(state, 0))[0][0]
    return res

  def train(self, data):
    """Trains the model from a list of input data.

    Args:
      data: A list of (state, target_value, target_policy) tuples.
    """
    states = []
    target_values = []
    target_policies = []
    for state, target_value, target_policy in data:
      states.append(state)
      target_values.append(target_value)
      target_policies.append(np.clip(target_policy, 0, None))
    self.model.fit(
        x=np.array(states),
        y=[np.array(target_policies),
           np.array(target_values)],
        batch_size=self.batch_size,
        epochs=self.data_passes)
