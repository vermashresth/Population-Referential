from ray.rllib.utils import try_import_tf, try_import_tfp

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

tf = try_import_tf()
tfp = try_import_tfp()

import numpy as np
class MyModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        activation= tf.nn.tanh
        last_layer = layer_out = self.inputs
        i = 1
        hiddens = [256, 256, 256]
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1
        layer_out = tf.keras.layers.Dense(
                    num_outputs,
                    name="fc_out",
                    activation=None,
                    kernel_initializer=normc_initializer(0.01))(last_layer)


        # build a parallel set of hidden layers for the value net
        last_layer = self.inputs
        i = 1
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_value_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1

        value_out = tf.keras.layers.Dense(
            3,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self, i=-1):

        all_vals = tf.reshape(self._value_out, [-1])
        s0, s1, s2 = tf.split(all_vals, num_or_size_splits=3, axis=-1)
        if i==-1:
            a= tf.math.reduce_mean(all_vals, axis=-1, keepdims=False, name=None)

            if len(a.shape)==0:
              return s0
            return a
        else:
            return [s0,s1,s2][i]

    def uncertainity(self):
        all_vals = tf.reshape(self._value_out, [-1])
        cov = tfp.stats.covariance(all_vals,all_vals, sample_axis=0, event_axis=None)
        print(cov, cov.shape, len(cov.shape))

        if len(cov.shape)==0:
          return self._value_out
        return cov
