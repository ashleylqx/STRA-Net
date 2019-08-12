from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
# from keras.layers.recurrent import _generate_dropout_mask
import keras.layers.recurrent as RE

import numpy as np
import warnings
from keras.engine import InputSpec, Layer
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.layers import Recurrent
# from keras.layers import RNN

from keras.utils.generic_utils import has_arg

# from keras.layers.convolutional_recurrent import ConvRNN2D
from keras.layers.convolutional_recurrent import ConvRecurrent2D
              

class ConvGRU2D(ConvRecurrent2D):
    """Convolutional GRU.

       Adapted from the implementation of Convolutional LSTM in Keras
    """

    @interfaces.legacy_convlstm2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 # unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ConvGRU2D, self).__init__(filters,
                                         kernel_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format,
                                         dilation_rate=dilation_rate,
                                         return_sequences=return_sequences,
                                         go_backwards=go_backwards,
                                         stateful=stateful,
                                         **kwargs)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
            
        batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:])
        
        if self.stateful:
            self.reset_states()
        else:
            self.states = [None]

        if self.data_format == 'channels_first':
            channel_axis = 2
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        state_shape = [None] * 3
        state_shape[channel_axis] = input_dim
        state_shape = tuple(state_shape)
        self.state_spec = [InputSpec(shape=state_shape)]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 3) # ok
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 3) # ok

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        # pending_end
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters * 3,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :, :, :self.filters]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :, :, :self.filters]
        self.kernel_r = self.kernel[:, :, :, self.filters: self.filters * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:, :, :, self.filters: self.filters * 2]
        self.kernel_h = self.kernel[:, :, :, self.filters * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, :, :, self.filters * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.filters]
            self.bias_r = self.bias[self.filters: self.filters * 2]
            self.bias_h = self.bias[self.filters * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True
        

    def get_initial_state(self, inputs):
        initial_state = K.zeros_like(inputs)
        initial_state = K.sum(initial_state, axis=1)
        shape = list(self.kernel_shape)
        shape[-1] = self.filters
        initial_state = self.input_conv(initial_state,
                                        K.zeros(tuple(shape)),
                                        padding=self.padding)

        initial_states = [initial_state]
        return initial_states

    def reset_states(self):
        if not self.stateful:
            raise RuntimeError('Layer must be stateful.')
        input_shape = self.input_spec[0].shape
        output_shape = self.compute_output_shape(input_shape)
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size). '
                             'Got input shape: ' + str(input_shape))
        if self.return_sequences:
            if self.return_state:
                output_shape = output_shape[1]
            else:
                output_shape = (input_shape[0],) + output_shape[2:]
        else:
            if self.return_state:
                output_shape = output_shape[1]
            else:
                output_shape = (input_shape[0],) + output_shape[1:]

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros(output_shape))
            
        else:
            self.states = [K.zeros(output_shape)]

    def get_constants(self, inputs, training=None):
        # change all 4 into 3
        constants = []
        if self.implementation == 0 and 0 < self.dropout < 1:
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones += 1

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(3)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.recurrent_dropout < 1:
            shape = list(self.kernel_shape)
            shape[-1] = self.filters
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones = self.input_conv(ones, K.zeros(shape),
                                   padding=self.padding)
            ones += 1.

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def reccurent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    def step(self, inputs, states):
        assert len(states) == 3
        h_tm1 = states[0]
        dp_mask = states[1]
        rec_dp_mask = states[2]

        x_z = self.input_conv(inputs * dp_mask[0], self.kernel_z, self.bias_z,
                              padding=self.padding)
        x_r = self.input_conv(inputs * dp_mask[1], self.kernel_r, self.bias_r,
                              padding=self.padding)
        x_h = self.input_conv(inputs * dp_mask[2], self.kernel_h, self.bias_h,
                              padding=self.padding)
        
        h_z = self.reccurent_conv(h_tm1 * rec_dp_mask[0],
                                  self.recurrent_kernel_z)
        h_r = self.reccurent_conv(h_tm1 * rec_dp_mask[1],
                                  self.recurrent_kernel_r)       

        z = self.recurrent_activation(x_z + h_z)
        r = self.recurrent_activation(x_r + h_r)

        h_h = self.reccurent_conv(r * h_tm1 * rec_dp_mask[2],
                                  self.recurrent_kernel_h)
        hh = self.activation(x_h + h_h)
        
        h = (1 - z) * h_tm1 + z * hh
        return h, [h]

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  # 'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

