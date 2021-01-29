# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

""" This module contains architecture of Text Recognition model."""

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.backend import squeeze
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D, MaxPool2D, Reshape, Dense, Permute, LSTMCell, Bidirectional
# from tensorflow.compat.v1.contrib import rnn
# import tensorflow.contrib.slim as slim


class TextRecognition:
    """ Text recognition model definition. """

    def __init__(self, is_training, num_classes, backbone_dropout=0.0):
        self.is_training = is_training
        self.lstm_dim = 256
        self.num_classes = num_classes
        self.backbone_dropout = backbone_dropout

    def __call__(self, inputdata):
        with tf.compat.v1.variable_scope('shadow'):
            features = self.feature_extractor(inputdata=inputdata)
            logits = self.encoder_decoder(inputdata=squeeze(features, axis=1))
            #logits = self.encoder_decoder(inputdata=tf.squeeze(features, axis=1))

        return logits

    # pylint: disable=too-many-locals
    def feature_extractor(self, inputdata):
        """ Extracts features from input text image. """
        
        # with tf.compat.v1.layers.arg_scope([slim.conv2d], padding='SAME',
        #                     weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
        #                     weights_regularizer=slim.l2_regularizer(0.00025),
        #                     biases_initializer=None, activation_fn=None):
        #     with slim.arg_scope([slim.batch_norm], updates_collections=None):
        #         bn0 = slim.batch_norm(inputdata, 0.9, scale=True, is_training=self.is_training,
        #                               activation_fn=None)

        model = Sequential()
        model.add(BatchNormalization(axis=1, momentum=0.9, scale=True, trainable=self.is_training))
        
        model.add(Dropout(rate=self.backbone_dropout, trainable=self.is_training))
        model.add(Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu))
        model.add(BatchNormalization(axis=1, momentum=0.9, scale=True, trainable=self.is_training))
        model.add(MaxPool2D(pool_size=2, strides=2))
        
        model.add(Dropout(rate=self.backbone_dropout, trainable=self.is_training))
        model.add(Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu))
        model.add(BatchNormalization(axis=1, momentum=0.9, scale=True, trainable=self.is_training))
        model.add(MaxPool2D(pool_size=2, strides=2))

        model.add(Dropout(rate=self.backbone_dropout, trainable=self.is_training))
        model.add(Conv2D(filters=256, kernel_size=3, activation=tf.nn.relu))
        model.add(BatchNormalization(axis=1, momentum=0.9, scale=True, trainable=self.is_training))

        model.add(Dropout(rate=self.backbone_dropout, trainable=self.is_training))
        model.add(Conv2D(filters=256, kernel_size=3, activation=tf.nn.relu))
        model.add(BatchNormalization(axis=1, momentum=0.9, scale=True, trainable=self.is_training))
        model.add(MaxPool2D(pool_size=[2, 1], strides=[2, 1]))

        model.add(Dropout(rate=self.backbone_dropout, trainable=self.is_training))
        model.add(Conv2D(filters=512, kernel_size=3, activation=tf.nn.relu))
        model.add(BatchNormalization(axis=1, momentum=0.9, scale=True, trainable=self.is_training))

        model.add(Dropout(rate=self.backbone_dropout, trainable=self.is_training))
        model.add(Conv2D(filters=512, kernel_size=3, activation=tf.nn.relu))
        model.add(BatchNormalization(axis=1, momentum=0.9, scale=True, trainable=self.is_training))
        model.add(MaxPool2D(pool_size=[2, 1], strides=[2, 1]))

        model.add(Dropout(rate=self.backbone_dropout, trainable=self.is_training))
        model.add(Conv2D(filters=512, kernel_size=2, strides=[2, 1], activation=tf.nn.relu))
        model.add(BatchNormalization(axis=1, momentum=0.9, scale=True, trainable=self.is_training))

        return model
        # bn0 = tf.keras.layers.BatchNormalization(input=inputdata, axis=1, momentum=0.9, scale=True, training=self.is_training)
        
        # dropout1 = tf.keras.layers.Dropout(input=bn0, rate=self.backbone_dropout, training=self.is_training)
        # conv1 = tf.keras.layers.Conv2D(input=dropout1, filters=64, kernel_size=3, activation=tf.nn.relu)
        # bn1 = tf.keras.layers.BatchNormalization(input=conv1, axis=1, momentum=0.9, scale=True, training=self.is_training)
        # pool1 = tf.keras.layers.MaxPool2D(input=bn1, pool_size=2, strides=2)

        # dropout2 = tf.keras.layers.Dropout(input=pool1, rate=self.backbone_dropout, training=self.is_training)
        # conv2 = tf.keras.layers.Conv2D(input=dropout2, filters=128, kernel_size=3, activation=tf.nn.relu)
        # bn2 = tf.keras.layers.BatchNormalization(input=conv2, axis=1, momentum=0.9, scale=True, training=self.is_training)
        # pool2 = tf.keras.layers.MaxPool2D(input=bn2, pool_size=2, strides=2)

        # dropout3 = tf.keras.layers.Dropout(input=pool2, rate=self.backbone_dropout, training=self.is_training)
        # conv3 = tf.keras.layers.Conv2D(input=dropout3, filters=256, kernel_size=3, activation=tf.nn.relu)
        # bn3 = tf.keras.layers.BatchNormalization(input=conv3, axis=1, momentum=0.9, scale=True, training=self.is_training)

        # dropout4 = tf.keras.layers.Dropout(input=bn3, rate=self.backbone_dropout, training=self.is_training)
        # conv4 = tf.keras.layers.Conv2D(input=dropout4, filters=256, kernel_size=3, activation=tf.nn.relu)
        # bn4 = tf.keras.layers.BatchNormalization(input=conv4, axis=1, momentum=0.9, scale=True, training=self.is_training)
        # pool4 = tf.keras.layers.MaxPool2D(input=bn4, pool_size=[2, 1], strides=[2, 1])

        # dropout5 = tf.keras.layers.Dropout(input=pool4, rate=self.backbone_dropout, training=self.is_training)
        # conv5 = tf.keras.layers.Conv2D(dropout5, filters=512, kernel_size=3, activation=tf.nn.relu)
        # bn5 = tf.keras.layers.MaxPool2D(conv5, axis=1, momentum=0.9, scale=True, training=self.is_training)

        # dropout6 = tf.keras.layers.Dropout(input=bn5, rate=self.backbone_dropout, training=self.is_training)
        # conv6 = tf.keras.layers.Conv2D(dropout6, filters=512, kernel_size=3, activation=tf.nn.relu)
        # bn6 = tf.keras.layers.BatchNormalization(conv6, axis=1, momentum=0.9, scale=True, training=self.is_training)
        # pool6 = tf.keras.layers.MaxPool2D(bn6, pool_size=(2, 1), strides=(2, 1))

        # dropout7 = tf.keras.layers.Dropout(input=pool6, rate=self.backbone_dropout, training=self.is_training)
        # conv7 = tf.keras.layers.Conv2D(dropout7, filters=512, kernel_size=2, strides=[2, 1], activation=tf.nn.relu)
        # bn7 = tf.keras.layers.BatchNormalization(conv7, axis=1, momentum=0.9, scale=True, training=self.is_training)
        
        #bn0 = tf.compat.v1.layers.batch_normalization(inputdata, momentum=0.9, scale=True, training=self.is_training)#, activation_fn=None)
        #dropout1 = tf.compat.v1.layers.dropout(bn0, rate=1.0 - self.backbone_dropout, training=self.is_training)
        # conv1 = tf.compat.v1.layers.conv2d(dropout1, filters=64, kernel_size=3, activation=tf.nn.relu)
        # bn1 = tf.compat.v1.layers.batch_normalization(conv1, momentum=0.9, scale=True, training=self.is_training)
        # pool1 = tf.compat.v1.layers.max_pooling2d(bn1, pool_size=2, strides=2)
        
        # dropout2 = tf.compat.v1.layers.dropout(pool1, rate=1.0 - self.backbone_dropout, training=self.is_training)
        
        # conv2 = tf.compat.v1.layers.conv2d(dropout2, filters=128, kernel_size=3, activation=tf.nn.relu)
        # bn2 = tf.compat.v1.layers.batch_normalization(conv2, momentum=0.9, scale=True, training=self.is_training)
        # pool2 = tf.compat.v1.layers.max_pooling2d(bn2, pool_size=2, strides=2)
        
        # dropout3 = tf.compat.v1.layers.dropout(pool2, rate=1.0 - self.backbone_dropout, training=self.is_training)
        # conv3 = tf.compat.v1.layers.conv2d(dropout3, filters=256, kernel_size=3, activation=tf.nn.relu)
        # bn3 = tf.compat.v1.layers.batch_normalization(conv3, momentum=0.9, scale=True, training=self.is_training)
        
        # dropout4 = tf.compat.v1.layers.dropout(bn3, rate=1.0 - self.backbone_dropout, training=self.is_training)
        # conv4 = tf.compat.v1.layers.conv2d(dropout4, filters=256, kernel_size=3, activation=tf.nn.relu)
        # bn4 = tf.compat.v1.layers.batch_normalization(conv4, momentum=0.9, scale=True, training=self.is_training)
        # pool4 = tf.compat.v1.layers.max_pooling2d(bn4, pool_size=[2, 1], strides=[2, 1])

        # dropout5 = tf.compat.v1.layers.dropout(pool4, rate=1.0 - self.backbone_dropout, training=self.is_training)
        # conv5 = tf.compat.v1.layers.conv2d(dropout5, filters=512, kernel_size=3, activation=tf.nn.relu)
        # bn5 = tf.compat.v1.layers.batch_normalization(conv5, momentum=0.9, scale=True, training=self.is_training)

        # dropout6 = tf.compat.v1.layers.dropout(bn5, rate=1.0 - self.backbone_dropout, training=self.is_training)
        # conv6 = tf.compat.v1.layers.conv2d(dropout6, filters=512, kernel_size=3, activation=tf.nn.relu)
        # bn6 = tf.compat.v1.layers.batch_normalization(conv6, momentum=0.9, scale=True, training=self.is_training)
        # pool6 = tf.compat.v1.layers.max_pooling2d(bn6, pool_size=(2, 1), strides=(2, 1))

        # dropout7 = tf.compat.v1.layers.dropout(pool6, rate=1.0 - self.backbone_dropout, training=self.is_training)
        # conv7 = tf.compat.v1.layers.conv2d(dropout7, filters=512, kernel_size=2, strides=[2, 1], activation=tf.nn.relu)
        # bn7 = tf.compat.v1.layers.batch_normalization(conv7, momentum=0.9, scale=True, training=self.is_training)

        # return bn7

    def encoder_decoder(self, inputdata):
        """ LSTM-based encoder-decoder module. """
        model = Sequential()
        model.add(Input(shape=(None, inputdata)))
        
        with tf.compat.v1.variable_scope('LSTMLayers'):
            [batch_size, width, _] = inputdata.get_shape().as_list()

            with tf.compat.v1.variable_scope('encoder'):
                forward_cells = []
                backward_cells = []

                for _ in range(2):
                    forward_cells.append(LSTMCell(self.lstm_dim))
                    backward_cells.append(LSTMCell(self.lstm_dim))
                #     # forward_cells.append(tf.nn.rnn_cell.LSTMCell(self.lstm_dim))
                #     # backward_cells.append(tf.nn.rnn_cell.LSTMCell(self.lstm_dim))
                model.add(Bidirectional(forward_cells, backward_layer=backward_cells, dtype=tf.float32))
                # encoder_layer = Sequential()
                # encoder_layer.add(Input(shape=(None, inputdata)))
                # encoder_layer.add(Bidirectional(forward_cells, backward_layer=backward_cells, dtype=tf.float32))
                # encoder_layer = tf.keras.layers.Bidirectional(forward_cells, backward_layer=backward_cells, dtype=tf.float32)
                # encoder_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                #     forward_cells, backward_cells, inputdata, dtype=tf.float32)

            with tf.compat.v1.variable_scope('decoder'):
                forward_cells = []
                backward_cells = []

                for _ in range(2):
                    forward_cells.append(LSTMCell(self.lstm_dim))
                    backward_cells.append(LSTMCell(self.lstm_dim))

                # decoder_layer = Sequential()
                # decoder_layer.add(Input(shape=(None, encoder_layer)))
                # decoder_layer.add(Bidirectional(forward_cells, backward_layer=backward_cells, dtype=tf.float32))
                model.add(Bidirectional(forward_cells, backward_layer=backward_cells, dtype=tf.float32))
                # decoder_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                #     forward_cells, backward_cells, encoder_layer, dtype=tf.float32)
            
            #model.add(decoder_layer)
            model.add(Reshape([batch_size * width, -1]))
            model.add(Dense(self.num_classes))
            model.add(Reshape([batch_size, width, self.num_classes]))
            model.add(Permute((1, 0, 2)))

        return model
            #rnn_reshaped = tf.keras.layers.Reshape()
        #     rnn_reshaped = tf.reshape(decoder_layer, [batch_size * width, -1])

        #     logits = tf.compat.v1.layers.dense(rnn_reshaped, units=self.num_classes, activation=None)  #fully_connected(rnn_reshaped, self.num_classes, activation_fn=None)
        #     logits = tf.reshape(logits, [batch_size, width, self.num_classes])
        #     rnn_out = tf.transpose(logits, (1, 0, 2))

        # return rnn_out
