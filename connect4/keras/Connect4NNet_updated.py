import sys
sys.path.append('..')
from utils import *
import os
import argparse

import tensorflow as tf
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)

gpus = tf.config.experimental.list_physical_devices('GPU')

# change gpus[0] back to a 0 if it isn't and you don't want it another value.
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*40)])

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""
class Connect4NNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        ## my own additions --- transformer.

        self.emb_layer = TokenAndPositionEmbedding(self.board_x*self.board_y, 3, 64) # 64 divisible by 8 = num_heads
        self.transformer_block = TransformerBlock(64, 8, 32) # emb_dim, num_heads, ff_net_hidden_lay
        #self.last_layer = layers.Dense(9) # x*y board dim.
        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x*self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        #print(x_image.shape)
        #x_image = tf.reshape(x_image, shape=(-1, 9, 1))
        #x_image =
        #print("fist", x_image.shape)
        x_image = self.emb_layer(x_image)
        #print("emb",x_image.shape)
        x_image = self.transformer_block(x_image)
        x_image = tf.reshape(x_image, shape=(-1, x_image.shape[2], x_image.shape[1]))
        x_image = layers.GlobalAveragePooling1D()(x_image)
        #print("test!",x_image.shape)
        x_image = tf.reshape(x_image, shape=(-1, self.board_x, self.board_y, 1)) # 64 is batch size. Hardcoded here.
        
        #print("REACH")
        #print(x_image.shape)
        #x_image=x_image[:,-1,:] # -1 to take transformer at the very end.
        x_image = tf.reshape(x_image, shape = (-1, self.board_x, self.board_y, 1)) # for 3*3 tic-tac-toe board size
        #print(x_image.shape)
        
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv3)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        
        h_conv5 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv4))) 
        
        h_conv6 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv5))) 
        
        h_conv7 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv6))) 
        
        h_conv8 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv7))) 
        
        
        h_conv4_flat = Flatten()(h_conv8)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        #print(tf.trainable_variables())
        
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))

# https://keras.io/examples/nlp/text_classification_with_transformer/
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, output_dim = 42): # hardcode out_DIM
        # output_dim changes based on the game. It is the number of board positions.
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),] #output_dim
        )
        
        self.last_layer = layers.Dense(output_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        #print("REACH")
        #print("out1",out1.shape)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        #print("ffn_output",ffn_output.shape)
        return self.layernorm2(out1 + ffn_output)
        #output = self.layernorm2(out1 + ffn_output)
        #print("FOCUS",output.shape)
        #output = tf.reshape(output, shape=(output.shape[3], -1, 9)) # 64 is batch size. Hardcoded here.
        #print("FOCUS",output.shape)
        #return self.last_layer(output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, board_dim, num_players_plus_1, embed_dim):
        '''
        board_dim: width * height of the board. 
        num_players_plus_1: Represents the number of possible board states.
            e.g. (-1, 0, 1) for 2 player game.
        '''
        super(TokenAndPositionEmbedding, self).__init__()
        self.board_dim = board_dim
        self.board_emb = layers.Embedding(input_dim=num_players_plus_1, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=board_dim, output_dim=embed_dim)

    def call(self, x):
        #maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=self.board_dim, delta=1)
        positions = self.pos_emb(positions)
        x = self.board_emb(x)
        x = tf.reshape(x, shape = (-1, 42, 64)) ## HArdcoded 16/9 here depending on board size.
        #print("fjsljflks", x.shape)
        #print("pos", positions.shape)
        return x + positions
