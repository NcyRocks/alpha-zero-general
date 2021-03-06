import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys

#import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .Connect4NNet_transformer import Connect4NNet as onnet

#from livelossplot import PlotLossesKeras

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.24,
    'epochs': 20,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        hist = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], 
                                   batch_size = args.batch_size, epochs = args.epochs)
        #hist = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, 
                            #epochs = args.epochs, callbacks=[PlotLossesKeras()])
        return hist

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        print(filepath)
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        #print(folder, filename)
        filepath = os.path.join(folder, filename)
        #print(filepath)
        #print(os.getcwd())
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
        
    def set_num(self, number):
        self.number = number