# -*- coding: utf-8 -*-
# @Time    : 2019/5/16
# @Author  : Godder
# @Github  : https://github.com/WangGodder

from torch.utils.data import Dataset
import numpy as np
import xlrd
import torch
import os

FOLDER_URL = "./data/"


class DataFolder(Dataset):
    def __init__(self, folder: str, seq_length: int, data_length=1, train_pro=0.8, preorder=False):
        """
        init DataFolder
        :param folder: the url of the folder where data csv store
        :param data_length: the length of input data (each day = 1, each week = 7, each month = 30), default: 1
        :param seq_length: the seq length of data (if you want use first 46 point to forecast the 47th point, then seq_length should be 46)
        :param train_pro: the probability of train data in all data
        """
        super(DataFolder, self).__init__()
        self.step = 'train'
        self.folder_url = folder
        self.data_length = data_length
        # read all data
        self.all_data = self._read_all_data()
        if preorder:
            self.pre_order()
        if seq_length > self.all_data.shape[1]:
            raise ValueError("seq length: %d must smaller than the length of seq data: %d" % (seq_length, self.all_data.shape[1]))
        self.seq_length = seq_length
        self.all_num = self.all_data.shape[0]
        self.train_num = int(self.all_num * train_pro) - data_length - 1

    def _read_all_data(self):
        """
        read all data from csv files from csv storing folder
        :return: a numpy array with shape (total num, 48) where 48 is the seq length of each daily data.
        """

        files = os.listdir(self.folder_url)
        f = []
        for file in files:
            if not os.path.isdir(file):
                f.append(os.path.join(FOLDER_URL,file))

        tableWashed = []
        for i in range(0, len(f)):
            data = xlrd.open_workbook(f[i], formatting_info=False)
            table = data.sheets()[0]
            # read the data for row 20th
            for j in range(20, table.nrows):
                judge = table.row_values(j, 2, )[0]
                if (judge != ''):
                    tableWashed.append(table.row_values(j, 2, ))
        return np.array(tableWashed)

    def pre_order(self):
        Max = np.max(self.all_data,axis=0)
        Min = np.min(self.all_data,axis=0)
        Div = Max - Min
        self.all_data = (self.all_data - (Max + Min / 2)) / (Div / 2)
        #self.all_data = (self.all_data - (Max+Min)*0.5 )/ (Div*0.5)
    def train(self):
        self.step = 'train'

    def valid(self):
        self.step = 'valid'

    def __len__(self):
        if self.step is 'train':
            return self.train_num
        else:
            return self.all_num - self.train_num - (self.data_length + 1) * 2

    def __getitem__(self, item):
        """
        return data for dataloader or index item.
        :param item:
        :return: input tensor with shape (data_length, seq_length), label tensor with shape (data_length, 1)
        """
        data = self.all_data[item: item + self.data_length, 0: self.seq_length]
        data = data.reshape(self.seq_length * self.data_length)
        label = self.all_data[item + self.data_length: item + 1 + self.data_length, 0: self.seq_length]
        data = torch.tensor(data).float()
        label = torch.tensor(label).float()
        return data, label


def get_dataset(seq_length, data_length, pre_order):
    return DataFolder(FOLDER_URL, seq_length, data_length, preorder=pre_order)

