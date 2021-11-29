from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os

from plotly.graph_objs import *
import plotly
import torch

from numpy import dstack
from pandas import read_csv
from torch.utils.data import TensorDataset

import pickle
import dill
import scipy.io
import hdf5storage
import glob
from torchvision import transforms
from torch.utils.data import DataLoader


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/'
    filenames = list()
    for i in range(19, 37):
        filenames += ['s' + str(i) + '.txt']
    x = load_group(filenames, filepath)
    weight = load_file(filepath + 'p2.txt')
    height = load_file(filepath + 'p1.txt')
    y = load_file(filepath + 'lables.txt')
    return x, y, weight, height


# load the dataset, returns train and test X and y elements
def load_dataset(prefix, train,test):
    # load all train
    trainX, trainy, trainWeight, trainHeight = load_dataset_group(train, prefix)
    # load all test
    testX, testy, testWeight, testHeight = load_dataset_group(test, prefix)

    print(trainX.shape, trainy.shape, testX.shape, testy.shape, trainWeight.shape,trainHeight.shape,
          testHeight.shape, testWeight.shape)

    return trainX, trainy, testX, testy, trainHeight, trainWeight, testHeight, testWeight

# load the dataset, returns train and test X and y elements
def load_dataset_test(prefix=''):
    # load all test
    testX, testy, testWeight, testHeight = load_dataset_group('test3mezi', prefix)

    print(testX.shape, testy.shape, testHeight.shape, testWeight.shape)

    return testX, testy, testHeight, testWeight


def open_data_pickle_one_wheele2(folder, speed_vect=False, base='C:/Users/josef/Desktop/DeepLearning/Jednokolka/preocessedDataNumpy/',
                                 mul=None, muw=None, stdl=None, stdw=None):

    mat = hdf5storage.loadmat(base + folder + '/data.mat')
    # mat = scipy.io.loadmat(base + folder + '/data.mat')

    # train_data = torch.from_numpy(train_data_np).type(torch.FloatTensor)
    if speed_vect:
        speed = torch.Tensor(mat['d19']).type(torch.FloatTensor)
        legLength = torch.Tensor(mat['d20_21'][:, 0]).type(torch.FloatTensor)
        weigth = torch.Tensor(mat['d20_21'][:, 1]).type(torch.FloatTensor)
    else:
        speed = torch.Tensor(mat['d19_21'][:, 0]).type(torch.FloatTensor)
        legLength = torch.Tensor(mat['d19_21'][:, 1]).type(torch.FloatTensor)
        weigth = torch.Tensor(mat['d19_21'][:, 2]).type(torch.FloatTensor)


    train_data = torch.Tensor(mat['d1_18']).type(torch.FloatTensor)

    count=speed.shape[0]

    if not speed_vect:
        speed=(speed.view(count, 1))

    legLength = legLength.view(count, 1)
    if mul is None:
        mul = torch.mean(legLength)
        stdl = torch.std(legLength)
    legLength = (legLength-mul)/stdl

    weigth = weigth.view(count, 1)
    if muw is None:
        muw = torch.mean(weigth)
        stdw = torch.std(weigth)
    weigth = (weigth - muw) / stdw

    train_data = train_data.transpose(1, 2)
    dataset = TensorDataset(train_data, speed, weigth, legLength)

    return dataset, mul, muw, stdl, stdw


def getData(mat):
    speed = torch.Tensor(mat['d19_21'][:, 0]).type(torch.FloatTensor)
    legLength = torch.Tensor(mat['d19_21'][:, 1]).type(torch.FloatTensor)
    weigth = torch.Tensor(mat['d19_21'][:, 2]).type(torch.FloatTensor)

    train_data = torch.Tensor(mat['d1_18']).type(torch.FloatTensor)
    return speed, legLength, weigth, train_data

def getDataset(speed, legLength, weigth, train_data):
    count = speed.shape[0]
    speed = speed.view(count, 1)
    legLength = legLength.view(count, 1)
    weigth = weigth.view(count, 1)

    train_data = train_data.transpose(1, 2)
    return TensorDataset(train_data, speed, weigth, legLength)


def get_number_of_subjects(fold, base='C:/Users/josef/Desktop/DeepLearning/Jednokolka/preocessedDataNumpy/'):
    return len(sorted(glob.glob(base + fold +'/tes_sub*.mat')))


def open_data_pickle_one_wheele_separated(folder, test_subject_num, base='C:/Users/josef/Desktop/DeepLearning/Jednokolka/preocessedDataNumpy/'):

    testNames = sorted(glob.glob(base + folder +'/tes_sub*.mat'))
    trainNames = sorted(glob.glob(base + folder + '/tr_sub*.mat'))

    speedAll=torch.empty(0)
    legAll = torch.empty(0)
    weigthAll = torch.empty(0)
    dataAll = torch.empty(0,  18, 1024)

    for i in range(len(testNames)):
        if i != test_subject_num:
            mat = hdf5storage.loadmat(trainNames[i])
            speed, legLength, weigth, train_data = getData(mat)
            speedAll = torch.cat([speedAll, speed], 0)
            legAll = torch.cat([legAll, legLength], 0)
            weigthAll = torch.cat([weigthAll, weigth], 0)
            dataAll = torch.cat([dataAll, train_data], 0)
        else:
            mat = hdf5storage.loadmat(testNames[i])
            speed, legLength, weigth, train_data = getData(mat)
            testDataset = getDataset(speed, legLength, weigth, train_data)

    trainDataset = getDataset(speedAll, legAll, weigthAll, dataAll)

    return trainDataset, testDataset

def open_data_pickle_one_wheele_separated_validation(folder, test_subject_num, base='C:/Users/josef/Desktop/DeepLearning/Jednokolka/preocessedDataNumpy/'):

    testNames = sorted(glob.glob(base + folder +'/tes_sub*.mat'))
    trainNames = sorted(glob.glob(base + folder + '/tr_sub*.mat'))

    speedAll = torch.empty(0)
    legAll = torch.empty(0)
    weigthAll = torch.empty(0)
    dataAll = torch.empty(0, 18, 1024)

    speedAllVal = torch.empty(0)
    legAllVal = torch.empty(0)
    weigthAllVal = torch.empty(0)
    dataAllVal = torch.empty(0, 18, 1024)

    for i in range(len(testNames)):
        if i != test_subject_num:
            mat = hdf5storage.loadmat(trainNames[i])
            speed, legLength, weigth, train_data = getData(mat)

            length=len(speed)

            speedAllVal = torch.cat([speedAllVal, speed[:int(length / 10) - 1]], 0)
            legAllVal = torch.cat([legAllVal, legLength[:int(length / 10) - 1]], 0)
            weigthAllVal = torch.cat([weigthAllVal, weigth[:int(length / 10) - 1]], 0)
            dataAllVal = torch.cat([dataAllVal, train_data[:int(length / 10) - 1]], 0)

            speedAll = torch.cat([speedAll, speed[int(length / 10):]], 0)
            legAll = torch.cat([legAll, legLength[int(length / 10):]], 0)
            weigthAll = torch.cat([weigthAll, weigth[int(length / 10):]], 0)
            dataAll = torch.cat([dataAll, train_data[int(length / 10):]], 0)

        else:
            mat = hdf5storage.loadmat(testNames[i])
            speed, legLength, weigth, train_data = getData(mat)
            testDataset = getDataset(speed, legLength, weigth, train_data)

    trainDataset = getDataset(speedAll, legAll, weigthAll, dataAll)
    validDataset = getDataset(speedAllVal, legAllVal, weigthAllVal, dataAllVal)

    return trainDataset, testDataset, validDataset


def get_testing_dataset(numWindows):
    speedAll = torch.rand(numWindows)
    legAll = torch.rand(numWindows)
    weigthAll = torch.rand(numWindows)
    dataAll = torch.rand(numWindows, 18, 1024)

    trainDataset = getDataset(speedAll, legAll, weigthAll, dataAll)
    validDataset = getDataset(speedAll, legAll, weigthAll, dataAll)
    testDataset = getDataset(speedAll, legAll, weigthAll, dataAll)

    return trainDataset, testDataset, validDataset

def downscale(dataset, outputSize):

    dataTens=dataset.tensors[0]

    downSampledDat=torch.nn.functional.interpolate(dataTens.permute(0,2,1), size=outputSize, mode='linear')\
        .permute(0,2,1)

    outData= TensorDataset(downSampledDat, dataset.tensors[1], dataset.tensors[2],
                  dataset.tensors[3])
    return outData
