import glob

import hdf5storage
import torch
from torch.utils.data import TensorDataset


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
