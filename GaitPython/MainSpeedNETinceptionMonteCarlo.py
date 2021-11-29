import gc

from vrae.utils import *
import torch
from datetime import datetime
from InceptionSpeedNET import InceptionSpeedNET
import sys
from myUtils import TensorboardLogger
import argparse
import numpy
import random

# tensorboard --logdir=G:\deepLearn\localLog\compare --host localhost --port 8088
# chzba 0.043 /media/data2/justjo/datasets/jednokolka/ 2 16 /home/justjo/timeseries-clustering-vae/log/LogInception/

# --datasetBasePath  G:/deepLearn/jednokolka/ --logPath G:\deepLearn\localLog\perceiver\ --logPath2 G:\deepLearn\localLog\perceiverAll\ --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L' --startfromsubj 0
#        --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L'

# --datasetBasePath /media/data2/justjo/datasets/jednokolka/ --logPath /home/justjo/timeseries-clustering-vae/log/Perceiver/ --logPath2 /home/justjo/timeseries-clustering-vae/log/PerceiverAll/
#       --cuda 2 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L' --startfromsubj 0

# --datasetBasePath C:/Users/josef/PycharmProjects/jednokolka/ --logPath2 C:/Users/josef/PycharmProjects/Log/perceiverAll/
# --logPath C:/Users/josef/PycharmProjects/Log/perceiver/ --cuda 0 --sensor 0 --learningRate 0.001 --tag MonteCarlo --mess 'MonteCarlo-startFrom1' --startfromsubj 1 --LogParamLoadPath C:\Users\josef\PycharmProjects\Log\perceiverAll\-2021-08-01-01-45-'MonteCarlo'\


if __name__ == "__main__":
    # --datasetBasePath  G:/deepLearn/jednokolka/ --logPath G:\deepLearn\localLog\perceiver --logPath2 G:\deepLearn\localLog\perceiverAll --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L'
    parser = argparse.ArgumentParser(add_help=False, description='Process some integers.')
    parser.add_argument('--datasetBasePath', help='Path1')
    parser.add_argument('--logPath', help='Array of integers')
    parser.add_argument('--logPath2', help='Array of integers')
    parser.add_argument('--cuda', help='Array of integers', type=int)
    # parser.add_argument('--latent', help='Array of integers', type=int)
    parser.add_argument('--sensors', help='Array of integers', nargs='+', type=int)
    parser.add_argument('--learningRate', help='Array of integers', type=float)
    parser.add_argument('--tag', help='Array of integers')
    parser.add_argument('--mess', help='Array of integers')
    parser.add_argument('--startfromsubj', default=0, help='Array of integers', type=int)
    parser.add_argument('--LogParamLoadPath', help='Array of integers')
    parser.add_argument('--rep', help='Array of integers')

    args = parser.parse_args()

    datasetBasePath = args.datasetBasePath
    logPath = args.logPath
    cudaNum = args.cuda
    learning_rate = args.learningRate

    batch_size = 32
    n_epochs = 120

    if not args.rep:
        args.rep=1

    for x in range(int(args.rep)):

        startIndex = 0
        for x in [f.path for f in os.scandir(args.logPath2) if f.is_dir()]:
            if ('MonteCarlo' in x):
                if (int(x.split('\'')[1].split('MonteCarlo-')[1]) > startIndex):
                    startIndex = int(x.split('\'')[1].split('MonteCarlo-')[1])

        args.tag = args.mess = '\'MonteCarlo-' + str(startIndex + 1) + '\''

        n_filters = random.choice([2, 4, 8, 16, 32])  # ladit 2^n [2,4,8,16,32]
        kernel_sizes = random.choice([[5, 11, 21], [11, 21, 41], [21, 41, 81]])  # ladit (5/11/21 11/21/41 21/41/81)
        bottleneck_channels = random.choice([2, 4, 8])  # ladit 2 4 8

        if args.LogParamLoadPath:
            pathLog = args.LogParamLoadPath + 'log.txt'
            # Using readline()
            file1 = open(pathLog, 'r')

            while True:
                # Get next line from file
                line = file1.readline()
                if 'n_filters ' in line:
                    n_filters = int(line.split()[-1])
                if 'kernel_sizes ' in line:
                    kernel_sizes = float(line.split()[-1])
                if 'bottleneck_channels ' in line:
                    bottleneck_channels = int(line.split()[-1])
                if not line:
                    break

        listSens = []
        sensors = [int(d) for d in args.sensors]
        for s in sensors:
            listSens.extend(list(range(0 + s * 6, 6 + s * 6)))
        number_of_features = len(listSens)

        print("Program start time =", datetime.now().strftime("%H:%M:%S"))

        sub_num = get_number_of_subjects('separBig', base=datasetBasePath)

        logger = TensorboardLogger(args.logPath2, mess=args.mess)
        logger.log_txt("n_filters ", n_filters)
        logger.log_txt("kernel_sizes ", kernel_sizes)
        logger.log_txt("bottleneck_channels ", bottleneck_channels)

        for i in range(args.startfromsubj, sub_num):

            if 'test_dataset' in globals():
                del test_dataset
            if 'valid_dataset' in globals():
                del valid_dataset
            if 'train_dataset' in globals():
                del train_dataset

            if 'test_dataset_sub' in globals():
                del test_dataset_sub
            if 'valid_dataset_sub' in globals():
                del valid_dataset_sub
            if 'train_dataset_sub' in globals():
                del train_dataset_sub
            if 'test_dataset_sub' in globals():
                del test_dataset_sub

            if 'model' in globals():
                del model

            gc.collect()

            train_dataset, test_dataset, valid_dataset = open_data_pickle_one_wheele_separated_validation('separBig', i,
                                                                                                          base=datasetBasePath)

            test_dataset_sub = TensorDataset(test_dataset.tensors[0][:, :, listSens],
                                             test_dataset.tensors[1], test_dataset.tensors[2], test_dataset.tensors[3])
            train_dataset_sub = TensorDataset(train_dataset.tensors[0][:, :, listSens],
                                              train_dataset.tensors[1], train_dataset.tensors[2], train_dataset.tensors[3])
            valid_dataset_sub = TensorDataset(valid_dataset.tensors[0][:, :, listSens],
                                              valid_dataset.tensors[1], valid_dataset.tensors[2], valid_dataset.tensors[3])
            model = InceptionSpeedNET(
                filters=n_filters,
                learning_rate=learning_rate,
                kernel_sizes=kernel_sizes,
                bottleneckLayers=bottleneck_channels,
                cudaNum=cudaNum,
                batch_size=batch_size,
                num_features=number_of_features,
                logerPath=logPath, mess=args.tag + 'subj-' + str(i))

            print("Learning started =", datetime.now().strftime("%H:%M:%S"))
            print("Subject =", i)

            minTestErr, meanLast = model.fit(n_epochs, train_dataset_sub, test_dataset_sub, valid_dataset=valid_dataset_sub,
                                             shuffleTrain=True)

            logger.epoch = i
            logger.step(1)
            logger.log_value('subject', i)
            logger.log_value('minTestErr', minTestErr)
            logger.log_value('meanLastTenTestErr', meanLast)

        # torch.save(SpNET.state_dict(), 'speedNet1.pth')

        print("Learning finished =", datetime.now().strftime("%H:%M:%S"))
        stop = 0
        # z_run = vrae.transform(test_dataset)
        # os.system("shutdown /s /t 1")

