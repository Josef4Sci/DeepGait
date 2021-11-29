import gc

import myUtils
from SpeedNET2 import SpeedNET2
from vrae.utils import *
import torch
from datetime import datetime
from SpeedNET import SpeedNET
import sys
import argparse
from myUtils import TensorboardLogger
import random
from pathlib import Path

#  tensorboard --logdir /home/justjo/timeseries-clustering-vae/log/LogInception --host 0.0.0.0 --port 8090
# chzba 0.043 /media/data2/justjo/datasets/jednokolka/ 2 16 /home/justjo/timeseries-clustering-vae/log/LogInception/
# --datasetBasePath /media/data2/justjo/datasets/jednokolka/ --logPath /home/justjo/timeseries-clustering-vae/log/LogInception/ --cuda 2 --sensor 0 1 2 --filters 14
# --datasetBasePath  G:/deepLearn/jednokolka/ --logPath G:/deepLearn/localLog/Sin/  --cuda 0 --sensor 0 1 2 --latent 200 --speedWeight 0.001 --learningRate 0.001 --mess 'MSEtrain2'
#--datasetBasePath /media/data2/justjo/datasets/jednokolka/ --logPath /home/justjo/timeseries-clustering-vae/log/Sin/ --logPath2 /home/justjo/timeseries-clustering-vae/log/LogSinSep/ --cuda 2 --sensor 0 1 2 --latent 200 --speedWeight 0.001 --learningRate 0.001 --mess 'MAEtrain'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False, description='')
    parser.add_argument('--datasetBasePath', default="../Bipedal-Motion-Dataset/")
    parser.add_argument('--logPath', default="../Logs/SineIndividual/")
    parser.add_argument('--logPath2', default="../Logs/SineMonteCarlo/")
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--sensors', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--learningRate', type=float, default=0.001)
    parser.add_argument('--startfromsubj', type=int, default=0)
    parser.add_argument('--testnet', default=None)
    parser.add_argument('--LogParamLoadPath', default=None)
    parser.add_argument('--rep', type=int, default=1)

    args = parser.parse_args()

    datasetBasePath = args.datasetBasePath
    logPath = args.logPath
    cudaNum = args.cuda
    learning_rate = args.learningRate

    Path(args.logPath).mkdir(parents=True, exist_ok=True)
    Path(args.logPath2).mkdir(parents=True, exist_ok=True)

    if not args.rep:
        args.rep=1

    for x in range(int(args.rep)):

        hidden_size = random.choice([128, 256, 512])
        latent_length = random.choice([64, 128, 256])
        hidden_layer_depth = 1 #random.choice([1, 2])
        lossSpeed = random.choice([0.1, 0.01, 0.001, 0.0001])
        kl_loss_multip = random.choice([0.0001, 0.00001, 0.000001, 0.0000001])
        sin_depth = random.choice([10, 50, 100])
        conv_channels = random.choice([1, 2, 4, 8])

        batch_size = 32
        n_epochs = 120

        sequence_length = 1024
        hidden_size_FC = 1
        number_of_features = 18
        DATA_FOLDER = 'Processed'

        listSens = []
        sensors = args.sensors #[int(d) for d in ]
        for s in sensors:
            listSens.extend(list(range(0 + s * 6, 6 + s * 6)))
        number_of_features = len(listSens)

        args.mess = myUtils.getMonteCarloNextFolder(args.logPath2)

        if not args.testnet:
            print("Program start time =", datetime.now().strftime("%H:%M:%S"))

            sub_num = get_number_of_subjects(DATA_FOLDER, base=datasetBasePath)

            if args.LogParamLoadPath:
                pathLog = args.LogParamLoadPath + 'log.txt'
                # Using readline()
                file1 = open(pathLog, 'r')

                while True:
                    # Get next line from file
                    line = file1.readline()
                    if 'hidden_size ' in line:
                        hidden_size = int(line.split()[-1])
                    if 'latent_length ' in line:
                        latent_length = int(line.split()[-1])
                    if 'hidden_layer_depth ' in line:
                        hidden_layer_depth = int(line.split()[-1])
                    if 'lossSpeed ' in line:
                        lossSpeed = float(line.split()[-1])
                    if 'kl_loss_multip ' in line:
                        kl_loss_multip = float(line.split()[-1])
                    if 'conv_channels ' in line:
                        conv_channels = int(line.split()[-1])
                    if 'sin_depth ' in line:
                        sin_depth = int(line.split()[-1])
                    if not line:
                        break

            logger = TensorboardLogger(args.logPath2, mess=args.mess)
            logger.log_txt("lossSpeed ", lossSpeed)
            logger.log_txt("hidden_layer_depth ", hidden_layer_depth)
            logger.log_txt("latent_length ", latent_length)
            logger.log_txt("hidden_size ", hidden_size)
            logger.log_txt("kl_loss_multip ", kl_loss_multip)
            logger.log_txt("sin_depth ", sin_depth)
            logger.log_txt("conv_channels ", conv_channels)

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

                if 'SpNET' in globals():
                    del SpNET

                gc.collect()

                train_dataset, test_dataset, valid_dataset = open_data_pickle_one_wheele_separated_validation(DATA_FOLDER, i,
                                                                                                              base=datasetBasePath)

                test_dataset_sub = TensorDataset(test_dataset.tensors[0][:, :, listSens],
                                                 test_dataset.tensors[1], test_dataset.tensors[2], test_dataset.tensors[3])
                train_dataset_sub = TensorDataset(train_dataset.tensors[0][:, :, listSens],
                                                  train_dataset.tensors[1], train_dataset.tensors[2], train_dataset.tensors[3])
                valid_dataset_sub = TensorDataset(valid_dataset.tensors[0][:, :, listSens],
                                                  valid_dataset.tensors[1], valid_dataset.tensors[2], valid_dataset.tensors[3])

                SpNET = SpeedNET2(sequence_length=sequence_length,
                                 hidden_size=hidden_size,
                                 hidden_size_FC=hidden_size_FC,
                                 number_of_features=number_of_features,
                                 sin_depth=sin_depth,
                                 latent_length=latent_length,
                                 batch_size=batch_size,
                                 hidden_layer_depth=hidden_layer_depth,
                                 channels_conv=conv_channels,
                                 learning_rate=learning_rate,
                                 dropout_rate=0.0,
                                 dropRateFirstLay=0.0,
                                 block='LSTM',
                                 kl_loss_multip=kl_loss_multip,
                                 lossDFFmultip=lossSpeed, lossEncodMultip=1,
                                 reducing=False, reducePerc=0.97,
                                 cudaNum=cudaNum,
                                 logerPath=logPath,
                                 mess=args.mess+'-subj-'+str(i))

                print("Learning started =", datetime.now().strftime("%H:%M:%S"))
                print("Subject =", i)

                minTestErr, meanLast = SpNET.fit(n_epochs, train_dataset_sub, test_dataset_sub,
                                                 valid_dataset=valid_dataset_sub, shuffleTrain=True)

                logger.epoch = i
                logger.step(1)
                logger.log_value('subject', i)
                logger.log_value('minTestErr', minTestErr)
                logger.log_value('meanLastTenTestErr', meanLast)


            print("Learning finished =", datetime.now().strftime("%H:%M:%S"))
            stop = 0
        else:
            train_dataset, test_dataset, valid_dataset = get_testing_dataset(100)

            test_dataset_sub = TensorDataset(test_dataset.tensors[0][:, :, listSens],
                                             test_dataset.tensors[1], test_dataset.tensors[2], test_dataset.tensors[3])
            train_dataset_sub = TensorDataset(train_dataset.tensors[0][:, :, listSens],
                                              train_dataset.tensors[1], train_dataset.tensors[2],
                                              train_dataset.tensors[3])
            valid_dataset_sub = TensorDataset(valid_dataset.tensors[0][:, :, listSens],
                                              valid_dataset.tensors[1], valid_dataset.tensors[2],
                                              valid_dataset.tensors[3])

            model = SpeedNET(sequence_length, hidden_size, hidden_size_FC, number_of_features, sin_depth,
                                 latent_length=latent_length,
                                 batch_size=batch_size,
                                 hidden_layer_depth=hidden_layer_depth,
                                 learning_rate=learning_rate,
                                 dropout_rate=0.0,
                                 dropRateFirstLay=0.0,
                                 block='LSTM',
                                 kl_loss_multip=kl_loss_multip,
                                 lossDFFmultip=lossSpeed, lossEncodMultip=1,
                                 reducing=False, reducePerc=0.97,
                                 cudaNum=cudaNum,
                                 logerPath=logPath,
                                 mess='test-sub')

            print("Learning started =", datetime.now().strftime("%H:%M:%S"))

            minTestErr, meanLast = model.fit(n_epochs, train_dataset_sub, test_dataset_sub,
                                             valid_dataset=valid_dataset_sub,
                                             shuffleTrain=True)

