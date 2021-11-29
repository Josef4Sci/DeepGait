import gc

from vrae.utils import *
import torch
from datetime import datetime
from perceiver_pytorch import Perceiver
import sys
from myUtils import TensorboardLogger
import argparse
import numpy
import random

# tensorboard --logdir=G:\deepLearn\localLog\compare --host localhost --port 8088
# chzba 0.043 /media/data2/justjo/datasets/jednokolka/ 2 16 /home/justjo/timeseries-clustering-vae/log/LogInception/

# --datasetBasePath  G:/deepLearn/jednokolka/ --logPath G:\deepLearn\localLog\perceiver\ --logPath2 G:\deepLearn\localLog\perceiverAll\ --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L' --startfromsubj 0
#        --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L'

#--datasetBasePath /media/data2/justjo/datasets/jednokolka/ --logPath /home/justjo/timeseries-clustering-vae/log/Perceiver/ --logPath2 /home/justjo/timeseries-clustering-vae/log/PerceiverAll/
#       --cuda 2 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L' --startfromsubj 0

#--datasetBasePath C:/Users/josef/PycharmProjects/jednokolka/ --logPath2 C:/Users/josef/PycharmProjects/Log/perceiverAll/
# --logPath C:/Users/josef/PycharmProjects/Log/perceiver/ --cuda 0 --sensor 0 --learningRate 0.001 --tag MonteCarlo --mess 'MonteCarlo-startFrom1' --startfromsubj 1 --LogParamLoadPath C:\Users\josef\PycharmProjects\Log\perceiverAll\-2021-08-01-01-45-'MonteCarlo'\


if __name__ == "__main__":
#--datasetBasePath  G:/deepLearn/jednokolka/ --logPath G:\deepLearn\localLog\perceiver --logPath2 G:\deepLearn\localLog\perceiverAll --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L'
    parser = argparse.ArgumentParser(add_help=False, description='Process some integers.')
    parser.add_argument('--datasetBasePath', help='Path1')
    parser.add_argument('--logPath', help='Array of integers')
    parser.add_argument('--logPath2', help='Array of integers')
    parser.add_argument('--cuda', help='Array of integers', type=int)
    # parser.add_argument('--latent', help='Array of integers', type=int)
    parser.add_argument('--sensors', help='Array of integers', nargs='+', type=int)
    parser.add_argument('--learningRate', help='Array of integers', type=float)
    #parser.add_argument('--tag', help='Array of integers')
    #parser.add_argument('--mess', help='Array of integers')
    parser.add_argument('--startfromsubj',  default=0, help='Array of integers', type=int)
    parser.add_argument('--LogParamLoadPath', help='Array of integers')

    args = parser.parse_args()

    datasetBasePath = args.datasetBasePath
    logPath = args.logPath
    cudaNum = args.cuda
    learning_rate = args.learningRate

    batch_size = 32
    n_epochs = 120

    num_freq_bands = random.choice([6, 12])
    max_freq = random.choice([3., 5., 10., 15.])
    depth = random.choice([6, 12])
    num_latents = random.choice([128, 256, 512])
    latent_dim = random.choice([64, 128, 256])
    cross_dim = random.choice([512, 256, 128])
    cross_dim_head = random.choice([64, 32, 16])
    latent_dim_head = random.choice([64, 32, 16])

    if args.LogParamLoadPath:
        pathLog = args.LogParamLoadPath + 'log.txt'
        # Using readline()
        file1 = open(pathLog, 'r')

        while True:
            # Get next line from file
            line = file1.readline()
            if 'num_freq_bands ' in line:
                num_freq_bands = int(line.split()[-1])
            if 'max_freq ' in line:
                max_freq = float(line.split()[-1])
            if 'depth ' in line:
                depth = int(line.split()[-1])
            if 'num_latents ' in line:
                num_latents = int(line.split()[-1])
            if 'latent_dim ' in line:
                latent_dim = int(line.split()[-1])
            if 'cross_dim ' in line:
                cross_dim = int(line.split()[-1])
            if 'cross_dim_head ' in line:
                cross_dim_head = int(line.split()[-1])
            if 'latent_dim_head ' in line:
                latent_dim_head = int(line.split()[-1])

            if not line:
                break

    startIndex = 0
    for x in [f.path for f in os.scandir(args.logPath2) if f.is_dir()]:
        if ('MonteCarlo' in x):
            currentIndex=int(x.split('\'')[1].split('MonteCarlo-')[1].split('-')[0])
            if currentIndex > startIndex:
                startIndex = currentIndex

    mess1 = '\'MonteCarlo-' + str(startIndex + 1) + '-s' + str(args.startfromsubj) + '\''


    listSens = []
    sensors = [int(d) for d in args.sensors]
    for s in sensors:
        listSens.extend(list(range(0 + s * 6, 6 + s * 6)))
    number_of_features = len(listSens)

    print("Program start time =", datetime.now().strftime("%H:%M:%S"))

    sub_num=get_number_of_subjects('separBig', base=datasetBasePath)

    logger = TensorboardLogger(args.logPath2, mess=mess1)
    logger.log_txt("num_freq_bands ", num_freq_bands)
    logger.log_txt("max_freq ", max_freq)
    logger.log_txt("depth ", depth)
    logger.log_txt("num_latents ", num_latents)
    logger.log_txt("latent_dim ", latent_dim)
    logger.log_txt("cross_dim ", cross_dim)
    logger.log_txt("cross_dim_head ", cross_dim_head)
    logger.log_txt("latent_dim_head ", latent_dim_head)

    for i in range(args.startfromsubj, sub_num):

        mess = '\'MonteCarlo-' + str(startIndex + 1) + '\'' + 'subj-' + str(i) + '-started_sub-' +\
               str(args.startfromsubj)

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

        train_dataset, test_dataset, valid_dataset = open_data_pickle_one_wheele_separated_validation('separBig', i, base=datasetBasePath)

        test_dataset_sub = TensorDataset(test_dataset.tensors[0][:, :, listSens],
                                       test_dataset.tensors[1], test_dataset.tensors[2], test_dataset.tensors[3])
        train_dataset_sub = TensorDataset(train_dataset.tensors[0][:, :, listSens],
                                          train_dataset.tensors[1], train_dataset.tensors[2], train_dataset.tensors[3])
        valid_dataset_sub = TensorDataset(valid_dataset.tensors[0][:, :, listSens],
                                          valid_dataset.tensors[1], valid_dataset.tensors[2], valid_dataset.tensors[3])

        model = Perceiver(
            input_channels=number_of_features,  # number of channels for each token of the input
            input_axis=1,  # number of axis for input data (2 for images, 3 for video)
            num_freq_bands=num_freq_bands,  # number of freq bands, with original value (2 * K + 1) -ladit
            max_freq=max_freq,  # maximum frequency, hyperparameter depending on how fine the data is  -ladit
            depth=depth,  # depth of net  -ladit jen 12
            num_latents=num_latents,
            # number of latents, or induced set points, or centroids. different papers giving it different names -ladit 128 a 512
            cross_dim=cross_dim,  # cross attention dimension -ladit halfs
            latent_dim=latent_dim,  # latent dimension 128
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=cross_dim_head, #-ladit halfs
            latent_dim_head=latent_dim_head, #-ladit halfs
            num_classes=1,  # output number of classes
            attn_dropout=0.2,
            ff_dropout=0.3,
            weight_tie_layers=False, # whether to weight tie layers (optional, as indicated in the diagram)
            cudaNum=cudaNum,
            logerPath=logPath,
            learning_rate=learning_rate,
            batch=batch_size,
            mess=mess
        )

        print("Learning started =", datetime.now().strftime("%H:%M:%S"))
        print("Subject =", i)

        minTestErr, meanLast = model.fit(n_epochs, train_dataset_sub, test_dataset_sub, valid_dataset=valid_dataset_sub, shuffleTrain=True)

        logger.epoch = i
        logger.step(1)
        logger.log_value('subject', i)
        logger.log_value('minTestErr', minTestErr)
        logger.log_value('meanLastTenTestErr', meanLast)

    # torch.save(SpNET.state_dict(), 'speedNet1.pth')

    print("Learning finished =", datetime.now().strftime("%H:%M:%S"))
    stop = 0
    # z_run = vrae.transform(test_dataset)
    #os.system("shutdown /s /t 1")