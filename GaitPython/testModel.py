import torch

from SpeedNET2 import SpeedNET2
from vrae.utils import *
import torch
from datetime import datetime
import argparse


#  tensorboard --logdir /home/justjo/timeseries-clustering-vae/log/LogInception --host 0.0.0.0 --port 8090
# chzba 0.043 /media/data2/justjo/datasets/jednokolka/ 2 16 /home/justjo/timeseries-clustering-vae/log/LogInception/
# --datasetBasePath /media/data2/justjo/datasets/jednokolka/ --logPath /home/justjo/timeseries-clustering-vae/log/LogInception/ --cuda 2 --sensor 0 1 2 --filters 14
# --datasetBasePath  G:/deepLearn/jednokolka/ --logPath G:/deepLearn/localLog/Sin/  --cuda 0 --sensor 0 1 2 --latent 200 --speedWeight 0.001 --learningRate 0.001 --mess 'MSEtrain2'
#--datasetBasePath /media/data2/justjo/datasets/jednokolka/ --logPath /home/justjo/timeseries-clustering-vae/log/Sin/ --logPath2 /home/justjo/timeseries-clustering-vae/log/LogSinSep/ --cuda 2 --sensor 0 1 2 --latent 200 --speedWeight 0.001 --learningRate 0.001 --mess 'MAEtrain'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False,description='Process some integers.')
    parser.add_argument('--datasetBasePath', help='Path1')
    parser.add_argument('--cuda', help='Array of integers', type=int)
    parser.add_argument('--sensors', help='Array of integers', nargs='+', type=int)
    parser.add_argument('--LogParamLoadPath', help='Array of integers')

    args = parser.parse_args()

    datasetBasePath = args.datasetBasePath
    cudaNum = args.cuda

    batch_size = 1

    sequence_length = 1024
    hidden_size_FC = 1
    number_of_features = 18

    listSens = []
    sensors = args.sensors #[int(d) for d in ]
    for s in sensors:
        listSens.extend(list(range(0 + s * 6, 6 + s * 6)))
    number_of_features = len(listSens)

    print("Program start time =", datetime.now().strftime("%H:%M:%S"))

    sub_num = get_number_of_subjects('separBig', base=datasetBasePath)

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

    hidden_layer_depth = 1



    SpNET = SpeedNET2(sequence_length=sequence_length,
                      hidden_size=hidden_size,
                      hidden_size_FC=hidden_size_FC,
                      number_of_features=number_of_features,
                      sin_depth=sin_depth,
                      latent_length=latent_length,
                      batch_size=batch_size,
                      hidden_layer_depth=hidden_layer_depth,
                      channels_conv=conv_channels,
                      learning_rate=0.0,
                      dropout_rate=0.0,
                      dropRateFirstLay=0.0,
                      block='LSTM',
                      kl_loss_multip=kl_loss_multip,
                      lossDFFmultip=lossSpeed, lossEncodMultip=1,
                      reducing=False, reducePerc=0.97,
                      cudaNum=cudaNum,
                      logerPath='',
                      mess='-test-')

    SpNET.to(torch.device("cpu"))
    #SpNET.load_my_state_dict(torch.load('C:\Users\josef\Desktop\Logs\Sin\-2021-08-24-16-34-\'MonteCarlo-2\'-subj-0/model.pth'))
    weig=torch.load(args.LogParamLoadPath + "best.pth",map_location='cuda:0')
    SpNET.load_state_dict(weig)

    SpNET.to(torch.device("cuda:" + str(cudaNum)))

    train_dataset, test_dataset, valid_dataset = open_data_pickle_one_wheele_separated_validation('separBig', 6,
                                                                                                  base=datasetBasePath)

    test_dataset_sub = TensorDataset(test_dataset.tensors[0][:, :, listSens],
                                     test_dataset.tensors[1], test_dataset.tensors[2], test_dataset.tensors[3])
    train_dataset_sub = TensorDataset(train_dataset.tensors[0][:, :, listSens],
                                      train_dataset.tensors[1], train_dataset.tensors[2],
                                      train_dataset.tensors[3])
    valid_dataset_sub = TensorDataset(valid_dataset.tensors[0][:, :, listSens],
                                      valid_dataset.tensors[1], valid_dataset.tensors[2],
                                      valid_dataset.tensors[3])

    print("Learning started =", datetime.now().strftime("%H:%M:%S"))

    numpyArray = SpNET.getError(test_dataset_sub)

