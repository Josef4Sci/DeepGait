import gc

import myUtils
from vrae.utils import *
from datetime import datetime
from SpeedNetConvLSTM import SpeedNetConvLSTM
from myUtils import TensorboardLogger
import argparse
import random

# tensorboard --logdir=G:\deepLearn\localLog\compare --host localhost --port 8088
# chzba 0.043 /media/data2/justjo/datasets/jednokolka/ 2 16 /home/justjo/timeseries-clustering-vae/log/LogInception/

# --datasetBasePath  G:/deepLearn/jednokolka/ --logPath G:\deepLearn\localLog\perceiver\ --logPath2 G:\deepLearn\localLog\perceiverAll\ --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L' --startfromsubj 0
#        --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L'

#--datasetBasePath /media/data2/justjo/datasets/jednokolka/ --logPath /home/justjo/timeseries-clustering-vae/log/Perceiver/ --logPath2 /home/justjo/timeseries-clustering-vae/log/PerceiverAll/
#       --cuda 2 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L' --startfromsubj 0
if __name__ == "__main__":
#--datasetBasePath  G:/deepLearn/jednokolka/ --logPath G:\deepLearn\localLog\perceiver --logPath2 G:\deepLearn\localLog\perceiverAll --cuda 0 --sensor 0 --latent 64  --learningRate 0.001 --tag MAE-S0-latent128 --mess 'Perc-64L'
    parser = argparse.ArgumentParser(add_help=False, description='Process some integers.')
    parser.add_argument('--datasetBasePath', help='Path1')
    parser.add_argument('--logPath', help='Array of integers')
    parser.add_argument('--logPath2', help='Array of integers')
    parser.add_argument('--cuda', help='Array of integers', type=int)
    parser.add_argument('--sensors', help='Array of integers', nargs='+', type=int)
    parser.add_argument('--learningRate', help='Array of integers', type=float)
    parser.add_argument('--startfromsubj',  default=0, help='Array of integers', type=int)
    parser.add_argument('--LogParamLoadPath', help='Array of integers')
    parser.add_argument('--testnet', help='Array of integers', type=bool, default=False)
    parser.add_argument('--rep', help='Array of integers')

    args = parser.parse_args()

    datasetBasePath = args.datasetBasePath
    logPath = args.logPath
    cudaNum = args.cuda
    learning_rate = args.learningRate

    batch_size = 32
    n_epochs = 120

    sequence_length = 1024
    hidden_size_FC = 1

    listSens = []
    sensors = [int(d) for d in args.sensors]
    for s in sensors:
        listSens.extend(list(range(0 + s * 6, 6 + s * 6)))
    number_of_features = len(listSens)

    args.mess = myUtils.getMonteCarloNextFolder(args.logPath2)

    if not args.rep:
        args.rep = 1

    for x in range(int(args.rep)):

        hidden_size = random.choice([128, 256, 512])
        latent_length = random.choice([64, 128, 256])  #
        hidden_layer_depth = 1  # random.choice([1, 2])
        lossSpeed = random.choice([0.1, 0.01, 0.001, 0.0001])
        conv_channels = random.choice([8, 16])
        kl_loss_multip = random.choice([0.0001, 0.00001, 0.000001, 0.0000001])
        weight_decay = random.choice([0.1, 0.01, 0.001, 0.0])

        if not args.testnet:
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
                    if 'weight_decay ' in line:
                        weight_decay = int(line.split()[-1])
                    if 'latent_length ' in line:
                        latent_length = int(line.split()[-1])
                    if 'hidden_layer_depth ' in line:
                        hidden_layer_depth = int(line.split()[-1])
                    if 'lossSpeed ' in line:
                        lossSpeed = float(line.split()[-1])
                    if 'conv_channels ' in line:
                        conv_channels = int(line.split()[-1])
                    if 'kl_loss_multip ' in line:
                        kl_loss_multip = float(line.split()[-1])
                    if not line:
                        break

            logger = TensorboardLogger(args.logPath2, mess=args.mess)
            logger.log_txt("lossSpeed ", lossSpeed)
            logger.log_txt("hidden_layer_depth ", hidden_layer_depth)
            logger.log_txt("latent_length ", latent_length)
            logger.log_txt("hidden_size ", hidden_size)
            logger.log_txt("conv_channels ", conv_channels)
            logger.log_txt("kl_loss_multip ", kl_loss_multip)
            logger.log_txt("weight_decay ", kl_loss_multip)

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
                train_dataset, test_dataset, valid_dataset = open_data_pickle_one_wheele_separated_validation('separBig', i, base=datasetBasePath)

                test_dataset_sub=TensorDataset(test_dataset.tensors[0][:, :, listSens],
                                               test_dataset.tensors[1], test_dataset.tensors[2], test_dataset.tensors[3])
                train_dataset_sub = TensorDataset(train_dataset.tensors[0][:, :, listSens],
                                                  train_dataset.tensors[1], train_dataset.tensors[2], train_dataset.tensors[3])
                valid_dataset_sub = TensorDataset(valid_dataset.tensors[0][:, :, listSens],
                                                  valid_dataset.tensors[1], valid_dataset.tensors[2], valid_dataset.tensors[3])

                model = SpeedNetConvLSTM(
                    sequence_length, hidden_size, conv_channels, number_of_features,
                    latent_length=latent_length,
                    batch_size=batch_size,
                    hidden_layer_depth=hidden_layer_depth,
                    learning_rate=learning_rate,
                    kl_loss_multip=kl_loss_multip,
                    lossEncodMultip=lossSpeed,
                    cudaNum=cudaNum,
                    logerPath=logPath,
                    mess=args.mess+'-subj-'+str(i)
                )
                print("Learning started =", datetime.now().strftime("%H:%M:%S"))

                minTestErr, meanLast = model.fit(n_epochs, train_dataset_sub, test_dataset_sub, valid_dataset=valid_dataset_sub, shuffleTrain=True)

                logger.epoch = i
                logger.step(1)
                logger.log_value('minTestErr', minTestErr)
                logger.log_value('meanLastTenTestErr', meanLast)

            # torch.save(SpNET.state_dict(), 'speedNet1.pth')

            print("Learning finished =", datetime.now().strftime("%H:%M:%S"))
            stop = 0
            # z_run = vrae.transform(test_dataset)
            #os.system("shutdown /s /t 1")
        else:
            train_dataset, test_dataset, valid_dataset = get_testing_dataset(100)

            test_dataset_sub = TensorDataset(test_dataset.tensors[0][:, :, listSens],
                                             test_dataset.tensors[1], test_dataset.tensors[2], test_dataset.tensors[3])
            train_dataset_sub = TensorDataset(train_dataset.tensors[0][:, :, listSens],
                                              train_dataset.tensors[1], train_dataset.tensors[2], train_dataset.tensors[3])
            valid_dataset_sub = TensorDataset(valid_dataset.tensors[0][:, :, listSens],
                                              valid_dataset.tensors[1], valid_dataset.tensors[2], valid_dataset.tensors[3])

            model = SpeedNetConvLSTM(
                sequence_length, hidden_size, 4, number_of_features,
                latent_length=latent_length,
                batch_size=batch_size,
                hidden_layer_depth=hidden_layer_depth,
                learning_rate=learning_rate,
                kl_loss_multip=0.000001,
                lossEncodMultip=lossSpeed,
                cudaNum=cudaNum,
                logerPath=logPath,
                mess='-subj-test'
            )
            print("Learning started =", datetime.now().strftime("%H:%M:%S"))

            minTestErr, meanLast = model.fit(n_epochs, train_dataset_sub, test_dataset_sub, valid_dataset=valid_dataset_sub,
                                             shuffleTrain=True)
