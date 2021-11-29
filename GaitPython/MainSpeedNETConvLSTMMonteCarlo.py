import gc

from Utils import myUtils
from Utils.utils import *
from datetime import datetime
from Models.SpeedNetConvLSTM import SpeedNetConvLSTM
from Utils.myUtils import TensorboardLogger
import Utils.Constants
import argparse
import random


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

    for x in range(int(args.rep)):

        hidden_size = random.choice([128, 256, 512])
        latent_length = random.choice([64, 128, 256])
        hidden_layer_depth = 1  # random.choice([1, 2])
        lossSpeed = random.choice([0.1, 0.01, 0.001, 0.0001])
        conv_channels = random.choice([8, 16])
        kl_loss_multip = random.choice([0.0001, 0.00001, 0.000001, 0.0000001])
        weight_decay = random.choice([0.1, 0.01, 0.001, 0.0])

        if not args.testnet:
            print("Program start time =", datetime.now().strftime("%H:%M:%S"))

            sub_num = get_number_of_subjects(Utils.Constants.DATA_FOLDER, base=datasetBasePath)

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
                train_dataset, test_dataset, valid_dataset = open_data_pickle_one_wheele_separated_validation(Utils.Constants.DATA_FOLDER, i, base=datasetBasePath)

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
