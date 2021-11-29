import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from Utils.myUtils import TensorboardLogger

from torch.autograd import Variable
from tqdm import tqdm

import os
from os import path

from Models.inception import InceptionBlock


class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)


class Adap(nn.Module):
    def __init__(self):
        super(Adap, self).__init__()
        self.ad = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        x = self.ad(x)
        return x


class Conv1(nn.Module):
    def __init__(self, inputchan, maxpoolwind):
        super(Conv1, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=inputchan,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False)
        self.conv2 = nn.Conv1d(
            in_channels=inputchan,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False)

        self.batch_norm = nn.BatchNorm1d(num_features=1)
        self.activ = nn.ReLU()
        self.maxpool=nn.AvgPool1d(kernel_size=maxpoolwind+1, padding=maxpoolwind//2, stride=maxpoolwind, count_include_pad=False)#

        self.ad = nn.AdaptiveAvgPool1d(output_size=1)
    def forward(self, x):
        change = self.maxpool(x)
        x = self.ad(x).expand(x.shape[0], x.shape[1], x.shape[2])
        x = self.activ(self.conv(x))

        change = self.activ(self.conv2(change))
        c=torch.cat((change[:, :, 0].unsqueeze(2).expand(x.shape[0], x.shape[1], x.shape[2] // 4),
                   change[:, :, 1].unsqueeze(2).expand(x.shape[0], x.shape[1], x.shape[2] // 4),
                   change[:, :, 2].unsqueeze(2).expand(x.shape[0], x.shape[1], x.shape[2] // 4),
                   change[:, :, 3].unsqueeze(2).expand(x.shape[0], x.shape[1], x.shape[2] // 4)), dim=2)


        x = x + c
        return x

class Average(nn.Module):
    def __init__(self, maxpoolwind):
        super(Average, self).__init__()

        self.maxpool=nn.AvgPool1d(kernel_size=maxpoolwind+1, padding=maxpoolwind//2, stride=1, count_include_pad=False)

    def forward(self, x):
        x = self.maxpool(x)
        return x

class InceptionSpeedNET(nn.Module):
    """
    Deep Feed Forward Network
    """

    def __init__(self, filters=32, batch_size=32,
                 learning_rate=0.005, bottleneckLayers=32, kernel_sizes=[9, 45, 95], cudaNum=0, num_features=18, speed_weight=1,
                 logerPath=None,
                 mess=None):

        super(InceptionSpeedNET, self).__init__()

        if logerPath:
            self.logger = TensorboardLogger(logerPath, mess)
            self.logger.log_txt("Bottleneck layers", bottleneckLayers)
            self.logger.log_txt("Filters", filters)
            self.logger.log_txt("kernel size [9, 45, 95],", 0)
            self.logger.log_txt("attention_weight", speed_weight)
            self.logger.log_txt("learning_rate", learning_rate)
        if mess:
            self.logger.log_txt("Mess", mess)

        self.device = torch.device("cuda:" + str(cudaNum))
        self.dtype = torch.cuda.FloatTensor
        self.speed_weight = speed_weight

        self.InceptionTime1 = InceptionBlock(
                in_channels=num_features,
                n_filters=filters, # ladit 2^n [2,4,8,16,32]
                kernel_sizes=kernel_sizes, # ladit (5/11/21 11/21/41 21/41/81)
                bottleneck_channels=bottleneckLayers,  # ladit 2 4 8
                use_residual=True,
                activation=nn.ReLU()
            )

        self.InceptionTime2 = InceptionBlock(
                in_channels=filters * 4,
                n_filters=filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneckLayers,
                use_residual=True,
                activation=nn.ReLU()
            )

        self.out = nn.Sequential(
            Adap(),
            Flatten(out_features=filters * 4 * 1),
            nn.Linear(in_features=4 * filters * 1, out_features=1)
        )

        # self.conv = Conv1(filters * 4, 256)
        #self.avrg = Average(256)

        self.batch_size = batch_size

        # self.ensamble1_weight = nn.Linear(1, 1).type(self.dtype).to(self.device)
        # self.ensamble1_height = nn.Linear(1, 1).type(self.dtype).to(self.device)
        # self.ensamble2 = nn.Linear(2, 1).type(self.dtype).to(self.device)

        self.learning_rate = learning_rate

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                if sum([np.prod(p.size()) for p in own_state[name]]) != sum([np.prod(p.size()) for p in param]):
                    continue
                own_state[name].copy_(param)
            except:
                print('exetion occured')

    def forward(self, x, weight, height):

        x = self.InceptionTime1(x) #, decod1
        x = self.InceptionTime2(x) #, decod2

        x = self.out(x)
        # x = self.conv(x)
        # x = self.avrg(x)

        # weight = F.leaky_relu(self.ensamble1_weight(weight))
        # height = F.leaky_relu(self.ensamble1_height(height))
        # features = torch.cat((weight, height), dim=1)
        # attention = torch.sigmoid(self.ensamble2(features)) * 2

        # x = F.dropout(latent, p=self.dropRateFirstLay, training=self.training)
        #
        # x = F.leaky_relu(self.fc1(x))

        # x = self.fc4(x)
        #
        return x, 0#, decod1

    def fit(self, num_epochs, train_dataset, test_dataset, valid_dataset=None, shuffleTrain=True):

        trainable_params = list(filter(
            lambda p: p.requires_grad, self.parameters()))
        # trainable_params = list(self.fc1.parameters())+list(self.fc2.parameters())+list(self.fc3.parameters())\
        #                       +list(self.fc3_2.parameters())+list(self.fc4.parameters())+list(self.ensamble2.parameters())\
        #                       +list(self.ensamble1_weight.parameters())+list(self.ensamble1_height.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=shuffleTrain,
                                       drop_last=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if valid_dataset:
            valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False,
                                           drop_last=True)


        self.to(self.device)

        MAE = nn.L1Loss().to(self.device)
        MSE = nn.MSELoss().to(self.device)

        minimErrorMem = 1000000000.0

        lastErrors = []

        validErrors = []
        lastErrorsValidStop = []

        for epoch in range(num_epochs):

            print('Epoch:', epoch)

            self.train()

            steps = 0
            loss_x = 0

            torch.cuda.empty_cache()

            # training
            for (samples, labels, weight, height) in tqdm(train_data_loader):
                steps += 1
                loss_speed = self.filter_step(samples, labels, weight, height, self.device,
                                                                  optimizer, MSE)
                loss_x += loss_speed.item()

            print(loss_x / steps)

            self.logger.epoch = epoch
            self.logger.step(1)
            self.logger.log_value('loss', loss_x / steps)

            # test train data without regularization
            if valid_dataset:
                self.eval()
                steps = 0
                loss_x = 0

                for i, (samples, labels, weight, height) in enumerate(valid_data_loader):
                    with torch.no_grad():
                        loss_speed = self.filter_step(samples, labels, weight, height,
                                                                          self.device, optimizer, MAE, train=False)
                        loss_x += loss_speed.item()
                        steps += 1

                print("Valid loss:", loss_x / steps)
                self.logger.log_value('{} MAE'.format("valid"), loss_x / steps)

                validErrors.append(loss_x / steps)
                if min(validErrors[-20:]) != min(validErrors):
                    lastErrors = lastErrorsValidStop[-20:]
                    print('Finished at epoch, because of validation:', epoch)
                    break

            # test data1
            if test_dataset:
                steps = 0
                lossMAE_x = 0

                for i, (samples, labels, weight, height) in enumerate(test_data_loader):
                    with torch.no_grad():
                        loss_speed = self.filter_step(samples, labels, weight, height,
                                                                          self.device, optimizer, MAE, train=False)
                        lossMAE_x += loss_speed.item()
                        steps += 1

                test_loss_MAE = lossMAE_x / steps

                if test_loss_MAE < minimErrorMem:
                    minimErrorMem = test_loss_MAE
                    savePath = self.logger.log_dir + '/best.pth'
                    if path.exists(savePath):
                        os.remove(savePath)
                    torch.save(self.state_dict(), savePath)

                if (num_epochs-epoch) < 20:
                    lastErrors.append(test_loss_MAE)

                lastErrorsValidStop.append(test_loss_MAE)

                self.logger.log_value('{} MAE'.format("test"), test_loss_MAE)
                # print("Test loss:", loss_x / steps)
                print("Test loss MAE:", test_loss_MAE)

        lastAverage = sum(lastErrors) / len(lastErrors)
        return minimErrorMem, lastAverage

    def filter_step(self, samples, labels, trainW, trainH, device, optimizer, loss1, train=True):

        samples = samples.transpose(1, 2)
        labelsV = Variable(labels.to(device), requires_grad=True)
        weightV = Variable(trainW.to(device), requires_grad=True)
        heightV = Variable(trainH.to(device), requires_grad=True)
        samplesV = Variable(samples.to(device), requires_grad=True)

        if train:
            optimizer.zero_grad()

        y, attention = self(samplesV, weightV, heightV)#, decod

        loss_speed = loss1(y, labelsV)
        #loss_speed = loss1(labelsV[:, 512], y.squeeze())

        # loss_atten = loss1(attention, torch.ones(self.batch_size,1).to(device))

        loss = loss_speed  # + self.speed_weight * loss_atten

        if train:
            loss.backward()
            optimizer.step()

        return loss_speed#, loss_autoencod
