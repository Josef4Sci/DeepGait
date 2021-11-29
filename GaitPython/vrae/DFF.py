import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from myUtils import TensorboardLogger

from torch.autograd import Variable
from tqdm import tqdm

import os
from os import path

class DFF(nn.Module):
    """
    Deep Feed Forward Network
    """
    def __init__(self, vrae, number_of_features, hidden_size, batch_size, lossDFFmultip=1, lossEncodMultip=1,
                 dropRateFirstLay=0.0, dropRate=0.0):

        super(DFF, self).__init__()

        self.logger = TensorboardLogger("C:/Users/josef/Desktop/DeepLearning/Logs/Log7/")

        self.vrae = vrae

        self.batch_size = batch_size
        self.number_of_features = number_of_features
        self.lossDFFmultip = lossDFFmultip
        self.lossEncodMultip = lossEncodMultip

        self.lossChange = 0

        self.hidden_size = hidden_size
        self.dropRate = dropRate
        self.dropRateFirstLay = dropRateFirstLay

        self.fc1 = nn.Linear(self.number_of_features, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 1)

        self.ensamble1_weight = nn.Linear(1, 16)
        self.ensamble1_height = nn.Linear(1, 16)
        self.ensamble2 = nn.Linear(32, self.hidden_size)



    def forward(self, x, weight, height):

        weight = F.leaky_relu(self.ensamble1_weight(weight))
        height = F.leaky_relu(self.ensamble1_height(height))
        features = torch.cat((weight, height), dim=1)
        attention = torch.sigmoid(self.ensamble2(features)) * 2

        x = F.dropout(x, p=self.dropRateFirstLay, training=self.training)

        x = self.fc1(x)
        # x = F.dropout(x, p=self.dropRate, training=self.training)

        # x = F.leaky_relu(self.fc2(x))
        # x = F.dropout(x, p=self.dropRate, training=self.training)

        x = self.fc3(x)
        # x = F.dropout(x, p=self.dropRate, training=self.training)

        x = self.fc4(x * attention)

        return x

    def fit(self, num_epochs, train_dataset, test_dataset,test_dataset2 = 0):

        self.train()
        params = list(self.parameters()) + list(self.vrae.parameters())
        optimizer = torch.optim.Adam(params, lr=self.vrae.learning_rate, weight_decay=0.00002)

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if test_dataset2 != 0:
            test_data_loader2 = DataLoader(dataset=test_dataset2, batch_size=self.batch_size, shuffle=False, drop_last=True)

        device = torch.device("cuda:0")

        self.to(device)

        MAE = nn.L1Loss().to(device)
        MSE = nn.MSELoss().to(device)

        minimErrorMem = 1000000000.0

        for epoch in range(num_epochs):
            print('Epoch:', epoch)
            self.logger.epoch = epoch

            if(epoch<120):
                self.vrae.setNonZeroVect(200-epoch)

            self.train()
            #scheduler.step()
            # for i, (samples, labels) in enumerate(train_data_loader):
            steps = 0
            loss_x = 0

            self.logger.step(1)
            # training
            for (samples, labels, weight, height) in tqdm(train_data_loader):
                loss, loss_dummy, loss_vrae = self.filter_step(samples, labels, weight, height, device, optimizer, MSE)
                loss_x += loss.item()
                steps += 1

            self.logger.log_value('loss', loss_x / steps)

            # test train data without regularization
            self.eval()
            steps = 0
            loss_x = 0
            lossMAE_x = 0
            loss_vrae_x = 0

            for i, (samples, labels, weight, height) in enumerate(train_data_loader):
                with torch.no_grad():
                    loss_MSE, loss_MAE, loss_vrae = self.filter_step(samples, labels, weight, height, device, optimizer, MSE,
                                                          loss2=MAE, train=False)
                    loss_x += loss_MSE.item()
                    lossMAE_x += loss_MAE.item()
                    loss_vrae_x += loss_vrae.item()
                    steps += 1

            # print("Training loss:", loss_x / steps)
            print("Training loss MAE:", lossMAE_x / steps)
            self.logger.log_value('{} MSE'.format("train"), loss_x / steps)
            self.logger.log_value('{} MAE'.format("train"), lossMAE_x / steps)
            self.logger.log_value('{} VRAE'.format("train"), loss_vrae_x / steps)

            # test data1
            steps = 0
            loss_x = 0
            lossMAE_x = 0
            loss_vrae_x = 0

            for i, (samples, labels, weight, height) in enumerate(test_data_loader):
                with torch.no_grad():
                    loss_MSE, loss_MAE, loss_vrae = self.filter_step(samples, labels, weight, height, device, optimizer, MSE,
                                                          loss2=MAE, train=False)
                    loss_x += loss_MSE.item()
                    lossMAE_x += loss_MAE.item()
                    loss_vrae_x += loss_vrae.item()
                    steps += 1

            test_loss_MAE = lossMAE_x / steps
            if test_loss_MAE < minimErrorMem:
                minimErrorMem = test_loss_MAE
                savePath = self.logger.log_dir + '/best.pth'
                if path.exists(savePath):
                    os.remove(savePath)
                torch.save(self.state_dict(), savePath)

            self.logger.log_value('{} MSE'.format("test"), loss_x / steps)
            self.logger.log_value('{} VRAE'.format("test"), loss_vrae_x / steps)
            self.logger.log_value('{} MAE'.format("test"), test_loss_MAE)
            # print("Test loss:", loss_x / steps)
            print("Test loss MAE:", test_loss_MAE)

            # test data2
            if test_dataset2 != 0:
                steps = 0
                loss_x = 0
                loss_vrae_x = 0

                for i, (samples, labels, weight, height) in enumerate(test_data_loader2):
                    with torch.no_grad():
                        loss_MSE, loss_MAE, loss_vrae = self.filter_step(samples, labels, weight, height, device, optimizer, MSE,
                                                              loss2=MAE, train=False)
                        loss_x += loss_MSE.item()
                        lossMAE_x += loss_MAE.item()
                        loss_vrae_x += loss_vrae.item()
                        steps += 1

                self.logger.log_value('{} MSE'.format("test2"), loss_x / steps)
                self.logger.log_value('{} MAE'.format("test2"), lossMAE_x / steps)
                self.logger.log_value('{} VRAE'.format("test2"), loss_vrae_x / steps)

                # print("Test loss2:", loss_x / steps)
                print("Test loss MAE2:", lossMAE_x / steps)

        konec=0

    def filter_step(self, samples, labels, trainW, trainH, device, optimizer, loss1, loss2=0, train=True):
        samplesV = Variable(samples.to(device))
        labelsV = Variable(labels.to(device))
        weightV = Variable(trainW.to(device))
        heightV = Variable(trainH.to(device))

        if train:
            optimizer.zero_grad()
        # Forward + Backward + Optimize
        inp = samplesV.transpose(0, 1)
        x = self.vrae.encoder(inp)
        x = self.vrae.lmbd(x)

        sum_loss, recon_loss, kl_loss, dum_x = self.vrae.compute_loss(inp)
        loss_vrae = torch.sum(sum_loss) / self.batch_size
        loss_recon = torch.sum(recon_loss) / self.batch_size

        predict_label = self(x, weightV, heightV)

        loss_speed = loss1(predict_label, labelsV)

        if train:
            DFF_multip, encod_decod_multip = self.lossDFFmultip, self.lossEncodMultip
        else:
            DFF_multip, encod_decod_multip = 1, 0

        loss1res = loss_speed * DFF_multip + loss_vrae * encod_decod_multip

        loss2res = 0
        if loss2 != 0:
            loss_speed = loss2(predict_label, labelsV)
            loss2res = loss_speed * DFF_multip + loss_vrae * encod_decod_multip

        if train:
            loss1res.backward()
            optimizer.step()

        return loss1res, loss2res, loss_recon

    def interpolateLatent(self, samples, labels, trainW, trainH, device):
        samplesV = Variable(samples.to(device))
        labelsV = Variable(labels.to(device))
        weightV = Variable(trainW.to(device))
        heightV = Variable(trainH.to(device))

        # Forward + Backward + Optimize
        inp = samplesV.transpose(0, 1)
        x = self.vrae.encoder(inp)
        x = self.vrae.lmbd(x)

        interpolateLatent = (torch.sum(x, dim=0) / 32) #.repeat(32, 1)
        interpolateLabel = (torch.sum(labelsV, dim=0) / 32) #.repeat(32, 1)
        interpolateWeight = (torch.sum(weightV, dim=0) / 32)
        interpolateHeight = (torch.sum(heightV, dim=0) / 32)

        return interpolateLatent, interpolateLabel, interpolateWeight, interpolateHeight

    def final_tune_speed(self, num_epochs, train_dataset, test_dataset,test_dataset2 = 0):
        self.train()

        for p in self.vrae.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(self.parameters(), lr=self.vrae.learning_rate, weight_decay=0.00002)

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if test_dataset2 != 0:
            test_data_loader2 = DataLoader(dataset=test_dataset2, batch_size=self.batch_size, shuffle=False,
                                           drop_last=True)

        device = torch.device("cuda:0")

        self.to(device)

        MAE = nn.L1Loss().to(device)
        MSE = nn.MSELoss().to(device)

        minimErrorMem = 1000000000.0

        for epoch in range(num_epochs):
            print('Epoch:', epoch)
            self.logger.epoch = epoch
            self.training=True
            self.train()
            steps = 0
            loss_x = 0

            self.logger.step(1)
            # training
            for (samples, label, weight, height) in tqdm(train_data_loader):
                loss, loss_dummy = self.filter_step_latent(samples, label, weight, height, device, optimizer, MSE)
                loss_x += loss.item()
                steps += 1

            self.logger.log_value('loss', loss_x / steps)

            # test data1
            self.eval()
            self.training = False
            steps = 0
            loss_x = 0
            lossMAE_x = 0

            for i, (samples, labels, weight, height) in enumerate(test_data_loader):
                with torch.no_grad():
                    loss_MSE, loss_MAE, vrae_dumm = self.filter_step(samples, labels, weight, height, device, optimizer, MSE,
                                                          loss2=MAE, train=False)
                    loss_x += loss_MSE.item()
                    lossMAE_x += loss_MAE.item()
                    steps += 1

            test_loss_MAE = lossMAE_x / steps
            if test_loss_MAE < minimErrorMem:
                minimErrorMem = test_loss_MAE
                savePath = self.logger.log_dir + '/best.pth'
                if path.exists(savePath):
                    os.remove(savePath)
                torch.save(self.state_dict(), savePath)

            self.logger.log_value('{} MSE'.format("test"), loss_x / steps)
            self.logger.log_value('{} MAE'.format("test"), test_loss_MAE)

            print("Test loss MAE:", test_loss_MAE)

            # test data2
            if test_dataset2 != 0:
                steps = 0
                loss_x = 0

                for i, (samples, labels, weight, height) in enumerate(test_data_loader2):
                    with torch.no_grad():
                        loss_MSE, loss_MAE, vrae_dumm = self.filter_step(samples, labels, weight, height, device, optimizer, MSE,
                                                              loss2=MAE, train=False)
                        loss_x += loss_MSE.item()
                        lossMAE_x += loss_MAE.item()
                        steps += 1

                self.logger.log_value('{} MSE'.format("test2"), loss_x / steps)
                self.logger.log_value('{} MAE'.format("test2"), lossMAE_x / steps)

                print("Test loss MAE2:", lossMAE_x / steps)

    def filter_step_latent(self, samples, labels, trainW, trainH, device, optimizer, loss1, loss2=0, train=True):
        samplesV = Variable(samples.to(device))
        labelsV = Variable(labels.to(device))
        weightV = Variable(trainW.to(device))
        heightV = Variable(trainH.to(device))

        if train:
            optimizer.zero_grad()
        # Forward + Backward + Optimize
        predict_label = self(samplesV, weightV, heightV)
        loss_speed = loss1(predict_label, labelsV)

        if train:
            DFF_multip = self.lossDFFmultip
        else:
            DFF_multip = 1

        loss1res = loss_speed * DFF_multip

        loss2res = 0
        if loss2 != 0:
            loss_speed = loss2(predict_label, labelsV)
            loss2res = loss_speed * DFF_multip

        if train:
            loss1res.backward()
            optimizer.step()

        return loss1res, loss2res


    def test_model(self, test_dataset):

        device = torch.device("cuda:0")
        self.to(device)
        self.eval()

        MAE = nn.L1Loss().to(device)
        MSE = nn.MSELoss().to(device)

        steps = 0
        loss_x = 0
        lossMAE_x = 0
        correct_test = 0
        lSpeedVals = range(1, 16, 1)
        lSpeeds = np.zeros([15, 1])

        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        for i, (samples, labels, weight, height) in enumerate(test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.to(device))
                labelsV = Variable(labels.to(device))
                weightV = Variable(weight.to(device))
                heightV = Variable(height.to(device))

                x = samplesV.transpose(0, 1)
                x = self.vrae.encoder(x)
                x = self.vrae.lmbd(x)
                predict_label = self(x, weightV, heightV)

                loss = MSE(predict_label, labelsV)
                loss_x += loss.item()

                lossMae = MAE(predict_label, labelsV)
                lossMAE_x += lossMae.item()

                m = min(lSpeedVals, key=lambda x: abs(x - predict_label[0]))
                if m in range(0, 15):
                    if lSpeeds[m] == 0:
                        lSpeeds[m] = lossMae.item()
                    else:
                        lSpeeds[m] = lSpeeds[m] * 0.98 + lossMae.item() * 0.02

                steps += 1

        test_loss = loss_x / steps
        test_loss_MAE = lossMAE_x / steps
        print("MAE",test_loss_MAE,test_loss)

        print(lSpeeds)

    def test_model2(self, test_dataset):

        device = torch.device("cuda:0")
        self.to(device)
        self.eval()

        MAE = nn.L1Loss().to(device)
        MSE = nn.MSELoss().to(device)

        lSpeeds = np.zeros([20, 1])

        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False,
                                      drop_last=True)

        for i, (samples, labels, weight, height) in enumerate(test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.to(device))
                labelsV = Variable(labels.to(device))
                weightV = Variable(weight.to(device))
                heightV = Variable(height.to(device))

        x = samplesV.transpose(0, 1)
        x = self.vrae.encoder(x)
        x = self.vrae.lmbd(x)

        for m in range(0, 20):
            if lSpeeds[m] == 0:

                mod=torch.tensor((90.0+float(m))/100.0).to(device)
                lSpeeds[m] = self(x, weightV, heightV*mod).cpu().detach().numpy()
            else:
                lSpeeds[m] = lSpeeds[m] * 0.98 + self(x, weightV, heightV) * 0.02

        print(lSpeeds)

