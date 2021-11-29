import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from torch.utils.data import DataLoader
from myUtils import TensorboardLogger

from torch.autograd import Variable
from tqdm import tqdm

from siren_pytorch import SirenNet, SirenWrapper, SirenWrapperLinear

# import matplotlib as mpl
# mpl.use('Agg')

import os
from os import path
import gc
import copy

class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block='LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout=dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout=dropout)
        else:
            raise NotImplementedError

        for param in self.model.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)

    # for weight in self.model.parameters():
    #         if len(weight.size()) > 1:
    #             torch.nn.init.orthogonal(weight.data)

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)
        # _, h_end = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output, kl_loss_multip):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """
        latent_mean, latent_logvar = self.hidden_to_mean(cell_output), self.hidden_to_logvar(cell_output)
        kl_loss = -0.5 * kl_loss_multip * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())

        if self.training:
            std = torch.exp(0.5 * latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(latent_mean), kl_loss
        else:
            return latent_mean, kl_loss

class DecoderSirene(nn.Module):
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, depth,
                 step_size, dtype, pretrain, hid_length, drop, device, block='LSTM'):

        super(DecoderSirene, self).__init__()

        self.net = SirenNet(
            dim_in=output_size,  # input dimension, ex. 2d coor
            dim_hidden=32,  # hidden dimension
            dim_out=output_size,  # output dimension, ex. rgb value
            num_layers=5,  # number of layers
            w0_initial=30.
            # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        self.wrapper = SirenWrapperLinear(
            self.net,
            sequence_lenght=1024
        )

    def forward(self, latent):
        out=self.wrapper(latent)
        return out


class SpeedNETsirene(nn.Module):
    """
    Deep Feed Forward Network
    """

    def __init__(self, sequence_length, hidden_size, hidden_size_FC, number_of_features, dtype, batch_size=32,
                 hidden_layer_depth=2, latent_length=20, reducing=False, reducePerc=0.95,
                 learning_rate=0.005, dropout_rate=0., block='LSTM', kl_loss_multip=0.0001, lossDFFmultip=1,
                 lossEncodMultip=1, dropRateFirstLay=0.0, dropRate=0.0, sineNetHidenSize=32, cudaNum=0,
                 logerPath="C:/Users/josef/Desktop/DeepLearning/Logs/LogOneWheele1/", mess=""):

        super(SpeedNETsirene, self).__init__()

        self.device = torch.device("cuda:"+str(cudaNum))

        self.logger = TensorboardLogger(logerPath)
        self.logger.log_txt("sinHiddenSize", sineNetHidenSize)
        self.logger.log_txt("message ", mess)
        self.logger.log_txt("sin cely latent", 0)
        #torch.set_default_dtype(dtype)
        #self.dtype = dtype
        self.dtype = torch.cuda.FloatTensor

        self.batch_size = batch_size
        self.number_of_features = number_of_features
        self.lossDFFmultip = lossDFFmultip
        self.lossEncodMultip = lossEncodMultip

        self.lossChange = 0

        self.hidden_size = hidden_size
        self.dropRate = dropRate
        self.dropRateFirstLay = dropRateFirstLay

        self.encoder = Encoder(number_of_features=number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block).to(self.device)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length).to(self.device)


        self.decoder = DecoderSirene(sequence_length=sequence_length,
                                  batch_size=batch_size,
                                  hidden_size=hidden_size,
                                  hidden_layer_depth=hidden_layer_depth,
                                  latent_length=latent_length,
                                  output_size=number_of_features,
                                  block=block,
                                  depth=100,
                                  step_size=0.3,
                                  dtype=self.dtype,
                                  pretrain=False,
                                  device=self.device,
                                  hid_length=sineNetHidenSize,
                                  drop=0.0).to(self.device)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size_FC
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.kl_loss_multip = kl_loss_multip

        self.fc1 = nn.Linear(int(self.latent_length/3)*2, hidden_size_FC).type(self.dtype).to(self.device)
        self.fc2 = nn.Linear(hidden_size_FC, hidden_size_FC).type(self.dtype).to(self.device)
        self.fc3 = nn.Linear(hidden_size_FC, hidden_size_FC).type(self.dtype).to(self.device)
        self.fc3_2 = nn.Linear(hidden_size_FC, hidden_size_FC).type(self.dtype).to(self.device)
        self.fc4 = nn.Linear(self.hidden_size, 1).type(self.dtype).to(self.device)

        self.ensamble1_weight = nn.Linear(1, 1).type(self.dtype).to(self.device)
        self.ensamble1_height = nn.Linear(1, 1).type(self.dtype).to(self.device)
        self.ensamble2 = nn.Linear(2, self.hidden_size).type(self.dtype).to(self.device)

        self.reducing = reducing

        self.reduceStepPercent = reducePerc

        self.learning_rate=learning_rate

        self.mult=1

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if sum([np.prod(p.size()) for p in own_state[name]]) != sum([np.prod(p.size()) for p in param]):
                continue
            own_state[name].copy_(param)

    def forward(self, x, W, H):

        cell_output = self.encoder(x)
        latent, kl_loss = self.lmbd(cell_output, self.kl_loss_multip)
        x_decoded = self.decoder(x)

        return x_decoded, latent, kl_loss



    def fit(self, num_epochs, train_dataset, test_dataset, test_dataset2=None, shuffleTrain=True, train_verif=0):

        trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        # trainable_params = list(self.fc1.parameters())+list(self.fc2.parameters())+list(self.fc3.parameters())\
        #                       +list(self.fc3_2.parameters())+list(self.fc4.parameters())+list(self.ensamble2.parameters())\
        #                       +list(self.ensamble1_weight.parameters())+list(self.ensamble1_height.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=shuffleTrain,
                                       drop_last=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if test_dataset2 != 0:
            test_data_loader2 = DataLoader(dataset=test_dataset2, batch_size=self.batch_size, shuffle=False,
                                           drop_last=True)

        self.to(self.device)

        MAE = nn.L1Loss().to(self.device)
        MSE = nn.MSELoss().to(self.device)

        minimErrorMem = 1000000000.0
        self.lossVrae = 0

        self.counter=0
        self.samplesIncluded = 1
        # self.reduceAccur = 1.2e-3
        self.epoch_stop = 0

        for epoch in range(num_epochs):
        # while self.samplesIncluded<200:
            print('Epoch:', epoch)

            self.train()

            steps = 0
            loss_x = 0
            loss_vrae_x = 0

            torch.cuda.empty_cache()

            # training
            for (samples, labels, weight, height) in tqdm(train_data_loader):
                steps += 1

                loss1res, loss2res, recon_loss, latent, loss_speed = self.filter_step(samples, labels, weight, height, optimizer, MSE)
                loss_x += loss1res.item()
                loss_vrae_x += recon_loss.item()

            print(loss_x/steps)

            self.logger.epoch = epoch
            self.logger.step(1)
            self.logger.log_value('loss', loss_x / steps)

            self.lossVrae = loss_vrae_x / steps

            if epoch==self.epoch_stop:
                stop=0

            # test train data without regularization
            if train_verif:
                self.eval()
                steps = 0
                loss_x = 0
                lossSpeed = 0
                loss_vrae_x = 0

                for i, (samples, labels, weight, height) in tqdm(train_data_loader):
                    with torch.no_grad():
                        loss1res, loss2res, recon_loss, latent, loss_speed = self.filter_step(samples, labels, weight, height, optimizer, MSE,
                                                                                 loss2=MAE, train=False)
                        loss_x += loss1res.item()
                        lossSpeed += loss_speed.item()
                        loss_vrae_x += recon_loss.item()
                        steps += 1

                        if steps>int(len(train_data_loader.dataset)*train_verif/self.batch_size):
                            break

                print("Training loss:", loss_x / steps)
                print("Training loss speed MAE:", lossSpeed / steps)
                self.logger.log_value('{} MSE'.format("train"), loss_x / steps)
                self.logger.log_value('{} MAE'.format("train"), lossSpeed / steps)
                self.logger.log_value('{} VRAE'.format("train"), loss_vrae_x / steps)

            # test data1
            if test_dataset:
                steps = 0
                loss_x = 0
                lossSpeed = 0
                loss_vrae_x = 0

                for i, (samples, labels, weight, height) in tqdm(test_data_loader):
                    with torch.no_grad():
                        loss1res, loss2res, recon_loss, latent, loss_speed = self.filter_step(samples, labels, weight, height, optimizer, MSE,
                                                                                 loss2=MAE, train=False)
                        loss_x += loss1res.item()
                        lossSpeed += loss_speed.item()
                        loss_vrae_x += recon_loss.item()
                        steps += 1

                test_loss_MAE = lossSpeed / steps
                test_loss_Vrae = loss_vrae_x / steps
                if test_loss_Vrae < minimErrorMem:
                    minimErrorMem = test_loss_Vrae
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
            if test_dataset2:
                steps = 0
                loss_x = 0
                loss_vrae_x = 0
                lossSpeed = 0

                for i, (samples, labels) in enumerate(test_data_loader2):
                    with torch.no_grad():
                        loss1res, loss2res, recon_loss, latent, loss_speed  = self.filter_step(samples, labels, self.optimizer, MSE,
                                                                                 loss2=MAE, train=False)
                        loss_x += loss1res.item()
                        lossSpeed += loss_speed.item()
                        loss_vrae_x += recon_loss.item()
                        steps += 1

                self.logger.log_value('{} MSE'.format("test2"), loss_x / steps)
                self.logger.log_value('{} MAE'.format("test2"), lossSpeed / steps)
                self.logger.log_value('{} VRAE'.format("test2"), loss_vrae_x / steps)

                # print("Test loss2:", loss_x / steps)
                print("Test loss MAE2:", lossSpeed / steps)

        konec = 0

    def filter_step(self, samples, labels, weigth, height, optimizer, loss1, loss2=None, train=True):

        samples = samples.transpose(0, 1)
        labelsV = Variable(labels.to(self.device), requires_grad=True)
        weigthV = Variable(weigth.to(self.device), requires_grad=True)
        heightV = Variable(height.to(self.device), requires_grad=True)
        samplesV = Variable(samples.to(self.device), requires_grad=True)
        self.loss_fn = loss1

        if train:
            optimizer.zero_grad()

        x_decod, latent, kl_loss = self(samplesV, weigthV, heightV)

        recon_loss = self.loss_fn(samplesV, x_decod)

        loss_speed = 0 #self.loss_fn(latent[:, 0], labelsV)

        loss1res = kl_loss + self.lossEncodMultip * recon_loss

        loss2res = 0
        # if loss2:
            # loss_speed = loss2(y, labelsV)
            # loss2res = loss_speed * DFF_multip + (kl_loss+recon_loss) * encod_decod_multip

        if train:
            loss1res.backward()
            optimizer.step()

        return loss1res, loss2res, recon_loss, latent, loss_speed

