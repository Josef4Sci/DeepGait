import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from myUtils import TensorboardLogger

from torch.autograd import Variable
from tqdm import tqdm

import os
from os import path


# import matplotlib.pyplot as plt
# import matplotlib


class EncoderConv(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """

    def __init__(self, number_of_features, channels):
        super(EncoderConv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(number_of_features, channels, (5,), stride=(1,), padding=(2,)),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, (5,), stride=(2,), padding=(2,)),
            nn.ReLU(True),
            nn.Conv1d(channels, channels * 2, (5,), stride=(1,), padding=(2,)),
            nn.ReLU(True),
            nn.Conv1d(channels * 2, channels * 2, (3,), stride=(2,), padding=(1,)),
            nn.ReLU(True),
            nn.Conv1d(channels * 2, channels * 4, (3,), stride=(1,), padding=(1,)),
            nn.ReLU(True),
            nn.Conv1d(channels * 4, channels * 4, (3,), stride=(2,), padding=(1,)),
            nn.ReLU(True),
            nn.Conv1d(channels * 4, channels * 4, (3,), stride=(1,), padding=(1,))
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


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
        x = x.permute(2, 0, 1)
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


class DecoderSin(nn.Module):
    def __init__(self, sequence_length, batch_size, latent_length, output_size, depth,
                 step_size, dtype, device):
        super(DecoderSin, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype
        self.depth = depth

        self.base = torch.tensor(np.arange(sequence_length) * 2 * 3.14 / sequence_length, requires_grad=False).type(
            self.dtype).view(1, sequence_length).expand(self.depth, -1).type(self.dtype).to(device)

        self.A_lin = nn.Linear(self.latent_length, self.output_size * self.depth)
        self.B_lin = nn.Linear(self.latent_length, self.output_size * self.depth)
        self.C_lin = nn.Linear(self.latent_length, self.output_size * self.depth)
        self.D_lin = nn.Linear(self.latent_length, self.output_size)

        self.B_lin.bias = nn.Parameter(torch.tensor(np.arange(self.depth * step_size, step=step_size, ))
                                       .repeat(self.output_size).type(dtype))

    def forward(self, latent):
        A = self.A_lin(latent.permute(0, 1))
        C = self.C_lin(latent.permute(0, 1))
        D = self.D_lin(latent.permute(0, 1))
        B = self.B_lin(latent.permute(0, 1)).view(self.batch_size, self.depth, 1, self.output_size)

        A1 = A.view(self.batch_size, self.depth, 1, self.output_size) \
            .expand(self.batch_size, self.depth, self.sequence_length, self.output_size)
        C1 = C.view(self.batch_size, self.depth, 1, self.output_size) \
            .expand(self.batch_size, self.depth, self.sequence_length, self.output_size)
        B1 = B.view(self.batch_size, self.depth, 1, self.output_size) \
            .expand(self.batch_size, self.depth, self.sequence_length, self.output_size)

        b11 = self.base.view(1, self.depth, self.sequence_length, 1)
        b12 = b11.expand(self.batch_size, self.depth, self.sequence_length, self.output_size)
        b1 = b12 * B1

        sin = torch.sin(b1 + C1) * A1
        out = torch.sum(sin, dim=1) + D.view(self.batch_size, 1, self.output_size).expand(self.batch_size,
                                                                                          self.sequence_length,
                                                                                          self.output_size)

        out = out.transpose(1, 2)  # out.permute(1, 0, 2)

        if torch.isnan(out).any():
            stop = 0

        return out


class SpeedNET2(nn.Module):
    """
    Deep Feed Forward Network
    """

    def __init__(self, sequence_length, hidden_size, hidden_size_FC, number_of_features, sin_depth, channels_conv,
                 batch_size=32,
                 hidden_layer_depth=2, latent_length=20, reducing=False, reducePerc=0.95,
                 learning_rate=0.005, dropout_rate=0., block='LSTM', kl_loss_multip=0.0001, lossDFFmultip=1,
                 lossEncodMultip=1, dropRateFirstLay=0.0, dropRate=0.0, cudaNum=0,
                 logerPath="C:/Users/josef/Desktop/DeepLearning/Logs/LogOneWheele1/",
                 mess=None):

        super(SpeedNET2, self).__init__()

        self.device = torch.device("cuda:" + str(cudaNum))

        self.logger = TensorboardLogger(logerPath, mess)
        self.logger.log_txt("sin no hidden", 0)
        self.logger.log_txt("sin cely latent", 0)
        self.logger.log_txt("latent size", latent_length)

        if mess:
            self.logger.log_txt("Mess ", mess)

        # torch.set_default_dtype(dtype)
        # self.dtype = dtype
        self.dtype = torch.cuda.FloatTensor

        self.batch_size = batch_size
        self.number_of_features = number_of_features
        self.lossDFFmultip = lossDFFmultip
        self.lossEncodMultip = lossEncodMultip

        self.lossChange = 0

        self.hidden_size = hidden_size
        self.dropRate = dropRate
        self.dropRateFirstLay = dropRateFirstLay

        self.encoder_conv = EncoderConv(number_of_features=number_of_features, channels=channels_conv)

        self.encoder = Encoder(number_of_features=channels_conv * 4,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block).to(self.device)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length).to(self.device)

        # self.decoder = Decoder(sequence_length=sequence_length,
        #                        batch_size=batch_size,
        #                        hidden_size=hidden_size,
        #                        hidden_layer_depth=hidden_layer_depth,
        #                        latent_length=latent_length,
        #                        output_size=number_of_features,
        #                        block=block,
        #                        dtype=self.dtype).to(self.device)

        self.decoder = DecoderSin(sequence_length=sequence_length,
                                  batch_size=batch_size,
                                  latent_length=latent_length,
                                  output_size=number_of_features,
                                  depth=sin_depth,
                                  step_size=0.3,
                                  dtype=self.dtype,
                                  device=self.device).to(self.device)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size_FC
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.kl_loss_multip = kl_loss_multip

        self.fc1 = nn.Linear(int(latent_length), 1).type(self.dtype).to(self.device)

        self.ensamble1_weight = nn.Linear(1, 1).type(self.dtype).to(self.device)
        self.ensamble1_height = nn.Linear(1, 1).type(self.dtype).to(self.device)
        self.ensamble2 = nn.Linear(2, self.hidden_size).type(self.dtype).to(self.device)

        self.reducing = reducing

        self.reduceStepPercent = reducePerc

        self.zerovectState = nn.Parameter(
            torch.ones([1, latent_length]).type(self.dtype), requires_grad=False)

        self.zerovect = self.zerovectState.data[0, :].expand(batch_size, latent_length).to(self.device)

        self.learning_rate = learning_rate

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

    def forward(self, x, weight, height):

        x = self.encoder_conv(x)
        x = self.encoder(x)  # LSTM encoder hidden 300
        latent, kl_loss = self.lmbd(x, self.kl_loss_multip)  # VRAE latent, 200 mean, 200 variation
        x_decoded = self.decoder(latent)  # Sin decod (could be LSTM for short signals)

        weight = F.leaky_relu(self.ensamble1_weight(weight))  # some stuff to connect other info
        height = F.leaky_relu(self.ensamble1_height(height))
        features = torch.cat((weight, height), dim=1)

        # attention = torch.sigmoid(self.ensamble2(features)) * 2
        attention = self.ensamble2(features)

        x = F.leaky_relu(self.fc1(latent))

        x = x * attention

        return x, x_decoded, latent, kl_loss, attention

    def nullLeastSignificant(self, latentResponses, percent=0.95):

        nullCount = (int)(self.zerovectState.sum() * (1.0 - percent))
        variance = torch.var(latentResponses, dim=0)
        a, b = torch.topk(variance, nullCount, dim=0, largest=False, sorted=False, out=None)
        self.zerovect[:, b] = torch.tensor(0).type(self.dtype)
        self.zerovectState.data = self.zerovect[0, :]

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
        self.lossVrae = 0

        self.counter = 0
        self.samplesIncluded = 1
        self.reduceAccur = 1.2e-3
        self.epoch_stop = 0

        lastErrors = []

        validErrors = []
        lastErrorsValidStop = []

        for epoch in range(num_epochs):
            # while self.samplesIncluded<200:
            print('Epoch:', epoch)

            self.train()

            steps = 0
            loss_x = 0
            loss_vrae_x = 0
            latentBufffer = torch.empty((1, 200)).type(self.dtype)

            torch.cuda.empty_cache()
            string1 = list()
            # training
            for (samples, labels, weight, height) in tqdm(train_data_loader):
                steps += 1

                loss, loss_dummy, loss_vrae, latent, loss_sp = self.filter_step(samples, labels, weight, height,
                                                                                self.device,
                                                                                optimizer, MSE)
                loss_x += loss.item()
                loss_vrae_x += loss_vrae.item()

                if self.reducing and steps < 51 and epoch % 5 == 0 and self.lossVrae < self.reduceAccur:  #
                    latentBufffer = torch.cat((latentBufffer, latent))
                    if steps == 50:
                        self.nullLeastSignificant(latentBufffer, self.reduceStepPercent)

            print(loss_x / steps)

            self.logger.epoch = epoch
            self.logger.step(1)
            self.logger.log_value('loss', loss_x / steps)

            self.lossVrae = loss_vrae_x / steps

            if valid_dataset:
                self.eval()
                steps = 0
                loss_x = 0

                for i, (samples, labels, weight, height) in enumerate(valid_data_loader):
                    with torch.no_grad():
                        loss_speed, loss_dummy, loss_vrae, latent, loss_sp = self.filter_step(samples, labels, weight,
                                                                                              height,
                                                                                              self.device, optimizer,
                                                                                              MAE, train=False)
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
                loss_x = 0
                lossMAE_x = 0
                loss_vrae_x = 0

                for i, (samples, labels, weight, height) in enumerate(test_data_loader):
                    with torch.no_grad():
                        loss_MSE, loss_MAE, loss_vrae, latent, loss_sp = self.filter_step(samples, labels, weight,
                                                                                          height,
                                                                                          self.device, optimizer, MAE,
                                                                                          loss2=MAE, train=False)
                        loss_x += loss_MSE.item()
                        lossMAE_x += loss_MAE.item()
                        loss_vrae_x += loss_vrae.item()
                        steps += 1

                test_loss_MAE = lossMAE_x / steps
                test_loss_Vrae = loss_vrae_x / steps
                if test_loss_MAE < minimErrorMem:
                    minimErrorMem = test_loss_MAE
                    savePath = self.logger.log_dir + '/best.pth'
                    if path.exists(savePath):
                        os.remove(savePath)
                    torch.save(self.state_dict(), savePath)

                if (num_epochs - epoch) < 20:
                    lastErrors.append(test_loss_MAE)

                lastErrorsValidStop.append(test_loss_MAE)

                self.logger.log_value('{}_MSE'.format("test"), loss_x / steps)
                self.logger.log_value('{}_VRAE'.format("test"), loss_vrae_x / steps)
                self.logger.log_value('{}_MAE'.format("test"), test_loss_MAE)
                # print("Test loss:", loss_x / steps)
                print("Test loss MAE:", test_loss_MAE)

        lastAverage = sum(lastErrors) / len(lastErrors)
        return minimErrorMem, lastAverage

    def getError(self, test_dataset):

        trainable_params = list(filter(
            lambda p: p.requires_grad, self.parameters()))
        # trainable_params = list(self.fc1.parameters())+list(self.fc2.parameters())+list(self.fc3.parameters())\
        #                       +list(self.fc3_2.parameters())+list(self.fc4.parameters())+list(self.ensamble2.parameters())\
        #                       +list(self.ensamble1_weight.parameters())+list(self.ensamble1_height.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.to(self.device)

        MAE = nn.L1Loss().to(self.device)

        self.lossVrae = 0

        lableEstimList = []

        steps = 0
        loss_vrae_x = 0

        for i, (samples, labels, weight, height) in enumerate(test_data_loader):
            with torch.no_grad():
                loss_MSE, loss_MAE, loss_vrae, latent, loss_sp = self.filter_step(samples, labels, weight,
                                                                                  height,
                                                                                  self.device, optimizer, MAE,
                                                                                  loss2=MAE, train=False)

                lableEstimList.append([float(labels[0][0]), float(loss_MAE.detach().cpu())])
                loss_vrae_x += loss_vrae.item()
                steps += 1

        array = np.array(lableEstimList)
        scipy.io.savemat('test.mat', dict(x=array[:, 0], y=array[:, 1]))
        return array


    def filter_step(self, samples, labels, trainW, trainH, device, optimizer, loss1, loss2=None, train=True):

        samples = samples.transpose(1, 2)
        labelsV = Variable(labels.to(self.device), requires_grad=True)
        weightV = Variable(trainW.to(self.device), requires_grad=True)
        heightV = Variable(trainH.to(self.device), requires_grad=True)
        samplesV = Variable(samples.to(self.device), requires_grad=True)
        self.loss_fn = loss1

        # samples[int(samples.shape[0] / 2):, :, :] = torch.zeros(
        #     (int(self.sequence_length / 2), self.batch_size, self.number_of_features))

        if train:
            optimizer.zero_grad()
        # Forward + Backward + Optimize
        # samplesV1 = Variable(samples[0:int(samples.shape[0] / 2), :, :].type(self.dtype), requires_grad=True)
        # samplesV2 = Variable(samples[int(samples.shape[0] / 2):, :, :].type(self.dtype), requires_grad=False)

        y, x_decod, latent, kl_loss, aten = self(samplesV, weightV, heightV)

        recon_loss = self.loss_fn(samplesV, x_decod)
        # recon_loss = self.loss_fn(samplesV1, x_decod[0:int(samples.shape[0] / 2), :, :])

        # loss_speed = self.loss_fn(y.squeeze(), labelsV[:, 512])
        loss_speed = self.loss_fn(y, labelsV)

        loss_aten = loss1(aten, torch.ones((self.batch_size, 1)).type(self.dtype).to(self.device))

        if train:
            DFF_multip, encod_decod_multip = self.lossDFFmultip, self.lossEncodMultip
        else:
            DFF_multip, encod_decod_multip = 1, 0

        loss1res = loss_speed * DFF_multip + recon_loss * encod_decod_multip + kl_loss + 0.01 * loss_aten

        loss2res = 0
        if loss2:
            loss2res = loss2(y, labelsV)
        if train:
            loss1res.backward()
            optimizer.step()

        return loss1res, loss2res, recon_loss, latent, loss_speed

    def filter_step_atten(self):

        res=[]
        samplesV = Variable(torch.zeros(1, 18, 1024).to(self.device), requires_grad=False)
        for j in range(130):
            weightV = Variable(torch.tensor(float(j)).to(self.device), requires_grad=False).unsqueeze(0).unsqueeze(0)
            heigthV=Variable(torch.tensor(float(95)).to(self.device), requires_grad=False).unsqueeze(0).unsqueeze(0)
            y, x_decod, latent, kl_loss, aten = self(samplesV, weightV, heigthV)
            res.append(float(aten.detach().cpu()))

        res2 = []
        for j in range(130):
            weightV = Variable(torch.tensor(float(85)).to(self.device), requires_grad=False).unsqueeze(0).unsqueeze(0)
            heigthV=Variable(torch.tensor(float(j)).to(self.device), requires_grad=False).unsqueeze(0).unsqueeze(0)
            y, x_decod, latent, kl_loss, aten = self(samplesV, weightV, heigthV)
            res2.append(float(aten.detach().cpu()))

        return res, res2