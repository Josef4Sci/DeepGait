import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from myUtils import TensorboardLogger

from torch.autograd import Variable
from tqdm import tqdm

import SpeedNETUtils


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
            nn.Conv1d(channels, channels*2, (5,), stride=(1,), padding=(2,)),
            nn.ReLU(True),
            nn.Conv1d(channels*2, channels*2, (3,), stride=(2,), padding=(1,)),
            nn.ReLU(True),
            nn.Conv1d(channels*2, channels * 4, (3,), stride=(1,), padding=(1,)),
            nn.ReLU(True),
            nn.Conv1d(channels * 4, channels * 4, (3,), stride=(2,), padding=(1,)),
            nn.ReLU(True),
            nn.Conv1d(channels * 4, channels * 4, (3,), stride=(1,), padding=(1,))
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


class DecoderConv2(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """

    def __init__(self, number_of_features, channels, latent_length):
        super(DecoderConv2, self).__init__()

        self.channels=channels
        self.decoderLin = nn.Sequential(
            nn.Linear(latent_length, 32 * channels * 4),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(1,), padding=(1,), bias=True),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(2,), padding=(1,), bias=False, output_padding=(1,)),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(1,), padding=(1,), bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(2,), padding=(1,), bias=False,output_padding=(1,)),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(1,), padding=(1,), bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(2,), padding=(1,), bias=False, output_padding=(1,)),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 2, (3,), stride=(1,), padding=(1,), bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 2, channels * 2, (3,), stride=(2,), padding=(1,), bias=False, output_padding=(1,)),
            nn.ELU(True),
            nn.ConvTranspose1d(channels * 2, channels, (5,), stride=(1,), padding=(2,), bias=False),
            nn.ELU(True),
            nn.ConvTranspose1d(channels, channels, (5,), stride=(2,), padding=(2,), bias=False, output_padding=(1,)),
            nn.ELU(True),
            nn.ConvTranspose1d(channels, number_of_features, (5,), stride=(1,), bias=False , padding=(2,))
        )

    def forward(self, x):
        a = self.decoderLin(x)
        b = a.view((x.shape[0], self.channels * 4, 32))
        out = self.decoder(b)
        return out

class DecoderConv3(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """

    def __init__(self, number_of_features, channels, latent_length):
        super(DecoderConv3, self).__init__()

        self.channels=channels

        self.decoderLin = nn.Linear(latent_length, 32 * channels * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(1,), padding=(1,), bias=True),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(2,), padding=(1,), bias=False, output_padding=(1,)),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(1,), padding=(1,), bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(2,), padding=(1,), bias=False,output_padding=(1,)),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(1,), padding=(1,), bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 4, channels * 4, (3,), stride=(2,), padding=(1,), bias=False, output_padding=(1,)),
            nn.ConvTranspose1d(channels * 4, channels * 2, (3,), stride=(1,), padding=(1,), bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels * 2, channels * 2, (3,), stride=(2,), padding=(1,), bias=False, output_padding=(1,)),
            nn.ConvTranspose1d(channels * 2, channels, (5,), stride=(1,), padding=(2,), bias=False),
            nn.ELU(True),
            nn.ConvTranspose1d(channels, channels, (5,), stride=(2,), padding=(2,), bias=False, output_padding=(1,)),
            nn.ConvTranspose1d(channels, number_of_features, (5,), stride=(1,), bias=False , padding=(2,))
        )

    def forward(self, x):
        a = self.decoderLin(x)
        b = a.view((x.shape[0], self.channels * 4, 32))
        out = self.decoder(b)
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

    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, block='LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth)
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
        x=x.permute(2, 0, 1)
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


class Decoder(nn.Module):
    """Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """

    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype,
                 block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(
            self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)

        out=out.permute(1, 2, 0)

        return out


class SpeedNetConvLSTM(nn.Module):
    """
    Deep Feed Forward Network
    """

    def __init__(self, sequence_length, hidden_size, convChanels, number_of_features, batch_size=32,
                 hidden_layer_depth=2, latent_length=20,
                 learning_rate=0.005, kl_loss_multip=0.0001,
                 lossEncodMultip=1, cudaNum=0, weight_decay=0,
                 logerPath="C:/Users/josef/Desktop/DeepLearning/Logs/LogOneWheele1/",
                 mess=None):

        super(SpeedNetConvLSTM, self).__init__()

        self.device = torch.device("cuda:" + str(cudaNum))

        self.stop_epoch_valid = 120

        self.logger = TensorboardLogger(logerPath, mess)
        self.logger.log_txt("ConvLSTM network", 0)
        self.logger.log_txt("latent size", latent_length)
        self.logger.log_txt("Multiplier autoencod/fully", lossEncodMultip)

        if mess:
            self.logger.log_txt("Mess ", mess)

        self.dtype = torch.cuda.FloatTensor

        self.batch_size = batch_size
        self.number_of_features = number_of_features
        self.lossEncodMultip = lossEncodMultip
        self.weight_decay = weight_decay

        self.lossChange = 0

        self.hidden_size = hidden_size

        self.conv_encod = EncoderConv(number_of_features=number_of_features,
                                       channels=convChanels).to(self.device)

        self.LSTM_encod = Encoder(number_of_features=convChanels*4,
                                  hidden_size=hidden_size,
                                  hidden_layer_depth=hidden_layer_depth,
                                  latent_length=latent_length).to(self.device)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length).to(self.device)

        # self.LSTM_decod = Decoder(hidden_layer_depth=hidden_layer_depth,
        #                           sequence_length=sequence_length//8,
        #                           hidden_size=hidden_size,
        #                           latent_length=latent_length,
        #                           batch_size=batch_size,
        #                           dtype=self.dtype,
        #                           output_size=convChanels * 4,
        #                           ).to(self.device)

        # self.conv_decod = DecoderConv(number_of_features=number_of_features,
        #                                channels=convChanels).to(self.device)
        self.conv_decod = DecoderConv3(number_of_features=number_of_features,
                                       channels=convChanels, latent_length=latent_length).to(self.device)

        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.kl_loss_multip = kl_loss_multip

        self.fc1 = nn.Linear(int(latent_length), 1).type(self.dtype).to(self.device)

        # self.ensamble1_weight = nn.Linear(1, 1).type(self.dtype).to(self.device)
        # self.ensamble1_height = nn.Linear(1, 1).type(self.dtype).to(self.device)
        # self.ensamble2 = nn.Linear(2, self.hidden_size).type(self.dtype).to(self.device)

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

        x = self.conv_encod(x)
        x = self.LSTM_encod(x)
        latent, kl_loss = self.lmbd(x, self.kl_loss_multip)
        # x = self.LSTM_decod(latent)
        x_decoded = self.conv_decod(latent)

        # weight = F.leaky_relu(self.ensamble1_weight(weight)) #some stuff to connect other info
        # height = F.leaky_relu(self.ensamble1_height(height))
        # features = torch.cat((weight, height), dim=1)
        # attention = torch.sigmoid(self.ensamble2(features)) * 2
        #
        x = F.relu(self.fc1(latent))

        # x = self.fc4(x * attention)

        return x, x_decoded, kl_loss

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
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

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

                loss1res, recon_loss, loss_speed = self.filter_step(samples, labels, weight, height,
                                                                                self.device,
                                                                                optimizer, MSE)
                loss_x += loss1res.item()
                loss_vrae_x += recon_loss.item()

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
                        loss1res, recon_loss, loss_speed = self.filter_step(samples, labels, weight,
                                                                                              height,
                                                                                              self.device, optimizer,
                                                                                              MAE, train=False)
                        loss_x += loss_speed.item()
                        steps += 1

                print("Valid loss:", loss_x / steps)
                self.logger.log_value('{}_MAE'.format("valid"), loss_x / steps)

                validErrors.append(loss_x / steps)
                if min(validErrors[-self.stop_epoch_valid:]) != min(validErrors):
                    lastErrors = lastErrorsValidStop[-self.stop_epoch_valid:]
                    print('Finished at epoch, because of validation:', epoch)
                    break

            # test data1
            if test_dataset:
                steps = 0
                loss_x = 0
                loss_vrae_x = 0

                for i, (samples, labels, weight, height) in enumerate(test_data_loader):
                    with torch.no_grad():
                        loss1res, recon_loss, loss_speed = self.filter_step(samples, labels, weight,
                                                                                          height,
                                                                                          self.device, optimizer, MAE,
                                                                                          loss2=MAE, train=False)
                        loss_x += loss_speed.item()
                        loss_vrae_x += recon_loss.item()
                        steps += 1

                test_loss_MAE = loss_x / steps
                lastErrorsValidStop.append(test_loss_MAE)
                if (num_epochs - epoch) < self.stop_epoch_valid:
                    lastErrors.append(test_loss_MAE)

                self.logger.log_value('{}_VRAE'.format("test"), loss_vrae_x / steps)
                self.logger.log_value('{}_MAE'.format("test"), test_loss_MAE)
                # print("Test loss:", loss_x / steps)
                print("Test loss MAE:", test_loss_MAE)

        (samples, labels, weight, height)=next(iter(test_data_loader))
        self.recon_log(self.logger.writer, samples, weight, height)

        lastAverage = sum(lastErrors) / len(lastErrors)
        return minimErrorMem, lastAverage

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

        y, x_decod, k_loss = self(samplesV, weightV, heightV)

        recon_loss = self.loss_fn(samplesV, x_decod)
        # recon_loss = self.loss_fn(samplesV1, x_decod[0:int(samples.shape[0] / 2), :, :])

        # loss_speed = self.loss_fn(y.squeeze(), labelsV[:, 512])
        loss_speed = self.loss_fn(y, labelsV)

        # loss_aten = loss1(aten, torch.ones((self.batch_size, 1)).type(self.dtype).to(self.device))

        if train:
            encod_decod_multip = self.lossEncodMultip
        else:
            encod_decod_multip = 0

        loss1res = encod_decod_multip * loss_speed + recon_loss + k_loss

        if train:
            loss1res.backward()
            optimizer.step()

        if loss2:
            loss_speed = loss2(y, labelsV)

        return loss1res, recon_loss, loss_speed

    def recon_log(self, logger, samples, trainW, trainH):

        samples = samples.transpose(1, 2)
        weightV = Variable(trainW.to(self.device), requires_grad=True)
        heightV = Variable(trainH.to(self.device), requires_grad=True)
        samplesV = Variable(samples.to(self.device), requires_grad=True)

        y, x_decod, kloss = self(samplesV, weightV, heightV)

        for batch_sample in range(2):
            for i in range(self.sequence_length):
                logger.add_scalar(f'check_info/orig_ch0_b'+str(batch_sample), samples[batch_sample, 0, i], i)
                logger.add_scalar(f'check_info/decod_ch0_b'+str(batch_sample), x_decod[batch_sample, 0, i], i)
                logger.add_scalar(f'check_info/orig_ch4_b'+str(batch_sample), samples[batch_sample, 4, i], i)
                logger.add_scalar(f'check_info/decod_ch4_b'+str(batch_sample), x_decod[batch_sample, 4, i], i)



