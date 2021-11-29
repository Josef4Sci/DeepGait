import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.autograd import Variable
from tqdm import tqdm

import matplotlib.pyplot as plt
# mpl.use('Agg')
import matplotlib

def showData(x, y, batchN, sen):
    plt.close()
    plt.plot(x.detach().cpu()[:, batchN, sen])
    plt.plot(y.detach().cpu()[:, batchN, sen])
    plt.show()
    # plt.savefig("mygraph.png")

def showData2(x, y, batchN, sen):
    plt.close()
    plt.plot(x.detach().cpu()[batchN, sen, :])
    plt.plot(y.detach().cpu()[batchN, sen, :])
    plt.show()

class SpeedNETUtilsClass:

    @staticmethod
    def showData(x, y, batchN, sen):
        plt.close()
        plt.plot(x.detach().cpu()[batchN, sen, :])
        plt.plot(y.detach().cpu()[batchN, sen, :])
        plt.show()

    @staticmethod
    def showDataTest(xi, y, batchN, sen):
        a=matplotlib.get_backend()

        x = np.linspace(0, 6.28, 100)

        plt.plot(x, x ** xi, label='square root')
        plt.plot(x, np.sin(x), label='sinc')

        plt.xlabel('x label')
        plt.ylabel('y label')

        plt.title("test plot")

        plt.legend()

        plt.show()

    def filter_step(self, SpeedNetTeacher, SpeedNetStudent, samples, labels, trainW, trainH, optimizer, loss1, train=True):

        samples = samples.transpose(0, 1)
        # labelsV = Variable(labels.type(SpeedNetStudent.dtype), requires_grad=True)
        weightV = Variable(trainW.type(SpeedNetStudent.dtype), requires_grad=True)
        heightV = Variable(trainH.type(SpeedNetStudent.dtype), requires_grad=True)

        if train:
            optimizer.zero_grad()
        # Forward + Backward + Optimize
        samplesV = Variable(samples.type(SpeedNetStudent.dtype), requires_grad=True)

        y, x_decod, latentT, kl_loss, aten = SpeedNetTeacher(samplesV, weightV, heightV)

        y, x_decod, latentS, kl_loss, aten = SpeedNetStudent(samplesV, weightV, heightV)

        loss = loss1(latentS, latentT)

        recon_loss = loss1(samplesV, x_decod)

        loss1res = self.lossMultip * loss

        if train:
            loss1res.backward()
            optimizer.step()

        return loss, recon_loss


    def studenTeacherReduction(self, SpeedNetTeacher, SpeedNetStudent, train_dataset, learning_rate, num_epochs):

        self.lossMultip=1
        # trainable_params = list(filter(
        #     lambda p: p.requires_grad, SpeedNetStudent.parameters()))
        # trainable_params = list(self.fc1.parameters())+list(self.fc2.parameters())+list(self.fc3.parameters())\
        #                       +list(self.fc3_2.parameters())+list(self.fc4.parameters())+list(self.ensamble2.parameters())\
        #                       +list(self.ensamble1_weight.parameters())+list(self.ensamble1_height.parameters())
        trainable_params = list(SpeedNetStudent.encoder.parameters()) + list(SpeedNetStudent.lmbd.parameters())

        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=SpeedNetStudent.batch_size, shuffle=True,
                                       drop_last=True)

        SpeedNetTeacher.to(SpeedNetTeacher.device)
        SpeedNetStudent.to(SpeedNetStudent.device)

        MAE = nn.L1Loss().to(SpeedNetStudent.device)
        MSE = nn.MSELoss().to(SpeedNetStudent.device)

        self.epoch_stop = 0

        for epoch in range(num_epochs):
            print('Epoch:', epoch)

            SpeedNetStudent.train()
            SpeedNetTeacher.train()

            steps = 0
            loss_x = 0
            loss_rec=0

            torch.cuda.empty_cache()
            # training
            for (samples, labels, weight, height) in tqdm(train_data_loader):
                steps += 1

                loss, loss_r = self.filter_step(SpeedNetTeacher, SpeedNetStudent, samples, labels, weight, height, optimizer, MSE)
                loss_x += loss.item()
                loss_rec += loss_r.item()

            print("Training loss:", loss_x / steps)
            print("Training loss rec:", loss_rec / steps)
            SpeedNetStudent.logger.log_value('{} MSE'.format("train"), loss_x / steps)
            SpeedNetStudent.logger.log_value('{} VRAE'.format("train"), loss_rec / steps)


    def analyzeErrorDistribution(self, SpeedNET, test_dataset, divide):
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=SpeedNET.batch_size, shuffle=True, drop_last=True)

        maxSp = max(test_dataset.tensors[1].numpy().squeeze())
        stepSp = maxSp / (divide - 1)

        SpeedNET.to(SpeedNET.device)
        steps = 0

        data = np.empty(divide, dtype=np.object)
        for i in range(data.shape[0]):
            data[i] = []

        mae=[]

        for i, (samples, labels, weight, height) in enumerate(test_data_loader):
            with torch.no_grad():
                samples = samples.transpose(1, 2)
                labelsV = Variable(labels.type(SpeedNET.dtype), requires_grad=False)
                weightV = Variable(weight.type(SpeedNET.dtype), requires_grad=False)
                heightV = Variable(height.type(SpeedNET.dtype), requires_grad=False)
                samplesV = Variable(samples.type(SpeedNET.dtype), requires_grad=False)

                # heightV = Variable(torch.tensor(np.arange(16)*2 + 70).type(self.dtype), requires_grad=False).view(16,1)
                #weightV = Variable(torch.tensor(np.arange(16) * 2 + 60).type(SpeedNET.dtype), requires_grad=False).view(16, 1)

                y = SpeedNET(samplesV, weightV, heightV)

                err = labelsV - y

                indexes = (labels.numpy().squeeze() / stepSp)
                err = err.detach().cpu().numpy()
                mae.append(np.absolute(err))

                for i in range(SpeedNET.batch_size):
                    ind = int(indexes[i])
                    if ind < 0:
                        ind = 0
                    if ind > divide:
                        ind = divide
                    data[ind].append(err[i][0])
                steps += 1

        x = np.linspace(0, maxSp, divide)

        return data, x, np.mean(mae)