from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from torch.utils.data import DataLoader
from myUtils import TensorboardLogger

from torch.autograd import Variable
from tqdm import tqdm

import os
from os import path
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_bands, base = base, device = device, dtype = dtype)
    scales = rearrange(scales, 's -> () () s')

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x * self.g / norm.clamp(min=self.eps)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        freq_base = 2,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        cross_dim = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        cudaNum=0,
        mess=None,
        logerPath='',
        learning_rate=0.001,
        batch=1
    ):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        self.batch_size = batch
        self.learning_rate = learning_rate
        self.dtype = torch.cuda.FloatTensor
        self.device = torch.device("cuda:" + str(cudaNum))

        if logerPath:
            self.logger = TensorboardLogger(logerPath, mess)

            with open(self.logger.log_dir + "/log.txt", "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write(mess + "\n")
                file_object.write("num_freq_bands " + str(num_freq_bands) + "\n")
                file_object.write("max_freq "+ str(max_freq) + "\n")
                file_object.write("depth "+ str(depth) + "\n")
                file_object.write("num_latents "+ str(num_latents) + "\n")
                file_object.write("latent_dim "+ str(latent_dim) + "\n")
                file_object.write("cross_dim "+ str(cross_dim) + "\n")
                file_object.write("cross_dim_head "+ str(cross_dim_head) + "\n")
                file_object.write("latent_dim_head "+ str(latent_dim_head) + "\n")

        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        if weight_tie_layers:
            get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                get_cross_attn(),
                get_cross_ff(),
                get_latent_attn(),
                get_latent_ff()
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, data, mask = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
        #pos = torch.tensor(torch.meshgrid(*axis_pos))
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base = self.freq_base)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)

        # concat to channels of data and flatten axis

        data = torch.cat((data, enc_pos), dim = -1)
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x
            x = latent_attn(x) + x
            x = latent_ff(x) + x

        x = x.mean(dim = -2)

        return self.to_logits(x)


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

            #torch.cuda.empty_cache()

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
                self.logger.log_value('{}_MAE'.format("valid"), loss_x / steps)

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

                self.logger.log_value('{}_MAE'.format("test"), test_loss_MAE)
                # print("Test loss:", loss_x / steps)
                print("Test loss MAE:", test_loss_MAE)

        lastAverage = sum(lastErrors) / len(lastErrors)
        return minimErrorMem, lastAverage

    def filter_step(self, samples, labels, trainW, trainH, device, optimizer, loss1, train=True):

        #samples = samples.transpose(1, 2)
        labelsV = Variable(labels.to(device), requires_grad=True)
        samplesV = Variable(samples.to(device), requires_grad=True)

        if train:
            optimizer.zero_grad()

        y = self(samplesV)
        y1 = torch.squeeze(y, 0)
        loss_speed = loss1(y1, labelsV)

        if train:
            loss_speed.backward()
            optimizer.step()

        return loss_speed#, loss_autoencod

