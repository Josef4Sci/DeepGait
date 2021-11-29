from vrae.utils import *
import torch.nn.functional as F

def autocorelation(data, zero_mean = True , zero_padding=True):
    input_FFT = data
    if zero_mean:
        input_FFT -= input_FFT.mean(dim=0)[None, :]
    if zero_padding:
        input_FFT = torch.cat([torch.zeros([int(input_FFT.shape[0]/2), input_FFT.shape[1]]), input_FFT, torch.zeros([int(input_FFT.shape[0]/2), input_FFT.shape[1]])])

    fft = torch.rfft(input_FFT.permute(1, 0), 1, normalized=False, onesided=True)
    abs_val_sqr = fft[:, :, 0].pow(2) + fft[:, :, 1].pow(2)
    prepare_for_ifft = torch.stack([abs_val_sqr, torch.zeros(18, 513)]).permute(1, 2, 0)
    ifft = torch.irfft(prepare_for_ifft, 1, normalized=False, onesided=True)

    return ifft


def get_base_periode(ifft, index_min=50, index_max=512, window=30):
    sumIfftHalf = ifft[:, index_min:index_max].sum(dim=0)
    window_maxima = torch.nn.functional.max_pool1d_with_indices(sumIfftHalf.view(1, 1, -1), window, 1, padding=window // 2)[1].squeeze()
    candidates = window_maxima.unique()
    nice_peaks = candidates[(window_maxima[candidates] == candidates).nonzero()]
    return nice_peaks[torch.argmax(sumIfftHalf[nice_peaks], dim=0)]+index_min

def my_FFT(input_FFT):
    out=torch.rfft((input_FFT-input_FFT.mean(dim=0)[None, :]).permute(1, 0), 1, normalized=False, onesided=True)
    return out

train_dataset, test_dataset, test_dataset2 = open_data_pepa2(train='train4rn', test='test5')
t=train_dataset.tensors[0]

for i in range(0,t.shape[0]):
    indata = torch.squeeze(t[i, :, :])
    ifft = autocorelation(indata)
    periode = get_base_periode(ifft)
    normsized = F.interpolate(indata[0:periode, :], size=(512, 18))


fin=0




