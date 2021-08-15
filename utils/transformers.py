#coding=utf-8
import numpy as np
from scipy import signal

class SlideAndCut(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, n_segment=1, window_size=4992, sampling_rate=500, test_time_aug=False):
        self.n_segment = n_segment
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.test_time_aug = test_time_aug

    def __call__(self, sample):
        data = sample

        length = data.shape[1]
        # print("length:", length)
        if length < self.window_size:
            segments = []
            ecg_filled = np.zeros((data.shape[0], self.window_size))
            ecg_filled[:, 0:length] = data[:, 0:length]
            # ecg_filled = ecg_filling2(data, window_size)
            # try:
            #     ecg_filled = ecg_filling(data, sampling_rate, window_size)
            # except:
            #     ecg_filled = ecg_filling2(data, window_size)
            segments.append(ecg_filled)
            segments = np.array(segments)
        elif self.test_time_aug == False:
            # print("not using test-time-aug")
            offset = (length - self.window_size * self.n_segment) / (self.n_segment + 1)
            if offset >= 0:
                start = 0 + offset
            else:
                offset = (length - self.window_size * self.n_segment) / (self.n_segment - 1)
                start = 0
            segments = []
            recording_count = 0
            for j in range(self.n_segment):
                recording_count += 1
                # print(recording_count)
                ind = int(start + j * (self.window_size + offset))
                segment = data[:, ind:ind + self.window_size]
                segments.append(segment)
            segments = np.array(segments)
        elif self.test_time_aug == True:
            # print("using test-time-aug")
            ind = 0
            rest_length = length
            segments = []
            recording_count = 0
            while rest_length - ind >= self.window_size:
                recording_count += 1
                # print(recording_count)
                segment = data[:, ind:ind + self.window_size]
                segments.append(segment)
                ind += int(self.window_size / 2)
            segments = np.array(segments)
        data = segments[0]
        return data

class Transformation:
    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def get_params(self):
        return self.params


class BaseLineFilter:
    def __init__(self, window_size=1000):
        self.window_size = window_size

    def __call__(self, sample, **kwargs):
        for channel_idx in range(sample.shape[0]):
            running_mean = BaseLineFilter._running_mean(sample[channel_idx], self.window_size)
            sample[channel_idx] = sample[channel_idx] - running_mean
        return sample

    @staticmethod
    def _running_mean(sample, window_size):
        window = signal.windows.blackman(window_size)
        window = window / np.sum(window)
        return signal.fftconvolve(sample, window, mode="same")
