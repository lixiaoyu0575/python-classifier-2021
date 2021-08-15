import numpy as np
import scipy
import pywt
import pandas as pd

def WTfilt_1d(sig):
    # https://blog.csdn.net/weixin_39929602/article/details/111038295
    coeffs = pywt.wavedec(data=sig, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    # 将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)
    # 将其他中低频信号按软阈值公式滤波
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    # if np.isnan(rdata).any() == True:
    #     print(sig)
    #     print(rdata)
    return rdata

def filter_and_detrend(data):
    num_leads = len(data)
    filtered_data = pd.DataFrame()
    for k in range(num_leads):
        if np.sum(data[k]) == 0:
            filtered_data[k] = data[k]
        else:
            filtered_data[k] = WTfilt_1d(data[k])
        # try:
        #     filtered_data[k] = scipy.signal.detrend(WTfilt_1d(data[k]))
        # except ValueError:
        #     ##有些数据全是0，记录下来，无法进行detrend处理
        #     filtered_data[k] = WTfilt_1d(data[k])
    filtered_data = (filtered_data.values).T
    filtered_data[np.isnan(filtered_data)] = 0
    return filtered_data
#
# class filter_and_detrend(object):
#     """
#     Args:
#     """
#     def __init__(self):
#         pass
#
#     def __call__(self, data):
#         """
#         Args:
#             data: 12 lead ECG data . For example,the shape of data is (12,5000)
#         Returns:
#             Tensor: 12 lead ECG data after filtered and detrended
#         """
#
#         filtered_data = pd.DataFrame()
#         for k in range(12):
#             try:
#                 filtered_data[k] = scipy.signal.detrend(WTfilt_1d(data[k]))
#             except ValueError:
#                 ##有些数据全是0，记录下来，无法进行detrend处理
#                 filtered_data[k] = WTfilt_1d(data[k])
#
#         return (filtered_data.values).T
#
#     def __repr__(self):
#         return self.__class__.__name__