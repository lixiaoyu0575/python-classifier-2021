import torch
from torch.utils.data.dataset import Dataset
import os
import pickle as dill
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import torch


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


# Find Challenge files.
def my_find_challenge_files(label_directory):
    label_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f)  # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            # root, ext = os.path.splitext(f)
            label_files.append(F)
    if label_files:
        return label_files
    else:
        raise IOError('No label or output files found.')


# For each set of equivalent classes, replace each class with the representative class for the set.
def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes:
                classes[j] = multiple_classes[0]  # Use the first class as the representative class.
    return classes


# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table) - 1
    if num_rows < 1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i]) - 1 for i in range(num_rows))
    if len(num_cols) != 1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols < 1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j + 1] for j in range(num_rows)]
    cols = [table[i + 1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i + 1][j + 1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values


# Load weights.
def load_weights(weight_file):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert (rows == cols)

    # For each collection of equivalent classes, replace each class with the representative class for the set.
    # rows = replace_equivalent_classes(rows, equivalent_classes)

    # Check that equivalent classes have identical weights.
    for j, x in enumerate(rows):
        for k, y in enumerate(rows[j + 1:]):
            if x == y:
                assert (np.all(values[j, :] == values[j + 1 + k, :]))
                assert (np.all(values[:, j] == values[:, j + 1 + k]))

    # Use representative classes.
    classes = [x for j, x in enumerate(rows) if x not in rows[:j]]
    indices = [rows.index(x) for x in classes]
    weights = values[np.ix_(indices, indices)]

    return classes, weights, indices


# Load labels from header/label files.
def load_labels(label_files, classes):
    # The labels should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)
    num_classes = len(classes)

    labels_onehot = np.zeros((num_recordings, num_classes), dtype=np.bool)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        name = label_files[i].split('/')[-1].split('.')[0]
        with open(label_files[i], 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    dxs = [arr.strip() for arr in l.split(': ')[1].split(',')]
                    for dx in dxs:
                        if dx == "164909002":
                            dx = "733534002"
                        elif dx == "59118001":
                            dx = "713427006"
                        elif dx == "284470004":
                            dx = "63593006"
                        elif dx == "17338001":
                            dx = "427172004"
                        if dx in classes:
                            labels_onehot[i][classes.index(dx)] = 1
                            # add LBBB and RBBB to BBB
                            if dx == "733534002" or dx == "713427006":
                                labels_onehot[i][classes.index("6374002")] = 1
                            # deal with AF and AFL in Ningbo
                            if dx == "164890007" and name[0] == 'J' and int(name[2:]) > 10646:
                                labels_onehot[i][classes.index("164889003")] = 1
    return labels_onehot


# Load outputs from output files.
def load_outputs(output_files, classes, equivalent_classes):
    # The outputs should have the following form:
    #
    # diagnosis_1, diagnosis_2, diagnosis_3
    #           0,           1,           1
    #        0.12,        0.34,        0.56
    #
    num_recordings = len(output_files)
    num_classes = len(classes)

    # Load the outputs. Perform basic error checking for the output format.
    tmp_labels = list()
    tmp_binary_outputs = list()
    tmp_scalar_outputs = list()
    for i in range(num_recordings):
        with open(output_files[i], 'r') as f:
            lines = [l for l in f if l.strip() and not l.strip().startswith('#')]
            lengths = [len(l.split(',')) for l in lines]
            if len(lines) >= 3 and len(set(lengths)) == 1:
                for j, l in enumerate(lines):
                    arrs = [arr.strip() for arr in l.split(',')]
                    if j == 0:
                        row = arrs
                        row = replace_equivalent_classes(row, equivalent_classes)
                        tmp_labels.append(row)
                    elif j == 1:
                        row = list()
                        for arr in arrs:
                            number = 1 if arr in ('1', 'True', 'true', 'T', 't') else 0
                            row.append(number)
                        tmp_binary_outputs.append(row)
                    elif j == 2:
                        row = list()
                        for arr in arrs:
                            number = float(arr) if is_number(arr) else 0
                            row.append(number)
                        tmp_scalar_outputs.append(row)
            else:
                print('- The output file {} has formatting errors, so all outputs are assumed to be negative for this recording.'.format(
                    output_files[i]))
                tmp_labels.append(list())
                tmp_binary_outputs.append(list())
                tmp_scalar_outputs.append(list())

    # Use one-hot encoding for binary outputs and the same order for scalar outputs.
    # If equivalent classes have different binary outputs, then the representative class is positive if any equivalent class is positive.
    # If equivalent classes have different scalar outputs, then the representative class is the mean of the equivalent classes.
    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for j, x in enumerate(classes):
            indices = [k for k, y in enumerate(dxs) if x == y]
            if indices:
                binary_outputs[i, j] = np.any([tmp_binary_outputs[i][k] for k in indices])
                tmp = [tmp_scalar_outputs[i][k] for k in indices]
                if np.any(np.isfinite(tmp)):
                    scalar_outputs[i, j] = np.nanmean(tmp)
                else:
                    scalar_outputs[i, j] = float('nan')

    # If any of the outputs is a NaN, then replace it with a zero.
    binary_outputs[np.isnan(binary_outputs)] = 0
    scalar_outputs[np.isnan(scalar_outputs)] = 0

    return binary_outputs, scalar_outputs


def loaddata(data_path):
    ##TODO
    # further modification
    # data_path = '/data/weiyuhua/data/Challenge2018_500hz/preprocessed_data_new/'
    print("Loading data training set")
    with open(os.path.join(data_path, 'data_aug_train.pkl'), 'rb') as fin:
        res = dill.load(fin)
    x_train = res['trainset']
    y_train = res['traintarget']

    with open(os.path.join(data_path, 'data_aug_val.pkl'), 'rb') as fin:
        res = dill.load(fin)
    x_val = res['val_set']
    y_val = res['val_target']

    with open(os.path.join(data_path, 'data_aug_test.pkl'), 'rb') as fin:
        res = dill.load(fin)
    x_test = res['test_set']
    y_test = res['test_target']

    # x_train = x_train.swapaxes(1,2)
    # x_val = x_val.swapaxes(1,2)
    # x_test = x_test.swapaxes(1,2)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class ECGDatasetWithIndex(Dataset):
    '''Challenge 2017'''

    def __init__(self, X, Y):
        self.x, self.y = X, Y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.x[idx], self.y[idx], idx)

        return sample


# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


# Load challenge data.
def load_challenge_data(label_file, data_dir):
    file = os.path.basename(label_file)
    name, ext = os.path.splitext(file)
    with open(label_file, 'r') as f:
        header = f.readlines()
    mat_file = file.replace('.hea', '.mat')
    x = loadmat(os.path.join(data_dir, mat_file))
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header, name


# Divide ADC_gain and resample
def resample(data, header_data, resample_Fs=300):
    # get information from header_data
    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs = int(tmp_hea[2])
    sample_len = int(tmp_hea[3])
    gain_lead = np.zeros(num_leads)

    for ii in range(num_leads):
        tmp_hea = header_data[ii + 1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # divide adc_gain
    for ii in range(num_leads):
        data[ii] /= gain_lead[ii]

    resample_len = int(sample_len * (resample_Fs / sample_Fs))
    resample_data = signal.resample(data, resample_len, axis=1, window=None)

    return resample_data


def ecg_filling2(ecg, length):
    len = ecg.shape[1]
    ecg_filled = np.zeros((ecg.shape[0], length))
    ecg_filled[:, :len] = ecg
    sta = len
    while length - sta > len:
        ecg_filled[:, sta: sta + len] = ecg
        sta += len
    ecg_filled[:, sta:length] = ecg[:, :length - sta]

    return ecg_filled


def slide_and_cut(data, n_segment=1, window_size=3000, sampling_rate=300, test_time_aug=False):
    length = data.shape[1]
    print("length:", length)
    if length < window_size:
        segments = []
        ecg_filled = np.zeros((data.shape[0], window_size))
        ecg_filled[:, 0:length] = data[:, 0:length]
        # ecg_filled = ecg_filling2(data, window_size)
        # try:
        #     ecg_filled = ecg_filling(data, sampling_rate, window_size)
        # except:
        #     ecg_filled = ecg_filling2(data, window_size)
        segments.append(ecg_filled)
        segments = np.array(segments)
    elif test_time_aug == False:
        # print("not using test-time-aug")
        offset = (length - window_size * n_segment) / (n_segment + 1)
        if offset >= 0:
            start = 0 + offset
        else:
            offset = (length - window_size * n_segment) / (n_segment - 1)
            start = 0
        segments = []
        recording_count = 0
        for j in range(n_segment):
            recording_count += 1
            # print(recording_count)
            ind = int(start + j * (window_size + offset))
            segment = data[:, ind:ind + window_size]
            segments.append(segment)
        segments = np.array(segments)
    elif test_time_aug == True:
        # print("using test-time-aug")
        ind = 0
        rest_length = length
        segments = []
        recording_count = 0
        while rest_length - ind >= window_size:
            recording_count += 1
            # print(recording_count)
            segment = data[:, ind:ind + window_size]
            segments.append(segment)
            ind += int(window_size / 2)
        segments = np.array(segments)
    return segments


def custom_collate_fn(batch):
    data = [item[0].unsqueeze(0) for item in batch]
    target = [item[1].unsqueeze(0) for item in batch]
    return [data, target]
