import os
import json
import numpy as np
from numpy import inf
from scipy import signal
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import Dataset
import logging
# import neurokit2 as nk
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import matplotlib.pyplot as plt

# Utilty functions
# Data loading and processing

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Find Challenge files.
def load_label_files(label_directory):
    label_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f) # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            # root, ext = os.path.splitext(f)
            label_files.append(F)
    if label_files:
        return label_files
    else:
        raise IOError('No label or output files found.')

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

# Load weights.
def load_weights(weight_file):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)

    # For each collection of equivalent classes, replace each class with the representative class for the set.
    # rows = replace_equivalent_classes(rows, equivalent_classes)

    # Check that equivalent classes have identical weights.
    for j, x in enumerate(rows):
        for k, y in enumerate(rows[j+1:]):
            if x==y:
                assert(np.all(values[j, :]==values[j+1+k, :]))
                assert(np.all(values[:, j]==values[:, j+1+k]))

    # Use representative classes.
    classes = [x for j, x in enumerate(rows) if x not in rows[:j]]
    indices = [rows.index(x) for x in classes]
    weights = values[np.ix_(indices, indices)]

    return classes, weights, indices

# Load_table
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    print(os.getcwd())
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Divide ADC_gain and resample
def resample(data, header_data, resample_Fs = 300):
    # get information from header_data
    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs = int(tmp_hea[2])
    sample_len = int(tmp_hea[3])
    gain_lead = np.zeros(num_leads)

    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # divide adc_gain
    for ii in range(num_leads):
        data[ii] /= int(gain_lead[ii])

    resample_len = int(sample_len * (resample_Fs / sample_Fs))
    resample_data = signal.resample(data, resample_len, axis=1, window=None)

    return resample_data

def ecg_filling2(ecg, length):
    len = ecg.shape[1]
    ecg_filled = np.zeros((ecg.shape[0], length))
    ecg_filled[:, :len] = ecg
    sta = len
    while length - sta > len:
        ecg_filled[:, sta : sta + len] = ecg
        sta += len
    ecg_filled[:, sta:length] = ecg[:, :length-sta]

    return ecg_filled

import scipy
def slide_and_cut_beat_aligned(data, n_segment=1, window_size=3000, sampling_rate=300):
    channel_num, length = data.shape
    ecg_single_lead = data[1]
    # processed_ecg = nk.ecg_process(ecg_II, sampling_rate)
    ecg2save = np.zeros((10, channel_num, 400))
    info2save = np.zeros((10,))
    # ecg2save = []
    # info2save = []
    cleaned = nk.ecg_clean(ecg_single_lead, sampling_rate=sampling_rate)
    try:
        processed_ecg = nk.ecg_findpeaks(cleaned, sampling_rate=sampling_rate, method='neurokit')
        rpeaks = processed_ecg['ECG_R_Peaks']
        # ecg_segments = nk.ecg_segment(cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)
    except:
        return None, None
    for i in range(len(rpeaks) - 1):
        # key = str(i+1)
        # seg_values = ecg_segments[key].values
        # indexes = seg_values[:, 1]
        # start_index = indexes[0] if indexes[0] > 0 else 0
        # end_index = indexes[-1]
        start_index = rpeaks[i]
        end_index = rpeaks[i+1]
        beat = data[:, start_index:end_index]
        resample_ratio = beat.shape[1] / 400
        resampled_beat = scipy.signal.resample(beat, 400, axis=1) # Resample x to num samples using Fourier method along the given axis.
        ecg2save[i] = resampled_beat
        info2save[i] = resample_ratio
        # ecg2save.append(resampled_beat)
        # info2save.append(resample_ratio)
        if i >= 9:
            break
    # ecg2save = np.array(ecg2save)
    # info2save = np.array(info2save)
    return ecg2save, info2save

def slide_and_cut(data, n_segment=1, window_size=3000, sampling_rate=300, test_time_aug=False):
    length = data.shape[1]
    # print("length:", length)
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
            ind += int(window_size/2)
        segments = np.array(segments)
    return segments


# split into training and validation
def stratification(data_directory):
    classes_2021 = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
    ### equivalent SNOMED CT codes merged, noted as the larger one
    classes_2021 = classes_2021.split(',')
    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001'],
                          ['733534002', '164909002']]

    # input_directory_label = '/DATASET/challenge2020/All_data'
    # input_directory_label = '/data/ecg/raw_data/challenge2020/all_data_2021'
    # Find the label files.
    print('Finding label and output files...')
    label_files = load_label_files(data_directory)
    # label_files_tmp = []
    # for f in label_files:
    #     fname = f.split('/')[-1].split('.')[0]
    #     if fname[0] == 'A' or fname[0] == 'E':
    #         continue
    #     label_files_tmp.append(f)
    # label_files = label_files_tmp

    # Load the labels and classes.
    print('Loading labels and outputs...')
    labels_onehot = load_labels(label_files, classes_2021)

    X = np.zeros(len(labels_onehot))
    y = labels_onehot

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9, random_state=0)
    for train_index, val_index in msss.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        print('Saving split index...')
        savemat('model_training/split.mat', {'train_index': train_index, 'val_index': val_index})

    print('Stratification done.')

import time
# Training
def make_dirs(base_dir):

    checkpoint_dir = base_dir + '/checkpoints'
    log_dir = base_dir + '/log_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    tb_dir = base_dir + '/tb_log'
    result_dir = base_dir + '/results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    return result_dir, log_dir, checkpoint_dir, tb_dir

def init_obj(hype_space, name, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    """
    module_name = hype_space[name]['type']
    module_args = dict(hype_space[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def to_np(tensor, device):
    if device.type == 'cuda':
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

def get_mnt_mode(trainer):
    monitor = trainer.get('monitor', 'off')
    if monitor == 'off':
        mnt_mode = 'off'
        mnt_best = 0
        early_stop = 0
        mnt_metric_name = None
    else:
        mnt_mode, mnt_metric_name = monitor.split()
        assert mnt_mode in ['min', 'max']
        mnt_best = inf if mnt_mode == 'min' else -inf
        early_stop = trainer.get('early_stop', inf)

    return mnt_metric_name, mnt_mode, mnt_best, early_stop

def save_checkpoint(model, epoch, mnt_best, checkpoint_dir, file_name, classes, leads_num, config_json, save_best=True):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'monitor_best': mnt_best,
        'classes': classes,
        'leads': leads_num,
        'config': config_json
    }

    # save_path = checkpoint_dir + '/model_' + str(epoch) + '.pth'
    # torch.save(state, save_path)

    if save_best:
        best_path = checkpoint_dir + '/' + file_name
        torch.save(state, best_path)
        print("Saving current best: model_best.pth ...")

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def progress(data_loader, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(data_loader, 'n_samples'):
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
    else:
        current = batch_idx
        total = len(data_loader)
    return base.format(current, total, 100.0 * current / total)

def load_checkpoint(model, resume_path, logger):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    epoch = checkpoint['epoch']
    mnt_best = checkpoint['monitor_best']

    # load architecture params from checkpoint.
    model.load_state_dict(checkpoint['state_dict'])

    logger.info("Checkpoint loaded from epoch {}".format(epoch))

    return model

# Customed TensorDataset
class CustomTensorDataset_BeatAligned(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *tensors, transform=None, p=0.5):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.p = p

    def __getitem__(self, index):
        x = self.tensors[0][0][index]
        x2 = self.tensors[0][1][index]
        torch.randn(1)

        if self.transform:
            if torch.rand(1) >= self.p:
                x = self.transform(x)

        y = self.tensors[1][0][index]
        y2 = self.tensors[1][1][index]
        w = self.tensors[2][index]

        return [x, x2], [y, y2], w

    def __len__(self):
        return self.tensors[0][0].size(0)

# Customed TensorDataset
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *tensors, transform=None, lead_number=12):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.lead_number = lead_number

    def __getitem__(self, index):
        x = self.tensors[0][index]
        torch.randn(1)

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        w = self.tensors[2][index]


        ### 12 leads order: I II III aVL aVR aVF V1 V2 V3 V4 V5 V6
        if self.lead_number == 2:  # two leads: I II
            leads_index = [0, 1]
        elif self.lead_number == 3:  # three leads: I II V2
            leads_index = [0, 1, 7]
        elif self.lead_number == 4:  # four leads: I II III V2
            leads_index = [0, 1, 2, 7]
        elif self.lead_number == 6:  # six leads: I II III aVL aVR aVF
            leads_index = [0, 1, 2, 3, 4, 5]
        elif self.lead_number == 8:  # eight leads
            leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        else:  # twelve leads
            leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        x = x[leads_index, :]

        return x, y, w

    def __len__(self):
        return self.tensors[0].size(0)
