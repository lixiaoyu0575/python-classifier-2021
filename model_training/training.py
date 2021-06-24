import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import time
from model_training.utils import loadmat, CustomTensorDataset, load_weights, load_labels, resample, slide_and_cut, load_challenge_data
from model_training.util import my_find_challenge_files
import os
from utils.denoising import filter_and_detrend

# Challenge Dataloaders and Challenge metircs

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, train_dataset, val_dataset, test_dataset,  batch_size, shuffle, num_workers,
                 collate_fn=default_collate, pin_memory=True):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.batch_idx = 0
        self.shuffle = shuffle

        self.init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': True
        }
        super().__init__(**self.init_kwargs)

        self.n_samples = len(self.train_dataset)

        self.valid_data_loader_init_kwargs = {
            'dataset': self.val_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': True
        }

        self.valid_data_loader = DataLoader(**self.valid_data_loader_init_kwargs)

        self.valid_data_loader.n_samples = len(self.val_dataset)

        if self.test_dataset:

            self.test_data_loader_init_kwargs = {
                'dataset': self.test_dataset,
                'batch_size': batch_size,
                'shuffle': False,
                'collate_fn': collate_fn,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'drop_last': True
            }

            self.test_data_loader = DataLoader(**self.test_data_loader_init_kwargs)

            self.test_data_loader.n_samples = len(self.test_dataset)

class ChallengeDataLoader(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, split_index, batch_size=128, shuffle=True, num_workers=0, resample_Fs=500, window_size=5000, n_segment=1,
                 normalization=False, training_size=None, augmentations=None, p=0.5, lead_number=12, save_data=False, load_saved_data=False):
        self.label_dir = label_dir
        self.dir2save_data = '/data/ecg/challenge2021/data/'
        dir2save_data = '/data/ecg/challenge2021/data/'
        start = time.time()

        # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
        weights_file = 'weights.csv'

        # Load the scored classes and the weights for the Challenge metric.
        print('Loading weights...')
        _, weights, indices = load_weights(weights_file)
        classes = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
        ### equivalent SNOMED CT codes merged, noted as the larger one
        classes = classes.split(',')
        self.weights = weights

        # Load the label and output files.
        print('Loading label and output files...')
        label_files = my_find_challenge_files(label_dir)
        labels_onehot = load_labels(label_files, classes)

        split_idx = loadmat(split_index)
        train_index, val_index = split_idx['train_index'], split_idx['val_index']
        train_index = train_index.reshape((train_index.shape[1],))
        if training_size is not None:  # for test
            train_index = train_index[0:training_size]
        val_index = val_index.reshape((val_index.shape[1],))
        # test_index = test_index.reshape((test_index.shape[1],))

        num_files = len(label_files)
        train_recordings = list()
        train_class_weights = list()
        train_labels_onehot = list()

        val_recordings = list()
        val_class_weights = list()
        val_labels_onehot = list()

        file_names = list()

        ### class weights for datasets
        # equivalent diagnose [['713427006', '59118001'], ['63593006', '284470004'], ['427172004', '17338001'], ['733534002', '164909002']]
        # CPSC
        CPSC_classes = ['270492004', '164889003', '733534002', '63593006', '426783006',
                        '713427006']  # "59118001" = "713427006"
        CPSC_class_weight = np.zeros((26,))
        for cla in CPSC_classes:
            CPSC_class_weight[classes.index(cla)] = 1
        # CPSC_extra
        CPSC_extra_excluded_classes = ['6374002', '39732003', '445118002', '251146004', '365413008',
                                       '164947007', '365413008', '164947007', '698252002', '426783006',
                                       '10370003', '111975006', '164917005', '47665007', '427393009',
                                       '426177001', '164934002', '59931005']
        CPSC_extra_class_weight = np.ones((26,))
        for cla in CPSC_extra_excluded_classes:
            CPSC_extra_class_weight[classes.index(cla)] = 0
        # PTB-XL
        PTB_XL_excluded_classes = ['6374002', '426627000', '365413008', '427172004']  # , '17338001'
        PTB_XL_class_weight = np.ones((26,))
        for cla in PTB_XL_excluded_classes:
            PTB_XL_class_weight[classes.index(cla)] = 0
        # G12ECG
        G12ECG_excluded_classes = ['10370003', '365413008', '164947007']
        G12ECG_class_weight = np.ones((26,))
        for cla in G12ECG_excluded_classes:
            G12ECG_class_weight[classes.index(cla)] = 0
        # Chapman Shaoxing
        Chapman_excluded_classes = ['6374002', '426627000', '713426002', '445118002', '10370003', '365413008',
                                    '427172004', '427393009', '427084000', '63593006']
        Chapman_class_weight = np.ones((26,))
        for cla in Chapman_excluded_classes:
            Chapman_class_weight[classes.index(cla)] = 0
        # Ningbo
        Ningbo_excluded_classes = ['164889003', '164890007', '426627000']
        Ningbo_class_weight = np.ones((26,))
        for cla in Ningbo_excluded_classes:
            Ningbo_class_weight[classes.index(cla)] = 0

        original_recordings = []
        if load_saved_data == False:
            ### preprocess data and label
            for i in range(num_files):
                print('{}/{}'.format(i + 1, num_files))
                recording, header, name = load_challenge_data(label_files[i], label_dir)

                if name[0] == 'S' or name[0] == 'I':  # filter PTB or St.P dataset
                    continue
                elif name[0] == 'A':  # CPSC
                    class_weight = CPSC_class_weight
                elif name[0] == 'Q':  # CPSC-extra
                    class_weight = CPSC_extra_class_weight
                elif name[0] == 'H':  # PTB-XL
                    class_weight = PTB_XL_class_weight
                elif name[0] == 'E':  # G12ECG
                    class_weight = G12ECG_class_weight
                elif name[0] == 'J' and int(name[2:]) <= 10646:  # Chapman
                    class_weight = Chapman_class_weight
                elif name[0] == 'J' and int(name[2:]) > 10646:  # Ningbo
                    class_weight = Ningbo_class_weight
                else:
                    print('warning! not from one of the datasets:  ', name)
                    continue

                recording[np.isnan(recording)] = 0

                # divide ADC_gain and resample
                recording = resample(recording, header, resample_Fs)

                # to filter and detrend samples
                recording = filter_and_detrend(recording)

                # slide and cut
                recording = slide_and_cut(recording, n_segment, window_size, resample_Fs)
                file_names.append(name)
                if i in train_index or name[0] == 'A' or name[0] == 'Q':
                    for j in range(recording.shape[0]):
                        train_recordings.append(recording[j])
                        train_labels_onehot.append(labels_onehot[i])
                        train_class_weights.append(class_weight)
                elif i in val_index:
                    for j in range(recording.shape[0]):
                        val_recordings.append(recording[j])
                        val_labels_onehot.append(labels_onehot[i])
                        val_class_weights.append(class_weight)
                else:
                    pass

            train_recordings = np.array(train_recordings)
            train_class_weights = np.array(train_class_weights)
            train_labels_onehot = np.array(train_labels_onehot)

            val_recordings = np.array(val_recordings)
            val_class_weights = np.array(val_class_weights)
            val_labels_onehot = np.array(val_labels_onehot)

        else:
            train_recordings = np.load(os.path.join(dir2save_data, 'train_recordings_' + 'windowSize' + str(
                window_size) + '_' + 'samplingRate' + str(
                resample_Fs) + '.npy'))
            train_class_weights = np.load(os.path.join(dir2save_data, 'train_class_weights_' + 'windowSize' + str(
                window_size) + '_' + 'samplingRate' + str(
                resample_Fs) + '.npy'))
            train_labels_onehot = np.load(os.path.join(dir2save_data, 'train_labels_onehot_' + 'windowSize' + str(
                window_size) + '_' + 'samplingRate' + str(
                resample_Fs) + '.npy'))
            val_recordings = np.load(os.path.join(dir2save_data, 'val_recordings_' + 'windowSize' + str(
                window_size) + '_' + 'samplingRate' + str(
                resample_Fs) + '.npy'), )
            val_class_weights = np.load(os.path.join(dir2save_data, 'val_class_weights_' + 'windowSize' + str(
                window_size) + '_' + 'samplingRate' + str(
                resample_Fs) + '.npy'), )
            val_labels_onehot = np.load(os.path.join(dir2save_data, 'val_labels_onehot_' + 'windowSize' + str(
                window_size) + '_' + 'samplingRate' + str(
                resample_Fs) + '.npy'), )
            print('data loaded!')

        if save_data:
            if not os.path.exists(dir2save_data):
                os.mkdir(dir2save_data)
            np.save(os.path.join(dir2save_data,
                                 'train_recordings_' + 'windowSize' + str(window_size) + '_' + 'samplingRate' + str(
                                     resample_Fs) + '.npy'), train_recordings)
            np.save(os.path.join(dir2save_data,
                                 'train_class_weights_' + 'windowSize' + str(window_size) + '_' + 'samplingRate' + str(
                                     resample_Fs) + '.npy'), train_class_weights)
            np.save(os.path.join(dir2save_data,
                                 'train_labels_onehot_' + 'windowSize' + str(window_size) + '_' + 'samplingRate' + str(
                                     resample_Fs) + '.npy'), train_labels_onehot)
            np.save(os.path.join(dir2save_data,
                                 'val_recordings_' + 'windowSize' + str(window_size) + '_' + 'samplingRate' + str(
                                     resample_Fs) + '.npy'), val_recordings)
            np.save(os.path.join(dir2save_data,
                                 'val_class_weights_' + 'windowSize' + str(window_size) + '_' + 'samplingRate' + str(
                                     resample_Fs) + '.npy'), val_class_weights)
            np.save(os.path.join(dir2save_data,
                                 'val_labels_onehot_' + 'windowSize' + str(window_size) + '_' + 'samplingRate' + str(
                                     resample_Fs) + '.npy'), val_labels_onehot)
            print('data saved!!!')

        # Normalization
        if normalization:
            train_recordings = self.normalization(train_recordings)
            val_recordings = self.normalization(val_recordings)

        # nan_files_train_num = 0
        # nan_files_val_num = 0
        # for i in range(len(train_recordings)):
        #     if np.isnan(train_recordings[i]).any() == True:
        #         # tmp = train_recordings[i]
        #         # tmp2 = original_recordings[i]
        #         print('train_recordings', i)
        #         nan_files_train_num += 1
        # for i in range(len(val_recordings)):
        #     if np.isnan(val_recordings[i]).any() == True:
        #         print('val_recordings', i)
        #         nan_files_val_num += 1
        # print("files number with nan value: ", nan_files_train_num, nan_files_val_num)
        # assert np.isnan(train_recordings).any() == False
        # assert np.isnan(val_recordings).any() == False
        train_recordings[np.isnan(train_recordings)] = 0
        val_recordings[np.isnan(val_recordings)] = 0
        assert np.isnan(train_class_weights).any() == False
        assert np.isnan(val_class_weights).any() == False

        X_train = torch.from_numpy(train_recordings).float()
        X_train_class_weight = torch.from_numpy(train_class_weights).float()
        Y_train = torch.from_numpy(train_labels_onehot).float()

        X_val = torch.from_numpy(val_recordings).float()
        X_val_class_weight = torch.from_numpy(val_class_weights).float()
        Y_val = torch.from_numpy(val_labels_onehot).float()

        ### 12 leads order: I II III aVL aVR aVF V1 V2 V3 V4 V5 V6
        if lead_number == 2:  # two leads: I II
            leads_index = [0, 1]
        elif lead_number == 3:  # three leads: I II V2
            leads_index = [0, 1, 7]
        elif lead_number == 4:  # four leads: I II III V2
            leads_index = [0, 1, 2, 7]
        elif lead_number == 6:  # six leads: I II III aVL aVR aVF
            leads_index = [0, 1, 2, 3, 4, 5]
        elif lead_number == 8:  # eight leads
            leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        else:  # twelve leads
            leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        X_train = X_train[:, leads_index, :]
        X_val = X_val[:, leads_index, :]

        self.train_dataset = CustomTensorDataset(X_train, Y_train, X_train_class_weight)
        self.val_dataset = CustomTensorDataset(X_val, Y_val, X_val_class_weight)

        end = time.time()
        print('time to get and process data: {}'.format(end - start))
        super().__init__(self.train_dataset, self.val_dataset, None, batch_size, shuffle, num_workers)

        self.valid_data_loader.file_names = file_names
        self.valid_data_loader.idx = val_index
