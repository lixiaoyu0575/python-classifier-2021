from torch.utils import data
import torch
import numpy as np
from model_training.utils import loadmat, CustomTensorDataset, load_weights, load_labels, resample, slide_and_cut, load_challenge_data
from utils.denoising import filter_and_detrend
import neurokit2 as nk
import os

# from process.extract_peak_targets import get_target_peaks as get_target_peak_bak

class CustomDataset(data.Dataset):
    """
    PyTorch Dataset generator class
    Ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, label_files, labels_onehot, label_dir, leads_index, name_list_full=[], transform=None, sample_rate=500, to_get_feature=False):
        """Initialization"""
        self.file_names_list = label_files
        self.label_dir = label_dir
        self.labels_onehot = labels_onehot
        self.class_weights = self.get_class_weights()
        self.leads_index = leads_index
        self.transform = transform
        self.to_get_feature = to_get_feature
        self.sample_rate = sample_rate
        # self.normalization = TNormalize()


    def __len__(self):
        """Return total number of data samples"""
        return len(self.file_names_list)

    def __getitem__(self, idx):
        """Generate data sample"""
        sample_file_name = self.file_names_list[idx]
        # header_file_name = self.file_names_list[idx][:-3] + "hea"

        label = self.labels_onehot[idx]
        recording, header, name = load_challenge_data(sample_file_name, self.label_dir)

        # get class_weight by name
        class_weight, data_source = self.get_class_weight_and_source_by_name(name)

        # divide ADC_gain and resample
        recording = resample(recording, header, resample_Fs=self.sample_rate)
        for lead in recording:
            assert np.isnan(lead).any() == False
        #     if lead.sum() == 0:
        #         print(idx)
        # to extract features
        # recording = self.normalization(recording)
        feature = np.zeros((50,))
        if self.to_get_feature:
            feature = self.get_features(recording)

        feature = torch.tensor(feature)
        if self.transform is not None:
            recording = self.transform(recording)

        recording = filter_and_detrend(recording)
        # recording = nk.ecg_clean(recording, method='biosppy')


        # # slide and cut
        # recording = slide_and_cut(recording, n_segment=1, window_size=4992, sampling_rate=500)
        # recording = recording[0]
        # to filter and detrend samples

        recording = recording[self.leads_index, :]
        recording = torch.tensor(recording)
        label = torch.tensor(label)
        class_weight = torch.tensor(class_weight)
        data_source = torch.tensor(data_source)

        if self.to_get_feature:
            return recording, label, class_weight, data_source, feature

        return recording, label, class_weight

    def get_class_weights(self):
        classes = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
        ### equivalent SNOMED CT codes merged, noted as the larger one
        classes = classes.split(',')
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
        # PTB_XL_class_weight[classes.index('426783006')] = 0.1
        # G12ECG
        G12ECG_excluded_classes = ['10370003', '365413008', '164947007']
        G12ECG_class_weight = np.ones((26,))
        for cla in G12ECG_excluded_classes:
            G12ECG_class_weight[classes.index(cla)] = 0
        # Chapman Shaoxing
        Chapman_excluded_classes = ['6374002', '426627000', '713426002', '445118002', '10370003', '365413008',
                                    '427172004', '427393009', '63593006']
        Chapman_class_weight = np.ones((26,))
        for cla in Chapman_excluded_classes:
            Chapman_class_weight[classes.index(cla)] = 0
        # Ningbo
        Ningbo_excluded_classes = ['164889003', '164890007', '426627000']
        Ningbo_class_weight = np.ones((26,))
        for cla in Ningbo_excluded_classes:
            Ningbo_class_weight[classes.index(cla)] = 0
        return [CPSC_extra_class_weight, CPSC_extra_class_weight, PTB_XL_class_weight, G12ECG_class_weight, Chapman_class_weight, Ningbo_class_weight]
    def get_class_weight_and_source_by_name(self, name):
        if name[0] == 'A':  # CPSC
            class_weight = self.class_weights[0]
            data_source_class = 0
        elif name[0] == 'Q':  # CPSC-extra
            class_weight = self.class_weights[1]
            data_source_class = 2
        elif name[0] == 'H':  # PTB-XL
            class_weight = self.class_weights[2]
            data_source_class = 0
        elif name[0] == 'E':  # G12ECG
            class_weight = self.class_weights[3]
            data_source_class = 1
        elif name[0] == 'J' and int(name[2:]) <= 10646:  # Chapman
            class_weight = self.class_weights[4]
            data_source_class = 2
        elif name[0] == 'J' and int(name[2:]) > 10646:  # Ningbo
            class_weight = self.class_weights[5]
            data_source_class = 2
        elif name[0] == 'S' or name[0] == 'I':  # Ningbo
            class_weight = np.zeros((26,))
            data_source_class = 2
        return class_weight, data_source_class

class CustomDataset4PeakDetection(CustomDataset):
    """
    PyTorch Dataset generator class
    Ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """
    def __init__(self, label_files, labels_onehot, label_dir, leads_index, name_list_full=[], sample_rate=500, transform=None, to_get_feature=False):
        super().__init__(label_files, labels_onehot, label_dir, leads_index, name_list_full=name_list_full, sample_rate=sample_rate, transform=transform, to_get_feature=to_get_feature)

        name_list_full_path = './name_list_all.npy'
        self.name_list_full = np.load(name_list_full_path)
        # load peak targets
        if os.path.exists('./peak_targets_full_v2.npy') == False:
            os.system('gunzip ./peak_targets_full_v2.npy.gz')
        self.peak_targets_full = np.load('./peak_targets_full_v2.npy')
        print("get peak target")
    def __getitem__(self, idx):
        """Generate data sample"""
        sample_file_name = self.file_names_list[idx]
        # header_file_name = self.file_names_list[idx][:-3] + "hea"

        label = self.labels_onehot[idx]
        recording, header, name = load_challenge_data(sample_file_name, self.label_dir)

        # get class_weight by name
        class_weight, data_source = self.get_class_weight_and_source_by_name(name)

        # divide ADC_gain and resample
        recording = resample(recording, header, resample_Fs=self.sample_rate)
        for lead in recording:
            assert np.isnan(lead).any() == False
        #     if lead.sum() == 0:
        #         print(idx)
        # to extract features
        # recording = self.normalization(recording)

        if self.transform is not None:
            recording = self.transform(recording)
        assert len(recording) <= 4992
        recording = filter_and_detrend(recording)
        # recording = nk.ecg_clean(recording, method='biosppy')

        # target_peaks, _ = get_target_peak_bak(recording, type_num=5)
        # target_peaks = self.get_target_peaks(recording)
        target_peaks = self.peak_targets_full[np.where(self.name_list_full==name)]

        recording = recording[self.leads_index, :]
        recording = torch.tensor(recording)
        label = torch.tensor(label)
        # class_weight = torch.tensor([0])
        # data_source = torch.tensor([0])
        target_peaks = torch.tensor(target_peaks[0])
        # print(recording.size(), label.size(), class_weight.size(), target_peaks.size())
        return recording, label, class_weight, target_peaks