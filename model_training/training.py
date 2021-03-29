import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model_training.utils import *
import model_training.augmentation as module_augmentation

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Challenge Dataloaders and Challenge metircs

# Base Dataloader
class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, train_dataset, val_dataset, batch_size, shuffle, num_workers,
                 collate_fn=default_collate, pin_memory=True):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        super().__init__(**self.init_kwargs)

        self.valid_data_loader_init_kwargs = {
            'dataset': self.val_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        self.valid_data_loader = DataLoader(**self.valid_data_loader_init_kwargs)
        self.n_samples = len(self.train_dataset)
        self.valid_data_loader.n_samples = len(self.val_dataset)

# DataLoader (augmentation + 25 classes)
class ChallengeDataLoader(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, split_index, batch_size, shuffle=True, num_workers=0, resample_Fs=300, window_size=3000, n_segment=1, normalization=False, augmentations=None, p=0.5, _25classes=False, lead_number=12):
        self.label_dir = label_dir
        print('Loading data...')

        weights_file = 'model_training/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(label_dir)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)
        self.classes = classes

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)
        self.weights = weights

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        indices_unscored = ~indices

        # Get number of samples for each category
        self.indices = indices

        split_idx = loadmat(split_index)
        train_index, val_index = split_idx['train_index'], split_idx['val_index']
        train_index = train_index.reshape((train_index.shape[1],))
        val_index = val_index.reshape((val_index.shape[1],))

        num_files = len(label_files)
        train_recordings = list()
        train_class_weights = list()
        train_labels_onehot = list()

        val_recordings = list()
        val_class_weights = list()
        val_labels_onehot = list()

        file_names = list()

        ### class for dataset
        #equivalent diagnose [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
        #CPSC
        CPSC_classes = ['270492004', '164889003', '164909002', '284470004', '426783006', '713427006'] #"59118001" = "713427006"
        CPSC_class_weight = np.zeros((108,))
        for cla in CPSC_classes:
            CPSC_class_weight[classes.index(cla)] = 1
        #CPSC_extra
        CPSC_extra_excluded_classes = ['445118002', '39732003', '251146004', '698252002', '10370003', '427172004', '164947007', '111975006', '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '59931005', '17338001']
        CPSC_extra_class_weight = np.ones((108,))
        for cla in CPSC_extra_excluded_classes:
            CPSC_extra_class_weight[classes.index(cla)] = 0
        #PTB-XL
        PTB_XL_excluded_classes = ['426627000', '427172004'] #, '17338001'
        PTB_XL_class_weight = np.ones((108,))
        for cla in PTB_XL_excluded_classes:
            PTB_XL_class_weight[classes.index(cla)] = 0
        #G12ECG
        G12ECG_excluded_classes = ['10370003', '427172004', '164947007', '426627000', '10370003', '63593006']
        G12ECG_class_weight = np.ones((108,))
        for cla in G12ECG_excluded_classes:
            G12ECG_class_weight[classes.index(cla)] = 0


        for i in range(num_files):
            print('{}/{}'.format(i+1, num_files))
            recording, header, name = load_challenge_data(label_files[i], label_dir)
            if name[0] == 'S' or name[0] == 'I': # PTB or St.P dataset
                continue
            elif name[0] == 'A': # CPSC
                class_weight = CPSC_class_weight
            elif name[0] == 'Q': # CPSC-extra
                class_weight = CPSC_extra_class_weight
            elif name[0] == 'H': # PTB-XL
                class_weight = PTB_XL_class_weight
            elif name[0] == 'E': # G12ECG
                class_weight = G12ECG_class_weight
            else:
                print('warning! not from one of the datasets')
                print(name)

            recording[np.isnan(recording)] = 0

            # divide ADC_gain and resample
            recording = resample(recording, header, resample_Fs)

            # slide and cut
            recording = slide_and_cut(recording, n_segment, window_size, resample_Fs)

            file_names.append(name)
            if i in train_index or name[0] == 'A' or name[0] == 'Q':
                for j in range(recording.shape[0]):
                    train_recordings.append(recording[j])
                    if _25classes:
                        label = np.ones((25, )).astype(bool)
                        label[:24] = labels_onehot[i, indices]
                        label[24] = labels_onehot[i, indices_unscored].any()
                        train_labels_onehot.append(label)
                        train_class_weights.append(class_weight)
                    else:
                        train_labels_onehot.append(labels_onehot[i])
                        train_class_weights.append(class_weight)
            elif i in val_index:
                for j in range(recording.shape[0]):
                    val_recordings.append(recording[j])
                    if _25classes:
                        label = np.ones((25, )).astype(bool)
                        label[:24] = labels_onehot[i, indices]
                        label[24] = labels_onehot[i, indices_unscored].any()
                        val_labels_onehot.append(label)
                        val_class_weights.append(class_weight)
                    else:
                        val_labels_onehot.append(labels_onehot[i])
                        val_class_weights.append(class_weight)
                else:
                    pass

        print(np.isnan(train_recordings).any())
        print(np.isnan(val_recordings).any())

        train_recordings = np.array(train_recordings)
        train_class_weights = np.array(train_class_weights)
        train_labels_onehot = np.array(train_labels_onehot)

        val_recordings = np.array(val_recordings)
        val_class_weights = np.array(val_class_weights)
        val_labels_onehot = np.array(val_labels_onehot)

        # Normalization
        if normalization:
            train_recordings = self.normalization(train_recordings)
            val_recordings = self.normalization(val_recordings)

        X_train = torch.from_numpy(train_recordings).float()
        X_train_class_weight = torch.from_numpy(train_class_weights).float()
        Y_train = torch.from_numpy(train_labels_onehot).float()

        X_val = torch.from_numpy(val_recordings).float()
        X_val_class_weight = torch.from_numpy(val_class_weights).float()
        Y_val = torch.from_numpy(val_labels_onehot).float()

        X_train_tmp = torch.zeros_like(X_train)
        X_val_tmp = torch.zeros_like(X_val)
        leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        if lead_number == 2:
            # two leads
            leads_index = [1, 11]
        elif lead_number == 3:
            # three leads
            leads_index = [0, 1, 7]
        elif lead_number == 6:
            # six leads
            leads_index = [0, 1, 2, 3, 4, 5]

        X_train_tmp[:, leads_index, :] = X_train[:, leads_index, :]
        X_val_tmp[:, leads_index, :] = X_val[:, leads_index, :]
        X_train = X_train_tmp
        X_val = X_val_tmp

        if augmentations:
            transformers = list()

            for key, value in augmentations.items():
                module_args = dict(value['args'])
                transformers.append(getattr(module_augmentation, key)(**module_args))

            train_transform = transforms.Compose(transformers)
            self.train_dataset = CustomTensorDataset(X_train, Y_train, X_train_class_weight, transform=train_transform, p=p)
        else:
            self.train_dataset = CustomTensorDataset(X_train, Y_train, X_train_class_weight)

        self.val_dataset = CustomTensorDataset(X_val, Y_val, X_val_class_weight)
        super().__init__(self.train_dataset, self.val_dataset, batch_size, shuffle, num_workers)

        self.valid_data_loader.file_names = file_names
        self.valid_data_loader.idx = val_index


    def normalization(self, X):
        mm = MinMaxScaler()
        for i in range(len(X)):
            data = X[i].swapaxes(0, 1)
            data_scaled = mm.fit_transform(data)
            data_scaled = data_scaled.swapaxes(0, 1)
            X[i] = data_scaled
        return X

# Challenge2020 official evaluation
class ChallengeMetric():

    def __init__(self, input_directory, _25classes=False):

        # challengeMetric initialization
        weights_file = 'model_training/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(input_directory)

        # Load the labels and classes.
        print('Loading labels...')
        classes, _, _ = load_labels(label_files, normal_class, equivalent_classes)

        num_files = len(label_files)
        print("num_files:", num_files)

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)

        # Only consider classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        classes = [x for i, x in enumerate(classes) if indices[i]]
        weights = weights[np.ix_(indices, indices)]

        self.weights = weights
        self.classes = classes
        self.normal_class = normal_class
        self._return_metric_list = False

        if _25classes:
            indices_25 = np.ones((25, ))
            indices_25[24] = 0
            self.indices = indices_25.astype(bool)
        else:
            self.indices = indices

    def return_metric_list(self):
        self._return_metric_list = True

    # Compute recording-wise accuracy.
    def accuracy(self, outputs, labels):
        outputs = self.get_pred(outputs)
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]

        num_recordings, num_classes = np.shape(labels)

        num_correct_recordings = 0
        for i in range(num_recordings):
            if np.all(labels[i, :] == outputs[i, :]):
                num_correct_recordings += 1

        return float(num_correct_recordings) / float(num_recordings)

    # Compute confusion matrices.
    def confusion_matrices(self, outputs, labels, normalize=False):
        # Compute a binary confusion matrix for each class k:
        #
        #     [TN_k FN_k]
        #     [FP_k TP_k]
        #
        # If the normalize variable is set to true, then normalize the contributions
        # to the confusion matrix by the number of labels per recording.
        num_recordings, num_classes = np.shape(labels)

        if not normalize:
            A = np.zeros((num_classes, 2, 2))
            for i in range(num_recordings):
                for j in range(num_classes):
                    if labels[i, j] >= 0.5 and outputs[i, j] >= 0.5:  # TP
                        A[j, 1, 1] += 1
                    elif labels[i, j] < 0.5 and outputs[i, j] >= 0.5:  # FP
                        A[j, 1, 0] += 1
                    elif labels[i, j] >= 0.5 and outputs[i, j] < 0.5:  # FN
                        A[j, 0, 1] += 1
                    elif labels[i, j] < 0.5 and outputs[i, j] < 0.5:  # TN
                        A[j, 0, 0] += 1
                    else:  # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')
        else:
            A = np.zeros((num_classes, 2, 2))
            for i in range(num_recordings):
                normalization = float(max(np.sum(labels[i, :]), 1))
                for j in range(num_classes):
                    if labels[i, j] >= 0.5 and outputs[i, j] >= 0.5:  # TP
                        A[j, 1, 1] += 1.0 / normalization
                    elif labels[i, j] < 0.5 and outputs[i, j] >= 0.5:  # FP
                        A[j, 1, 0] += 1.0 / normalization
                    elif labels[i, j] >= 0.5 and outputs[i, j] < 0.5:  # FN
                        A[j, 0, 1] += 1.0 / normalization
                    elif labels[i, j] < 0.5 and outputs[i, j] < 0.5:  # TN
                        A[j, 0, 0] += 1.0 / normalization
                    else:  # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')

        return A

    # Compute macro F-measure.
    def f_measure(self, outputs, labels):
        outputs = self.get_pred(outputs)
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        num_recordings, num_classes = np.shape(labels)

        A = self.confusion_matrices(outputs, labels)

        f_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if 2 * tp + fp + fn:
                f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
            else:
                f_measure[k] = float('nan')

        macro_f_measure = np.nanmean(f_measure)

        if self._return_metric_list:
            return macro_f_measure, f_measure
        else:
            return macro_f_measure

    # Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
    def macro_f_beta_measure(self, outputs, labels, beta=2):
        macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = self.beta_measures(outputs, labels, beta)
        if self._return_metric_list:
            return macro_f_beta_measure, f_beta_measure
        else:
            return macro_f_beta_measure

    def macro_g_beta_measure(self, outputs, labels, beta=2):
        macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = self.beta_measures(outputs, labels,
                                                                                                        beta)
        if self._return_metric_list:
            return macro_g_beta_measure, g_beta_measure
        else:
            return macro_g_beta_measure

    def beta_measures(self, outputs, labels, beta=2):
        outputs = self.get_pred(outputs)
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        num_recordings, num_classes = np.shape(labels)

        A = self.confusion_matrices(outputs, labels, normalize=True)

        f_beta_measure = np.zeros(num_classes)
        g_beta_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if (1 + beta ** 2) * tp + fp + beta ** 2 * fn:
                f_beta_measure[k] = float((1 + beta ** 2) * tp) / float((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
            else:
                f_beta_measure[k] = float('nan')
            if tp + fp + beta * fn:
                g_beta_measure[k] = float(tp) / float(tp + fp + beta * fn)
            else:
                g_beta_measure[k] = float('nan')

        macro_f_beta_measure = np.nanmean(f_beta_measure)
        macro_g_beta_measure = np.nanmean(g_beta_measure)

        return macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure

    # Compute macro AUROC and macro AUPRC.
    def macro_auroc(self, outputs, labels):
        macro_auroc, macro_auprc, auroc, auprc = self.auc(outputs, labels)
        if self._return_metric_list:
            return macro_auroc, auroc
        else:
            return macro_auroc

    def macro_auprc(self, outputs, labels):
        macro_auroc, macro_auprc, auroc, auprc = self.auc(outputs, labels)
        if self._return_metric_list:
            return macro_auprc, auprc
        else:
            return macro_auprc

    def auc(self, outputs, labels):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        num_recordings, num_classes = np.shape(labels)

        # Compute and summarize the confusion matrices for each class across at distinct output values.
        auroc = np.zeros(num_classes)
        auprc = np.zeros(num_classes)

        for k in range(num_classes):
            # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
            thresholds = np.unique(outputs[:, k])
            thresholds = np.append(thresholds, thresholds[-1] + 1)
            thresholds = thresholds[::-1]
            num_thresholds = len(thresholds)

            # Initialize the TPs, FPs, FNs, and TNs.
            tp = np.zeros(num_thresholds)
            fp = np.zeros(num_thresholds)
            fn = np.zeros(num_thresholds)
            tn = np.zeros(num_thresholds)
            fn[0] = np.sum(labels[:, k] >= 0.5)
            tn[0] = np.sum(labels[:, k] < 0.5)

            # Find the indices that result in sorted output values.
            idx = np.argsort(outputs[:, k])[::-1]

            # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
            i = 0
            for j in range(1, num_thresholds):
                # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
                tp[j] = tp[j - 1]
                fp[j] = fp[j - 1]
                fn[j] = fn[j - 1]
                tn[j] = tn[j - 1]

                # Update the TPs, FPs, FNs, and TNs at i-th output value.
                while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                    if labels[idx[i], k]:
                        tp[j] += 1
                        fn[j] -= 1
                    else:
                        fp[j] += 1
                        tn[j] -= 1
                    i += 1

            # Summarize the TPs, FPs, FNs, and TNs for class k.
            tpr = np.zeros(num_thresholds)
            tnr = np.zeros(num_thresholds)
            ppv = np.zeros(num_thresholds)
            npv = np.zeros(num_thresholds)

            for j in range(num_thresholds):
                if tp[j] + fn[j]:
                    tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
                else:
                    tpr[j] = float('nan')
                if fp[j] + tn[j]:
                    tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
                else:
                    tnr[j] = float('nan')
                if tp[j] + fp[j]:
                    ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
                else:
                    ppv[j] = float('nan')

            # Compute AUROC as the area under a piecewise linear function with TPR/
            # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
            # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
            # (y-axis) for class k.
            for j in range(num_thresholds - 1):
                auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
                auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

        # Compute macro AUROC and macro AUPRC across classes.
        macro_auroc = np.nanmean(auroc)
        macro_auprc = np.nanmean(auprc)

        return macro_auroc, macro_auprc, auroc, auprc

    # Compute modified confusion matrix for multi-class, multi-label tasks.
    def modified_confusion_matrix(self, outputs, labels):
        # Compute a binary multi-class, multi-label confusion matrix, where the rows
        # are the labels and the columns are the outputs.
        num_recordings, num_classes = np.shape(labels)
        A = np.zeros((num_classes, num_classes))

        # Iterate over all of the recordings.
        for i in range(num_recordings):
            # Calculate the number of positive labels and/or outputs.
            normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
            # Iterate over all of the classes.
            for j in range(num_classes):
                # Assign full and/or partial credit for each positive class.
                if labels[i, j] > 0.5:
                    for k in range(num_classes):
                        if outputs[i, k] > 0.5:
                            A[j, k] += 1.0 / normalization

        return A

    # Compute the evaluation metric for the Challenge.
    def challenge_metric(self, outputs, labels):

        outputs = self.get_pred(outputs)
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]

        num_recordings, num_classes = np.shape(labels)
        normal_index = self.classes.index(self.normal_class)

        # Compute the observed score.
        A = self.modified_confusion_matrix(outputs, labels)
        observed_score = np.nansum(self.weights * A)

        # Compute the score for the model that always chooses the correct label(s).
        correct_outputs = labels
        A = self.modified_confusion_matrix(labels, correct_outputs)
        correct_score = np.nansum(self.weights * A)

        # Compute the score for the model that always chooses the normal class.
        inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
        inactive_outputs[:, normal_index] = 1
        A = self.modified_confusion_matrix(labels, inactive_outputs)
        inactive_score = np.nansum(self.weights * A)

        if correct_score != inactive_score:
            normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
        else:
            normalized_score = float('nan')

        return normalized_score

    def get_pred(self, outputs, alpha=0.5):
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
                if outputs[i, j] >= alpha:
                    outputs[i, j] = 1
                else:
                    outputs[i, j] = 0
        return outputs

# Loss function
def bce_with_logits_loss(output, target):
    loss = torch.nn.BCEWithLogitsLoss()
    # print(output.size(),target.size())
    return loss(output, target)


# Optimizer

# Lr_scheduler
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
