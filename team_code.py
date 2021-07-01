#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################

# Import functions. These functions are not required. You can change or remove them.
from helper_code import *
import json
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import torch.nn as nn
from model_training.training import *
from utils.loss import AsymmetricLossOptimized

import utils.lr_scheduler as custom_lr_scheduler
from utils.metric import ChallengeMetric
import classifier.se_resnet as module_arch_se_resnet
from model_training.utils import stratification, make_dirs, init_obj, get_logger, get_mnt_mode, to_np, save_checkpoint

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Setup Cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# model selection
files_models = {
    "inceptiontime": ['InceptionTimeV1', 'InceptionTimeV2'],
    "resnest": ['resnest50', 'resnest'],
    "resnet": ['resnet'],
    "swin_transformer": ['swin_transformer'],
    "beat_aligned_transformer": ['beat_aligned_transformer'],
    "beat_aligned_cnn_transformer": ['beat_aligned_cnn_transformer'],
    "beat_aligned_cnn": ['beat_aligned_cnn'],
    "nesT": ["nesT"],
    "se_resnet": ['se_resnet']
}

my_classes = []
log_step = 1

# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):

    # split into training and validation
    split_idx = 'model_training/split.mat'
    stratification(data_directory)

    #json files
    training_root = 'model_training/'
<<<<<<< HEAD
    configs = ['train_12leads_test.json', 'train_6leads.json', 'train_4leads.json', 'train_3leads.json', 'train_2leads.json']
=======
    configs = ['train_12leads.json', 'train_6leads.json', 'train_4leads.json', 'train_3leads.json', 'train_2leads.json']
>>>>>>> 3b1682dde26aec786854d0c06f69058abb87288c
    # configs = ['train_resnet.json']

    # configs = ['train_beat_aligned_swin_transformer.json']
    # configs = ['train_12leads_nested_transformer.json']

    challenge_dataset = ChallengeDataset(data_directory, split_idx,
                        window_size=5000,
                        resample_Fs=500)
    train_dataset, val_dataset = challenge_dataset.train_dataset, challenge_dataset.val_dataset
    for config_json_path in configs:
        train_model(training_root + config_json_path, split_idx, data_directory, model_directory, train_dataset, val_dataset)
    #
    #
    # # Find header and recording files.
    # print('Finding header and recording files...')
    #
    # header_files, recording_files = find_challenge_files(data_directory)
    # num_recordings = len(recording_files)
    #
    # if not num_recordings:
    #     raise Exception('No data was provided.')
    #
    # # Create a folder for the model if it does not already exist.
    # if not os.path.isdir(model_directory):
    #     os.mkdir(model_directory)

    # # Extract the classes from the dataset.
    # print('Extracting classes...')
    #
    # classes = set()
    # for header_file in header_files:
    #     header = load_header(header_file)
    #     classes |= set(get_labels(header))
    # if all(is_integer(x) for x in classes):
    #     classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    # else:
    #     classes = sorted(classes) # Sort classes alphanumerically if not numbers.
    # num_classes = len(classes)


def train_model(config_json, split_idx, data_directory, model_directory, train_dataset, val_dataset):
    # Get training configs
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    lead_number = config['data_loader']['args']['lead_number']
    assert config['arch']['args']['channel_num'] == lead_number
    # Data_loader
    train_dataset.lead_number = lead_number
    val_dataset.lead_number = lead_number
    print("batch_size: ", config['data_loader']['args']['batch_size'])
    train_loader = ChallengeDataLoader(train_dataset, val_dataset,
                                       batch_size=config['data_loader']['args']['batch_size'])
    if lead_number == 8:
        lead_number = 12
    # Paths to save log, checkpoint, tensorboard logs and results
    base_dir = 'model_training/training_results'
    result_dir, log_dir, checkpoint_dir, tb_dir = make_dirs(base_dir)

    # Build model architecture
    # global model
    for file, types in files_models.items():
        for type in types:
            if config["arch"]["type"] == type:
                model = init_obj(config, 'arch', eval("module_arch_" + file))
    model.to(device)
    # Logger for train
    logger = get_logger(log_dir + '/info_lead_' + str(lead_number) + '.log', name='train')
    logger.info(config["arch"]["type"])
    # Tensorboard
    # train_writer = SummaryWriter(tb_dir + '/train_lead_' + str(lead_number))
    # val_writer = SummaryWriter(tb_dir + '/valid_' + str(lead_number))

    valid_loader = train_loader.valid_data_loader


    header_files = my_find_challenge_files(data_directory)
    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x))  # Sort classes numerically if numbers.
    else:
        classes = sorted(classes)  # Sort classes alphanumerically if not numbers.
    num_classes = len(classes)
    train_loader.all_classes = classes


    # ### for test
    # config_json = 'model_training/train_2leads.json'
    # with open(config_json, 'r', encoding='utf8')as fp:
    #     config = json.load(fp)
    # checkpoint_path = model_directory + '/lead_12_model_best.pth'
    # model = load_my_model(config, checkpoint_path)

    # Get function handles of loss and metrics
    # criterion = getattr(modules, config['loss']['type'])
    criterion = AsymmetricLossOptimized()
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Get function handles of metrics
    metric = ChallengeMetric()

    # Build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = init_obj(config, 'optimizer', torch.optim, trainable_params)

    lr_scheduler = init_obj(config, 'lr_scheduler', custom_lr_scheduler, optimizer)

    # Begin training process
    trainer = config['trainer']
    epochs = trainer['epochs']
    # epochs = 1

    # Full train and valid logic
    mnt_metric_name, mnt_mode, mnt_best, early_stop = get_mnt_mode(trainer)
    not_improved_count = 0

    for epoch in range(epochs):
        best = False
        train_loss, train_metric = train(model, optimizer, train_loader, criterion, metric, epoch,
                                         device=device)
        val_loss, val_metric = valid(model, valid_loader, criterion, metric, device=device)

        lr_scheduler.step(epoch)

        logger.info(
            'Epoch:[{}/{}]\t {:10s}: {:.5f}\t {:10s}: {:.5f}'.format(epoch, epochs, 'loss', train_loss, 'metric',
                                                                     train_metric))
        logger.info(
            '             \t {:10s}: {:.5f}\t {:10s}: {:.5f}'.format('val_loss', val_loss, 'val_metric', val_metric))
        logger.info('             \t learning_rate: {}'.format(optimizer.param_groups[0]['lr']))

        # check whether model performance improved or not, according to specified metric(mnt_metric)
        if mnt_mode != 'off':
            mnt_metric = val_loss if mnt_metric_name == 'val_loss' else val_metric
            improved = (mnt_mode == 'min' and mnt_metric <= mnt_best) or \
                       (mnt_mode == 'max' and mnt_metric >= mnt_best)
            if improved:
                mnt_best = mnt_metric
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > early_stop:
                logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(early_stop))
                break
        file_name = 'lead_' + str(lead_number) + '_model_best.pth'
        # save_checkpoint(model, epoch, mnt_best, checkpoint_dir, file_name, save_best=False)
        if best == True:
            save_checkpoint(model, epoch, mnt_best, model_directory, file_name, train_loader.all_classes, leads_num=lead_number, config_json=config_json, save_best=True)
            logger.info("Saving current best: {}".format(file_name))

        # Tensorboard log
        # train_writer.add_scalar('loss', train_loss, epoch)
        # train_writer.add_scalar('metric', train_metric, epoch)
        # train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        #
        # val_writer.add_scalar('loss', val_loss, epoch)
        # val_writer.add_scalar('metric', val_metric, epoch)
    del model, train_loader, logger

################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.


def run_my_model(model, header, recording, config_path):
    recording[np.isnan(recording)] = 0
    recording = np.array(recording, dtype=float)
    with open(config_path, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    # lead_number = config['lead_number']

    # ### to get recording in shape [12, ?]
    # recording_tmp = np.zeros((12, recording.shape[1]))
    # leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # if lead_number == 2:
    #     # two leads
    #     leads_index = [1, 11]
    # elif lead_number == 3:
    #     # three leads
    #     leads_index = [0, 1, 7]
    # elif lead_number == 6:
    #     # six leads
    #     leads_index = [0, 1, 2, 3, 4, 5]
    # recording_tmp[leads_index] = recording
    # recording = recording_tmp

    # divide ADC_gain and resample
    resample_Fs = config["data_loader"]['args']["resample_Fs"]
    window_size = config["data_loader"]['args']["window_size"]
    header = header.split('\n')
    recording = resample(recording, header, resample_Fs)

    # to filter and detrend samples
    recording = filter_and_detrend(recording)

    n_segment = 1
    # slide and cut
    recording = slide_and_cut(recording, n_segment, window_size, resample_Fs, test_time_aug=True)

    if config["arch"]["args"]["channel_num"] == 8:
        leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        recording = recording[:, leads_index, :]
    recording[np.isnan(recording)] = 0
    data = torch.tensor(recording).to(device)
    data = data.to(device, dtype=torch.float)
    output = model(data)
    prediction = torch.sigmoid(output)
    prediction = prediction.detach().cpu().numpy()
    prediction = np.mean(prediction, axis=0)

    classes = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
    ### equivalent SNOMED CT codes merged, noted as the larger one
    classes = classes.split(',')
    all_classes = my_classes

    label = np.zeros((26,), dtype=int)

    threshold = 0.5
    indexes = np.where(prediction > threshold)
    label[indexes] += 1

    label_output = np.zeros((len(all_classes),), dtype=int)
    prediction_output = np.zeros((len(all_classes),))


    equivalent_classes = {
        "733534002": "164909002",
        "713427006": "59118001",
        "63593006": "284470004",
        "427172004": "17338001"
    }
    for i in range(len(classes)):
        dx = classes[i]
        ind = all_classes.index(dx)
        label_output[ind] = label[i]
        prediction_output[ind] = prediction[i]
        if dx == "733534002" or dx == "713427006" or dx == "63593006" or dx == "427172004":
            dx2 = equivalent_classes[dx]
            ind = all_classes.index(dx2)
            label_output[ind] = label[i]
            prediction_output[ind] = prediction[i]
        if dx == "733534002" or dx == "713427006":
            ind = all_classes.index("6374002")
            if label[i] == 1:
                label_output[ind] = label[i]
                prediction_output[ind] = prediction[i]
    #
    # data = torch.tensor(recording)
    # data = torch.reshape(data, (1, 12, window_size))
    # data = data.to(device, dtype=torch.float)
    # prediction = model(data)
    # prediction = torch.reshape(prediction, (108,))
    # prediction = torch.sigmoid(prediction)
    # prediction = prediction.detach().cpu().numpy()
    # label = np.zeros((108,), dtype=int)
    # threshold = 0.5
    # indexes = np.where(prediction > threshold)
    # label[indexes] += 1
    # # print(prediction)
    # classes = my_classes
    return all_classes, label_output, prediction_output

################################################################################
#
# File I/O functions
#
################################################################################

# Save a trained model. This function is not required. You can change or remove it.
def save_model(model_directory, leads, classes, imputer, classifier):
    d = {'leads': leads, 'classes': classes, 'imputer': imputer, 'classifier': classifier}
    filename = os.path.join(model_directory, get_model_filename(leads))
    joblib.dump(d, filename, protocol=0)

# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    leads_num_to_configs = {
        "2": "model_training/train_2leads.json",
        "3": "model_training/train_3leads.json",
        "4": "model_training/train_4leads.json",
        "6": "model_training/train_6leads.json",
        "12": "model_training/train_12leads.json"
    }
    leads_num = len(leads)
    print("current leads_num: ", leads)
    config_json = leads_num_to_configs[str(leads_num)]
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    global current_config_json
    current_config_json = config_json

    checkpoint_path = model_directory + '/lead_' + str(leads_num) + '_model_best.pth'
    model = load_my_model(config, checkpoint_path)
    model.eval()
    return model

def load_my_model(config, checkpoint_path=None):

    for file, types in files_models.items():
        for type in types:
            if config["arch"]["type"] == type:
                model = init_obj(config, 'arch', eval("module_arch_" + file))

    model.to(device)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        global my_classes
        my_classes = checkpoint["classes"]

    return model
# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_' + '-'.join(sorted_leads) + '.sav'

################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    recording = choose_leads(recording, header, leads)

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms
def train(model, optimizer, train_loader, criterion, metric, epoch, device=None):
    sigmoid = nn.Sigmoid()
    model.train()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    for batch_idx, (data, target, class_weights) in enumerate(train_loader):
        batch_start = time.time()
        data, target, class_weights = data.to(device), target.to(device), class_weights.to(device)
        # target_coarse = target_coarse.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target) * class_weights
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        prediction = to_np(sigmoid(output), device)
        prediction = metric.get_pred(prediction, alpha=0.5)
        target = to_np(target, device)
        c = metric.challenge_metric(prediction, target)
        cc += c
        Loss += float(loss)
        total += 1
        batchs += 1

        if batch_idx % log_step == 0:
            batch_end = time.time()
            # logger.debug('Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch, batch_idx, loss.item(),
            #                                                                           batch_end - batch_start))
            print('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch,
                                                                                     batch_idx,
                                                                                     loss.item(),
                                                                                     batch_end - batch_start))

    return Loss / total, cc / batchs

def valid(model, valid_loader, criterion, metric, device=None):
    sigmoid = nn.Sigmoid()
    model.eval()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    with torch.no_grad():
        for batch_idx, (data, target, class_weights) in enumerate(valid_loader):
            data, target, class_weights = data.to(device), target.to(device), class_weights.to(device)
            # target_coarse = target_coarse.to(device)
            output = model(data)

            loss = criterion(output, target) * class_weights
            loss = torch.mean(loss)
            # loss = (loss_coarse + loss) / 2

            prediction = to_np(sigmoid(output), device)
            prediction = metric.get_pred(prediction, alpha=0.5)
            target = to_np(target, device)
            c = metric.challenge_metric(prediction, target)
            cc += c
            Loss += loss
            total += 1
            batchs += 1

    return Loss / total, cc / batchs

def run_model(model, header, recording):
    # classes = model['classes']
    # leads = model['leads']
    config_json = current_config_json
    return run_my_model(model, header, recording, config_json)