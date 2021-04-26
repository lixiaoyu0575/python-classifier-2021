#!/usr/bin/env python

import json
import time
import torch.nn as nn
from model_training.training import *
from tensorboardX import SummaryWriter
import model_training.training as modules

import classifier.inceptiontime as module_arch_inceptiontime
import classifier.resnest as module_arch_resnest
import classifier.swin_transformer_1d as module_arch_swin_transformer

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
    "swin_transformer": ['swin_transformer']
}

my_classes = []
log_step = 1


# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    ### To change


    # split into training and validation
    split_idx = 'model_training/split.mat'
    stratification(data_directory)

    #json files
    training_root = 'model_training/'
    # configs = ['train_6leads.json', 'train_3leads.json', 'train_2leads.json', 'train.json']
    configs = ['train_6leads_swin_transformer.json']

    for config_json_path in configs:
        train_model(training_root + config_json_path, split_idx, data_directory, model_directory)

def train_model(config_json, split_idx, data_directory, model_directory ):
    # Get training configs
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    lead_number = config['lead_number']

    # Data_loader
    train_loader = ChallengeDataLoader(data_directory, split_idx,
                                       batch_size=config['data_loader']['batch_size'],
                                       normalization=config['data_loader']['normalization'],
                                       augmentations=config['data_loader']['augmentation']['method'],
                                       p=config['data_loader']['augmentation']['prob'],
                                       window_size=config['data_loader']['window_size'],
                                       resample_Fs=config['data_loader']['resample_Fs'],
                                       lead_number=lead_number)
    # Paths to save log, checkpoint, tensorboard logs and results
    base_dir = config['base_dir'] + '/training_results'
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

    # Tensorboard
    train_writer = SummaryWriter(tb_dir + '/train_lead_' + str(lead_number))
    val_writer = SummaryWriter(tb_dir + '/valid_' + str(lead_number))

    valid_loader = train_loader.valid_data_loader



    # ### for test
    # config_json = 'model_training/train_2leads.json'
    # with open(config_json, 'r', encoding='utf8')as fp:
    #     config = json.load(fp)
    # checkpoint_path = model_directory + '/lead_12_model_best.pth'
    # model = load_my_model(config, checkpoint_path)

    # Get function handles of loss and metrics
    # criterion = getattr(modules, config['loss']['type'])
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Get function handles of metrics
    challenge_metrics = ChallengeMetric(data_directory)
    metric = challenge_metrics.challenge_metric

    # Get indices of the scored labels
    if config['trainer']['only_scored']:
        indices = challenge_metrics.indices
    else:
        indices = None

    # Build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(config, 'optimizer', torch.optim, trainable_params)

    if config['lr_scheduler']['type'] == 'GradualWarmupScheduler':
        params = config["lr_scheduler"]["args"]
        scheduler_steplr_args = dict(params["after_scheduler"]["args"])
        scheduler_steplr = getattr(torch.optim.lr_scheduler, params["after_scheduler"]["type"])(optimizer,
                                                                                                **scheduler_steplr_args)
        lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=params["multiplier"],
                                              total_epoch=params["total_epoch"], after_scheduler=scheduler_steplr)
    else:
        lr_scheduler = init_obj(config, 'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Begin training process
    trainer = config['trainer']
    epochs = trainer['epochs']

    # Full train and valid logic
    mnt_metric_name, mnt_mode, mnt_best, early_stop = get_mnt_mode(trainer)
    not_improved_count = 0

    for epoch in range(epochs):
        best = False
        train_loss, train_metric = train(model, optimizer, train_loader, criterion, metric, indices, epoch,
                                         device=device)
        val_loss, val_metric = valid(model, valid_loader, criterion, metric, indices, device=device)

        if config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
            lr_scheduler.step(val_loss)
        elif config['lr_scheduler']['type'] == 'GradualWarmupScheduler':
            lr_scheduler.step(epoch, val_loss)
        else:
            lr_scheduler.step()

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
            save_checkpoint(model, epoch, mnt_best, model_directory, file_name, train_loader.classes, save_best=True)
            logger.info("Saving current best: {}".format(file_name))

        # Tensorboard log
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('metric', train_metric, epoch)
        train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        val_writer.add_scalar('loss', val_loss, epoch)
        val_writer.add_scalar('metric', val_metric, epoch)
    del model, train_loader, logger, train_writer, val_writer
################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

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

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    config_json = 'model_training/train.json'
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    checkpoint_path = model_directory + '/lead_' + str(config['lead_number']) + '_model_best.pth'
    model = load_my_model(config, checkpoint_path)
    model.eval()
    return model

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    config_json = 'model_training/train_6leads.json'
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    checkpoint_path = model_directory + '/lead_' + str(config['lead_number']) + '_model_best.pth'
    model = load_my_model(config, checkpoint_path)
    model.eval()
    return model

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    config_json = 'model_training/train_3leads.json'
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    checkpoint_path = model_directory + '/lead_' + str(config['lead_number']) + '_model_best.pth'
    model = load_my_model(config, checkpoint_path)
    model.eval()
    return model

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    config_json = 'model_training/train_2leads.json'
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    checkpoint_path = model_directory + '/lead_' + str(config['lead_number']) + '_model_best.pth'
    model = load_my_model(config, checkpoint_path)
    model.eval()
    return model

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_my_model(model, header, recording, config_path):
    recording[np.isnan(recording)] = 0
    with open(config_path, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    lead_number = config['lead_number']

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
    resample_Fs = config["data_loader"]["resample_Fs"]
    window_size = config["data_loader"]["window_size"]
    header = header.split('\n')
    recording = resample(recording, header, resample_Fs)
    n_segment = 1
    # slide and cut
    recording = slide_and_cut(recording, n_segment, window_size, resample_Fs)
    data = torch.tensor(recording)
    data = torch.reshape(data, (1, 12, window_size))
    data = data.to(device, dtype=torch.float)
    prediction = model(data)
    prediction = torch.reshape(prediction, (108,))
    prediction = torch.sigmoid(prediction)
    prediction = prediction.detach().cpu().numpy()
    label = np.zeros((108,), dtype=int)
    threshold = 0.5
    indexes = np.where(prediction > threshold)
    label[indexes] += 1
    # print(prediction)
    classes = my_classes
    return classes, label, prediction

def run_twelve_lead_model(model, header, recording):
    config_path = 'model_training/train.json'
    return run_my_model(model, header, recording, config_path)


# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    config_path = 'model_training/train_6leads.json'
    return run_my_model(model, header, recording, config_path)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    config_path = 'model_training/train_3leads.json'
    return run_my_model(model, header, recording, config_path)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    config_path = 'model_training/train_2leads.json'
    return run_my_model(model, header, recording, config_path)

# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    num_leads = len(leads)
    data = np.zeros(num_leads+2, dtype=np.float32)
    age, sex, rms = get_features(header, recording, leads)
    data[0:num_leads] = rms
    data[num_leads] = age
    data[num_leads+1] = sex

    # Impute missing data.
    features = data.reshape(1, -1)
    features = imputer.transform(features)

    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = np.asarray(labels, dtype=np.int)[0]

    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
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
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms

def train(model, optimizer, train_loader, criterion, metric, indices, epoch, device=None):
    sigmoid = nn.Sigmoid()
    model.train()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    for batch_idx, (data, target, class_weights) in enumerate(train_loader):
        batch_start = time.time()
        data, target, class_weights = data.to(device), target.to(device), class_weights.to(device)
        optimizer.zero_grad()
        output = model(data)
        if not indices is None:
            loss = criterion(output[:, indices], target[:, indices]) * class_weights[:, indices]
        else:
            loss = criterion(output, target) * class_weights
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        c = metric(to_np(sigmoid(output), device), to_np(target, device))
        cc += c
        Loss += float(loss)
        total += target.size(0)
        batchs += 1

        if batch_idx % log_step == 0:
            batch_end = time.time()
            # logger.debug('Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch, batch_idx, loss.item(),
            #                                                                           batch_end - batch_start))
            print('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch,
                                                                                     progress(train_loader, batch_idx),
                                                                                     loss.item(),
                                                                                     batch_end - batch_start))

    return Loss / total, cc / batchs

def valid(model, valid_loader, criterion, metric, indices, device=None):
    sigmoid = nn.Sigmoid()
    model.eval()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    with torch.no_grad():
        for batch_idx, (data, target, class_weights) in enumerate(valid_loader):
            data, target, class_weights = data.to(device), target.to(device), class_weights.to(device)
            output = model(data)
            if not indices is None:
                loss = criterion(output[:, indices], target[:, indices]) * class_weights[:, indices]
            else:
                loss = criterion(output, target) * class_weights
            loss = torch.mean(loss)
            c = metric(to_np(sigmoid(output), device), to_np(target, device))
            cc += c
            Loss += loss
            total += target.size(0)
            batchs += 1

    return Loss / total, cc / batchs
