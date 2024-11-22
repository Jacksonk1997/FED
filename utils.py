import os
import sys
import json
import shutil
import datetime

import logging
from logging.handlers import TimedRotatingFileHandler
import pandas as pd
import matplotlib.pyplot as plt
# set font
# plt.rcParams['font.family'] = "serif"
# plt.rcParams['text.usetex'] = False
# set grid bottom
plt.rcParams['axes.axisbelow'] = True

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.utils import plot_model


def custom_mkdir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
        os.chmod(filepath, 0o777)


def inverse_groundtruth(dpr):
    """
    :param dpr: DataProcessor
    :return:
    """
    inverse_train_y = dpr.scaler_fit.inverse_transform(dpr.normed_df_train[1])
    inverse_valid_y = dpr.scaler_fit.inverse_transform(dpr.normed_df_valid[1])
    inverse_test_y  = dpr.scaler_fit.inverse_transform(dpr.normed_df_test[1])

    return inverse_train_y, inverse_valid_y, inverse_test_y


def inverse_prediction(dpr, train_pred, valid_pred, test_pred):
    inverse_train_pred = dpr.scaler_fit.inverse_transform(train_pred)
    inverse_valid_pred = dpr.scaler_fit.inverse_transform(valid_pred)
    inverse_test_pred = dpr.scaler_fit.inverse_transform(test_pred)

    return inverse_train_pred, inverse_valid_pred, inverse_test_pred


def plot_history(args,
                 history,
                 title=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
                 figsize=(8, 6), dpi=600,
                 dict_keys=None,
                 ):
    assert (history is not None and type(history) is dict)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if dict_keys is not None and type(dict_keys) is list:
        # if we do not want to plot everything
        pd.DataFrame({key: history[key] for key in dict_keys}).plot(ax=plt.gca())
    else:
        pd.DataFrame(history).plot(ax=plt.gca())

    plt.grid(linestyle='--', alpha=0.5)

    if xlim is not None:
        plt.gca().set_xlim(xlim)

    if ylim is not None:
        plt.gca().set_ylim(ylim)

    plt.title(label=title, fontsize=14)
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    current_path = os.path.dirname(os.path.realpath(__file__))
    custom_mkdir(current_path + '\\plot_figures\\')
    plt.savefig(current_path + '\\plot_figures\\history.png', dpi=dpi)

    plt.show()


def save_history(log, args, history):
    current_path = os.path.dirname(os.path.realpath(__file__))
    try:
        log.info('Saving train history ...')
    except NameError:
        pass

    filepath = current_path + '\\train_history\\'
    custom_mkdir(filepath)

    dt = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M")
    filename = filepath + f'train_history-{str(dt)}.csv'

    hist_df = pd.DataFrame(history)

    if os.path.exists(filename):
        try:
            log.warning('File %s already exists. Backing it up to %s', filename, filename + f'{str(dt)}.csv')
        except NameError:
            pass
        shutil.copyfile(filename, filename + f'{str(dt)}.csv')

    with open(filename, 'w') as f:
        try:
            log.debug(f'Saving train history into {filename}')
        except NameError:
            pass
        # json.dump(history, f)
        # hist_df.to_json(f)
        hist_df.to_csv(f)


def save_results():
    pass


def save_model(args, model):
    current_path = os.path.dirname(os.path.realpath(__file__))
    filename = current_path + '\\saved_model\\'
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except IOError as err:
            print('Save model Error: ', err)

    dt = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M")
    # This method only save model architecture.
    json_file = folder + f'\\model-{args.model_choose}-{args.rnn_type}-{args.predict_seq_len}-{args.final_fusion}-{str(dt)}.json'
    with open(json_file, 'w') as jf:
        jf.write(model.to_json())

    # This method not save custom layers, loss and metrics.
    h5_file = folder + f'\\model-{args.model_choose}-{args.rnn_type}-{args.predict_seq_len}-{args.final_fusion}-{str(dt)}.h5'
    model.save(h5_file, save_format='h5')

    # Recomend using this method of SavedModel format.
    # - Model Architecture/Configuration
    # - Model Weights (learned from training process)
    # - Model Compile Information
    # - Optimizer Status (allows you to resume training where you left off, if any)
    # This will create a directory include two elements:
    # - saved_model.pb: optimizer, loss and metrics
    # - Directory variables/: model weights
    full_model = folder + f'\\full_model-{str(dt)}'
    model.save(full_model)


def load_saved_model(args, format, filepath, custom_objects):
    if format == 'json':
        with open(filepath, 'r') as jf:
            model = model_from_json(jf.read(), custom_objects=custom_objects)
    elif format == 'h5':
        model.load_weights(filepath)
    else:
        model = load_model(filepath, compile=False)

    return model


class LogConfig:
    def __init__(self, logger_name):
        self.logger_name = logger_name
        self.formatter = logging.Formatter('%(asctime)s => %(name)s * %(levelname)s : %(message)s')
        self.log_file = self.get_log_file()

    def get_log_file(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        log_path = current_path + "/log/"
        dt = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M")

        if not os.path.exists(log_path):
            os.mkdir(log_path)
            os.chmod(log_path, 0o777)

        log_file = log_path + str(self.logger_name) + '_' + str(dt) + '.log'
        return log_file

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def get_file_handler(self):
        file_handler = TimedRotatingFileHandler(self.log_file, when='midnight')
        file_handler.setFormatter(self.formatter)
        return file_handler

    def get_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
        logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler())
        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False

        return logger


def log_init_params(args, log):
    log.info('filepath: %s', args.filepath)
    log.info('filename: %s', args.filename)
    log.info('use_features: %s', args.use_features)
    log.info('target_idx: %s', args.target_idx)
    log.info('resolution: %s', args.resolution)
    log.info('sample_interval: %s', args.sample_interval)
    log.info('split_ratios: %s', args.split_ratios)
    log.info('history_seq_len: %s', args.history_seq_len)
    log.info('predict_seq_len: %s', args.predict_seq_len)
    log.info('normalize_method: %s', args.normalize_method)
    log.info('model_choose: %s', args.model_choose)

    log.info('pconv_filters: %s', args.pconv_filters)
    log.info('gconv_filters: %s', args.gconv_filters)
    log.info('gconv_ksize: %s', args.gconv_ksize)
    log.info('gconv_strides: %s', args.gconv_stride)
    log.info('gconv_padding: %s', args.gconv_padding)
    log.info('gconv_groups: %s', args.gconv_groups)
    log.info('dilation_rate: %s', args.dilation_rate)
    log.info('dropout_rate: %s', args.dropout_rate)
    log.info('use_conv: %s', args.use_conv)
    log.info('rnn_type: %s', args.rnn_type)
    log.info('rnn_units: %s', args.rnn_units)
    log.info('rnn_activation: %s', args.rnn_activation)
    log.info('res_conv_filters: %s', args.res_conv_filters)

    log.info('use_valid: %s', args.use_valid)
    log.info('random_seed: %s', args.random_seed)
    log.info('optimizer: %s', args.optimizer)
    log.info('learning_rate: %s', args.learning_rate)
    log.info('batch_size: %s', args.batch_size)
    log.info('epochs: %s', args.epochs)
    log.info('loss_func: %s', args.loss_func)
    log.info('metrics: %s', args.metrics)

    log.info('use_wp_lr: %s', args.use_wp_lr)
    log.info('print_lr: %s', args.print_lr)
    log.info('use_tensorboard: %s', args.use_tensorboard)
    log.info('save_ckpt: %s', args.save_ckpt)
    log.info('use_early_stop: %s', args.use_early_stop)
    log.info('es_monitor: %s', args.es_monitor)
    log.info('es_min_delta: %s', args.es_min_delta)
    log.info('es_patience: %s', args.es_patience)
    log.info('use_csv_logger: %s', args.use_csv_logger)

    log.info('dp_log_name: %s', args.dp_log_name)
    log.info('train_log_name: %s', args.train_log_name)
    log.info('test_log_name: %s', args.test_log_name)
    log.info('get_args_log_name: %s', args.get_args_log_name)

    log.info('plot_history: %s', args.plot_history)
    log.info('save_history: %s', args.save_history)
    log.info('save_results: %s', args.save_results)
