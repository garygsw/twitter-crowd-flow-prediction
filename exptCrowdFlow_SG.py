'''exptCrowdFlow_SG.py.

Crowd Flow Prediction experiment on the Singapore city.
'''

import logging
import sys
import os
import cPickle as pickle
import time
import numpy as np
import h5py
import warnings
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from deepst.models.STResNet import stresnet
import deepst.metrics as metrics
from dataset import load_data

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# Input parameters
np.random.seed(1337)  # for reproducibility
city_name = 'SG'
ds_name = 'VLDset1'   # dataset name
map_height, map_width = 49, 89  # (23, 44) - 1km, (46, 87) - 500m
len_interval = 30               # 30 minutes per time slot
CACHEDATA = False               # cache data or NOT
use_meta = True
use_weather = True
use_holidays = True
use_tweet_counts = True
use_future_tense_counts = False
use_past_tense_counts = False
use_positive_counts = False
use_negative_counts = False
aggregate_counts = True
batch_size = 32              # batch size
nb_epoch = 500               # number of epoch at training stage
nb_epoch_cont = 100          # number of epoch at training (cont) stage
T = 24 * 60 / len_interval   # number of time intervals in one day
lr = 0.0002                  # learning rate
nb_residual_unit = 2         # number of residual units
day_interval = 1             # day interval length (in days)
week_interval = 7            # week interval length (in days)
kernal_size = (3, 3)         # window for convolutional NN
nb_filters = 64              # for conv1 layer
days_test = 7 * 4            # number of days from the back as test set
len_test = T * days_test
validation_split = 0.1       # during development training phase
len_hourly = 4               # length of hourly historical sequence
len_daily = 1                # length of daily historical sequence
len_weekly = 1               # length of weekly historical sequence
tweet_lag = 1                # how many lags tweet info in dataset
tweet_lead = 10              # how many lead tweet info in dataset
len_lag_tweets = 2           # length of tweets lag dependent sequence
len_lead_tweets = 0          # length of tweets lead dependent sequence
len_tweets_window = len_lag_tweets + len_lead_tweets

# Define inputs filename formats
flow_data_fname = '{}_{}_M{}x{}_T{}_InOut.h5'.format(city_name,
                                                     ds_name,
                                                     map_width,
                                                     map_height,
                                                     len_interval)
weather_data_fname = '{}_{}_T{}_Weather.h5'.format(city_name,
                                                   ds_name,
                                                   len_interval)
holiday_data_fname = '{}_{}_Holidays.txt'.format(city_name, ds_name)
tweet_counts_data_fname = '{}_{}_M{}x{}_T{}_TweetCount-{}+{}.h5'.format(
    city_name,
    ds_name,
    map_width,
    map_height,
    len_interval,
    tweet_lag,
    tweet_lead,
)
future_counts_data_fname = '{}_{}_M{}x{}_T{}_FutureCount-{}+{}.h5'.format(
    city_name,
    ds_name,
    map_width,
    map_height,
    len_interval,
    tweet_lag,
    tweet_lead,
)
past_counts_data_fname = '{}_{}_M{}x{}_T{}_PastCount-{}+{}.h5'.format(
    city_name,
    ds_name,
    map_width,
    map_height,
    len_interval,
    tweet_lag,
    tweet_lead,
)
positive_counts_data_fname = '{}_{}_M{}x{}_T{}_PositiveCount-{}+{}.h5'.format(
    city_name,
    ds_name,
    map_width,
    map_height,
    len_interval,
    tweet_lag,
    tweet_lead,
)
negative_counts_data_fname = '{}_{}_M{}x{}_T{}_NegativeCount-{}+{}.h5'.format(
    city_name,
    ds_name,
    map_width,
    map_height,
    len_interval,
    tweet_lag,
    tweet_lead,
)
DATAPATH = 'dataset'
path_cache = os.path.join(DATAPATH, 'CACHE')     # cache path
path_norm = os.path.join(DATAPATH, 'NORM')       # normalization path
path_hist = 'HIST'                  # history path
path_model = 'MODEL'                # model path
path_log = 'LOG'                    # log path
path_predictions = 'PRED'           # predictions path
checkpoint_verbose = True
development_training_verbose = True
development_evaluate_verbose = True
full_training_verbose = True
full_evaluate_verbose = True
model_plot = False
model_fpath = 'model.png'
use_mask = True
warnings.filterwarnings('ignore')


# Make the folders and the respective paths if it does not already exists
DS_DATAPATH = os.path.join(DATAPATH, ds_name)  # add ds folder name
if not os.path.isdir(path_hist):
    os.mkdir(path_hist)
path_hist = os.path.join(path_hist, ds_name)
if not os.path.isdir(path_hist):
    os.mkdir(path_hist)
if not os.path.isdir(path_model):
    os.mkdir(path_model)
path_model = os.path.join(path_model, ds_name)
if not os.path.isdir(path_model):
    os.mkdir(path_model)
if not os.path.isdir(path_log):
    os.mkdir(path_log)
path_log = os.path.join(path_log, ds_name)
if not os.path.isdir(path_log):
    os.mkdir(path_log)
if not os.path.isdir(path_predictions):
    os.mkdir(path_predictions)
path_predictions = os.path.join(path_predictions, ds_name)
if not os.path.isdir(path_predictions):
    os.mkdir(path_predictions)
if CACHEDATA:
    if not os.path.isdir(path_cache):
        os.mkdir(path_cache)
    if not os.path.isdir(path_norm):
        os.mkdir(path_norm)


# Define filename of files based on meta, c, p & t parameters
meta_info = []
if use_meta and use_weather:
    meta_info.append('W')
if use_meta and use_holidays:
    meta_info.append('H')
if len(meta_info) > 1:
    meta_info = '_Ext.' + '.'.join(meta_info)
else:
    meta_info = ''
mask_info = '_masked' if use_mask else ''
tweet_count_info = '_tweetcount' if use_tweet_counts else ''
future_count_info = '_futurecount' if use_future_tense_counts else ''
past_count_info = '_pastcount' if use_past_tense_counts else ''
positive_count_info = '_positivecount' if use_positive_counts else ''
negative_count_info = '_negativecount' if use_negative_counts else ''
if use_tweet_counts:
    tweet_len_info = '_tweetlen-%s+%s' % (len_lag_tweets, len_lead_tweets)
else:
    tweet_len_info = ''

cache_fname = ('{0}_{1}_M{2}x{3}_T{4}_b{5}_c{6}.p{7}.t{8}'
               '{9}{10}{11}{12}{13}{14}{15}{16}.h5').format(
    city_name,            # 0
    ds_name,              # 1
    map_width,            # 2
    map_height,           # 3
    len_interval,         # 4
    batch_size,           # 5
    len_hourly,           # 6
    len_daily,            # 7
    len_weekly,           # 8
    meta_info,            # 9
    mask_info,            # 10
    tweet_count_info,     # 11
    tweet_len_info,       # 12
    future_count_info,    # 13
    past_count_info,      # 14
    positive_count_info,  # 15
    negative_count_info,  # 16
)
cache_fpath = os.path.join(path_cache, cache_fname)
norm_fname = '{}_{}_Normalizer.pkl'.format(city_name, ds_name)
norm_fpath = os.path.join(path_norm, norm_fname)

# Define the file paths of the result and model files
hyperparams_name = ('{0}_{1}_M{2}x{3}_T{4}_b{5}_c{6}.p{7}.t{8}{9}{10}'
                    '_resunit{11}_lr{12}{13}{14}{15}{16}{17}{18}').format(
    city_name,            # 0
    ds_name,              # 1
    map_width,            # 2
    map_height,           # 3
    len_interval,         # 4
    batch_size,           # 5
    len_hourly,           # 6
    len_daily,            # 7
    len_weekly,           # 8
    meta_info,            # 9
    mask_info,            # 10
    nb_residual_unit,     # 11
    lr,                   # 12
    tweet_count_info,     # 13
    tweet_len_info,       # 14
    future_count_info,    # 15
    past_count_info,      # 16
    positive_count_info,  # 17
    negative_count_info,  # 18
)
dev_checkpoint_fname = '{}.dev.best.h5'.format(hyperparams_name)
dev_checkpoint_fpath = os.path.join(path_model, dev_checkpoint_fname)
dev_weights_fname = '{}.dev.weights.h5'.format(hyperparams_name)
dev_weights_fpath = os.path.join(path_model, dev_weights_fname)
dev_history_fname = '{}.dev.history.pkl'.format(hyperparams_name)
dev_history_fpath = os.path.join(path_hist, dev_history_fname)
full_checkpoint_fname = '{}.full.best.h5'.format(hyperparams_name)
full_checkpoint_fpath = os.path.join(path_model, full_checkpoint_fname)
full_weights_fname = '{}.full.weights.h5'.format(hyperparams_name)
full_weights_fpath = os.path.join(path_model, full_weights_fname)
full_history_fname = '{}.full.history.pkl'.format(hyperparams_name)
full_history_fpath = os.path.join(path_hist, full_history_fname)
predictions_fname = '{}.stressnet.predictions.npy'.format(hyperparams_name)
predictions_fpath = os.path.join(path_predictions, predictions_fname)
pred_timestamps_fname = 'timestamps_T{}_{}.npy'.format(len_interval, len_test)
pred_timestamps_fpath = os.path.join(path_predictions, pred_timestamps_fname)
test_true_y_fname = 'true_y_T{}_{}.npy'.format(len_interval, len_test)
test_true_y_fpath = os.path.join(path_predictions, test_true_y_fname)

# Define logging parameters
local_time = time.localtime()
log_fname = time.strftime('%Y-%m-%d_%H-%M-%S_', local_time) + hyperparams_name
log_fname += '.out'
log_fpath = os.path.join(path_log, log_fname)
fileHandler = logging.FileHandler(log_fpath)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler(sys.stdout)
root_logger.handlers = [fileHandler, consoleHandler]


def build_model(external_dim, loss, metric):
    '''Define the model configuration and optimizer, and compiles it.'''
    len_hour = None if len_hourly <= 0 else len_hourly
    len_day = None if len_daily <= 0 else len_daily
    len_week = None if len_weekly <= 0 else len_weekly
    len_tweet = None if len_tweets_window <= 0 else len_tweets_window

    model = stresnet(map_height=map_height,
                     map_width=map_width,
                     len_hour=len_hour,
                     len_day=len_day,
                     len_week=len_week,
                     len_tweet=len_tweet,
                     external_dim=external_dim,
                     nb_residual_unit=nb_residual_unit,
                     nb_filters=nb_filters,
                     kernal_size=kernal_size,
                     use_tweet_counts=use_tweet_counts,
                     use_future_tense_counts=use_future_tense_counts,
                     use_past_tense_counts=use_past_tense_counts,
                     use_positive_counts=use_positive_counts,
                     use_negative_counts=use_negative_counts,
                     aggregate_counts=aggregate_counts)
    adam = Adam(lr=lr)
    model.compile(loss=loss, optimizer=adam, metrics=[metric])
    model.summary()
    if model_plot:
        from keras.utils import plot_model
        plot_model(model, to_file=model_fpath, show_shapes=True)
        logging.info('Model plotted in %s' % model_fpath)
    return model


def read_cache(cache_fpath, norm_fpath, use_mask):
    '''Read the prepared dataset (train and test set prepared).'''
    logging.info('reading %s...' % cache_fpath)
    mmn = pickle.load(open(norm_fpath, 'rb'))
    f = h5py.File(cache_fpath, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in xrange(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    mask = f['mask'].value if use_mask else None
    f.close()
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, \
        timestamp_test, mask


def cache(cache_fpath, X_train, Y_train, X_test, Y_test, external_dim,
          timestamp_train, timestamp_test, mask):
    '''Create cache file for the prepared dataset.'''
    h5 = h5py.File(cache_fpath, 'w')
    h5.create_dataset('num', data=len(X_train))
    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    if mask is not None:
        h5.create_dataset('mask', data=mask)
    h5.close()


def print_elasped(from_ts, title):
    '''helper function to print elasped time.'''
    elasped_seconds = time.time() - from_ts
    logging.info('\nelapsed time (%s): %.3f seconds\n' %
                 (title, elasped_seconds))


def print_header(message):
    '''helper function to print header.'''
    logging.info('=' * 10)
    logging.info(message + '\n')


def main():
    '''main function.'''
    # load data
    print_header('loading data...')
    ts = time.time()
    cache_exists = os.path.exists(cache_fpath)
    norm_exists = os.path.exists(norm_fpath)
    if CACHEDATA and cache_exists and norm_exists:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, \
            timestamp_test, mask = read_cache(cache_fpath,
                                              norm_fpath,
                                              use_mask)
        logging.info('loaded %s successfully' % cache_fpath)
    else:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, \
            timestamp_test, mask = load_data(
                datapath=DS_DATAPATH,
                flow_data_filename=flow_data_fname,
                T=T,
                len_hour=len_hourly,
                len_day=len_daily,
                len_week=len_weekly,
                len_lag_tweets=len_lag_tweets,
                len_lead_tweets=len_lead_tweets,
                day_interval=day_interval,
                week_interval=week_interval,
                len_test=len_test,
                norm_name=norm_fpath,
                meta_data=use_meta,
                use_weather=use_weather,
                use_holiday=use_holidays,
                use_tweet_counts=use_tweet_counts,
                use_future_tense_counts=use_future_tense_counts,
                use_past_tense_counts=use_past_tense_counts,
                use_positive_counts=use_positive_counts,
                use_negative_counts=use_negative_counts,
                aggregate_counts=aggregate_counts,
                tweet_lag=tweet_lag,
                tweet_lead=tweet_lead,
                weather_data_filename=weather_data_fname,
                holiday_data_filename=holiday_data_fname,
                tweet_count_data_filename=tweet_counts_data_fname,
                future_count_data_filename=future_counts_data_fname,
                past_count_data_filename=past_counts_data_fname,
                positive_count_data_filename=positive_counts_data_fname,
                negative_count_data_filename=negative_counts_data_fname,
                use_mask=use_mask
            )
        if CACHEDATA:
            cache(cache_fpath, X_train, Y_train, X_test, Y_test, external_dim,
                  timestamp_train, timestamp_test, mask)
    print_elasped(ts, 'loading data')

    print_header("compiling model...")
    ts = time.time()
    # use masked rmse if use_mask
    loss_function = metrics.masked_mse(mask) if use_mask else metrics.mse
    metric_function = metrics.masked_rmse(mask) if use_mask else metrics.rmse
    model = build_model(external_dim,
                        loss_function,
                        metric_function)
    print_elasped(ts, 'model compilation')

    print_header("training model (development)...")
    ts = time.time()
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(dev_checkpoint_fpath,
                                       monitor='val_loss',
                                       verbose=checkpoint_verbose,
                                       save_best_only=True,
                                       mode='min')
    history = model.fit(X_train,
                        Y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=development_training_verbose)
    model.save_weights(dev_weights_fpath, overwrite=True)
    pickle.dump((history.history), open(dev_history_fpath, 'wb'))
    total_training_time = time.time() - ts
    print_elasped(ts, 'development training')

    print_header('evaluate the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(dev_weights_fpath)
    score = model.evaluate(X_train,
                           Y_train,
                           batch_size=batch_size,
                           verbose=development_evaluate_verbose)
    logging.info('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                 (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    score = model.evaluate(X_test,
                           Y_test,
                           batch_size=batch_size,
                           verbose=development_evaluate_verbose)
    logging.info('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                 (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print_elasped(ts, 'development evaluation')

    print_header("training model (full)...")
    ts = time.time()
    model_checkpoint = ModelCheckpoint(full_checkpoint_fpath,
                                       monitor='rmse',
                                       verbose=checkpoint_verbose,
                                       save_best_only=True,
                                       mode='min')
    history = model.fit(X_train,
                        Y_train,
                        epochs=nb_epoch_cont,
                        verbose=full_training_verbose,
                        batch_size=batch_size,
                        callbacks=[model_checkpoint])
    pickle.dump((history.history), open(full_history_fpath, 'wb'))
    model.save_weights(full_weights_fpath, overwrite=True)
    total_training_time += time.time() - ts
    print_elasped(ts, 'full training')

    print_header('evaluating using the final model')
    ts = time.time()
    score = model.evaluate(X_train,
                           Y_train,
                           batch_size=batch_size,
                           verbose=full_evaluate_verbose)
    logging.info('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                 (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    score = model.evaluate(X_test,
                           Y_test,
                           batch_size=batch_size,
                           verbose=full_evaluate_verbose)
    logging.info('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                 (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print_elasped(ts, 'full evaluation')

    # saves the prediction results
    predictions = model.predict(X_test)
    np.save(predictions_fpath, predictions)
    np.save(test_true_y_fpath, Y_test)
    np.save(pred_timestamps_fpath, timestamp_test)

    # prints the full RMSE scores
    if use_mask:
        full_mask = np.tile(mask, [len_test, 1, 1, 1])
        errors = Y_test[full_mask] - predictions[full_mask]
        full_rmse = (errors ** 2).mean() ** 0.5
    else:
        full_rmse = ((Y_test - predictions) ** 2).mean() ** 0.5
    logging.info('Full RMSE: %.6f' % full_rmse)

    # log the total training time
    hours = total_training_time // 3600
    total_training_time %= 3600
    minutes = total_training_time // 60
    total_training_time %= 60
    seconds = total_training_time
    logging.info('Total training time: %d hrs %d mins %.6f s' % (hours,
                                                                 minutes,
                                                                 seconds))

if __name__ == '__main__':
    main()
