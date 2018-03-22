import os
import time
import logging
import cPickle as pickle
from copy import copy
import pandas as pd
import numpy as np
import h5py
from preprocessing import MinMaxNormalization, timestamp2vec, string2timestamp


class STMatrix(object):

    '''Class to prepare rolling horizons indexes in the required sequences.'''

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            logging.debug(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        ''' Returns the matrix given a timestamp.
        '''
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        ''' Check if the time stamps in depends in valid (in the dataset).
        '''
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_hour=3, len_day=3, day_interval=1,
                       len_week=3, week_interval=7, len_tweet=1,
                       use_tweet_features=False, aggregate_counts=True):
        '''Prepare rolling horizon dataset.'''
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XH = []
        XD = []
        XW = []
        Y = []
        timestamps_Y = []

        if use_tweet_features and aggregate_counts:
            len_tweet = 1

        # Generate a list of all look back time stamps
        depends = [range(1, len_tweet + 1),
                   range(1, len_hour + 1),
                   [day_interval * self.T * j
                    for j in range(1, len_day + 1)],
                   [week_interval * self.T * j
                    for j in range(1, len_week + 1)]]

        # Finding maximum look back in time per step (for period and trend only)
        i = max(self.T * day_interval * len_day,
                self.T * week_interval * len_week,
                len_hour)

        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame
                                      for j in depend])

            if Flag is False:  # roll forward until the first prediciton time step
                i += 1
                continue

            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[0]]
            x_h = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[1]]
            x_d = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[2]]
            x_w = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[3]]
            if use_tweet_features and len_tweet > 0:
                # separete the tweets from the flows
                x_t = [matrix[2:, :, :] for matrix in x_t]
                x_h = [matrix[:2, :, :] for matrix in x_h]
                x_d = [matrix[:2, :, :] for matrix in x_d]
                x_w = [matrix[:2, :, :] for matrix in x_w]

            if len_hour > 0:
                if use_tweet_features and len_tweet > 0:
                    x_h += x_t
                XH.append(np.vstack(x_h))
            if len_day > 0:
                XD.append(np.vstack(x_d))
            if len_week > 0:
                XW.append(np.vstack(x_w))

            y = self.get_matrix(self.pd_timestamps[i])
            if use_tweet_features and len_tweet > 0:  # remove tweet counts in Y
                y = y[:2, :, :]
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])

            i += 1

        XH = np.asarray(XH)
        XD = np.asarray(XD)
        XW = np.asarray(XW)
        Y = np.asarray(Y)
        return XH, XD, XW, Y, timestamps_Y


def stat(fname, T):
    def get_nb_timeslot(f):
        s = f['date'][0]   # start date
        e = f['date'][-1]  # end date
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = ((time.mktime(te) - time.mktime(ts)) / 86400 + 1) * T
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / T)
        mmax = f['data'].value.max()
        mmin = f['data'].value.min()
        single_mask = f['mask'].value
        if single_mask is not None:
            inflow_cells = np.sum(single_mask[0])
            outflow_cells = np.sum(single_mask[1])
            mask_info = '# of valid inflow cells: %i\n' % inflow_cells + \
                        '# of valid outflow cells: %i\n' % outflow_cells
        else:
            mask_info = ''
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + mask_info + \
               '=' * 5 + 'stat' + '=' * 5 + '\n'
        logging.info(stat)


def load_holiday(timeslots, datapath):
    f = open(datapath, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    logging.info("total number of holidays/weekends: " + str(H.sum()))
    return H[:, None]


def load_weather(timeslots, datapath):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the weather data in the predicted timeslot.
    Instead, we use the weather at previous timeslots
    i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    h5 file, which has four following subsets:
    date: a list of timeslots, which is associated the following kinds of data.
    Temperature: a list of continuous value, of which the i'th value is temperature at the timeslot date[i].
    WindSpeed: a list of continuous value, of which the i'th value is wind speed at the timeslot date[i].
    Weather: a 2D matrix, each of which is a one-hot vector (dim=8), showing one of the following weather types:
    sunny = 0
    cloudy = 1
    overcast = 2
    rain = 3
    light rain = 4
    heavy rain = 5
    fog = 6
    haze = 7
    '''
    f = h5py.File(datapath, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['windspeed'].value
    Weather = f['weather'].value
    Temperature = f['temperature'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # windspeed
    WR = []  # weather
    TE = []  # temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    logging.info("windspeed shape: " + str(WS.shape))
    logging.info("weather shape: " + str(WR.shape))
    logging.info("temperature shape: " + str(TE.shape))

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    logging.info('merged shape: ' + str(merge_data.shape))
    return merge_data


def read_count_data(count_name, datapath, flow_timestamps, aggregate_counts,
                    len_lag, len_lead, data_lag, data_lead):
    # Load tweet count data tile
    logging.info('reading counts data for %s...' % count_name)
    if not os.path.exists(datapath):
        raise Exception('counts input data path "%s" does not exists' % datapath)
    f = h5py.File(datapath, 'r')
    assert(flow_timestamps[0] == f['date'].value[data_lag])
    assert(flow_timestamps[-1] == f['date'].value[-data_lead - 1])
    counts_data = f['count'].value
    within_window_counts = counts_data[data_lag - 1:-data_lead - 1]
    assert(len(within_window_counts) == len(flow_timestamps))
    f.close()

    # Aggregate the counts
    logging.info('aggregating counts for %s...' % count_name)
    if aggregate_counts:
        for i, timestamp in enumerate(flow_timestamps):
            # add up the lags
            if len_lag:
                if len_lag < data_lag + i:
                    pass  # ignore those with insufficient history data
                elif len_lag > 1:
                    for j in range(1, len_lag):
                        within_window_counts[i] += counts_data[i - j]
            # add up the leads
            if len_lead > 0:
                for j in range(len_lead):
                    within_window_counts[i] += counts_data[i + j + len_lag]

    # Normalize counts
    logging.info('normalizing counts for %s...' % count_name)
    max_counts = within_window_counts.max(axis=0)
    min_counts = within_window_counts.min(axis=0)
    denominator = max_counts - min_counts
    for i, timestamp in enumerate(flow_timestamps):
        numerator = within_window_counts[i] - min_counts
        within_window_counts[i] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0
        )
    return within_window_counts * 2 - 1


def load_data(datapath, flow_data_filename=None, T=48,
              len_hour=None, len_day=None, len_week=None,
              len_lag_tweets=None, len_lead_tweets=None,
              day_interval=1, week_interval=7,
              use_mask=False, len_test=None, norm_name=None, meta_data=False,
              use_weather=False, use_holiday=False, use_tweet_counts=False,
              use_past_counts=False, use_present_counts=False,
              use_future_counts=False,
              use_positive_counts=False, use_negative_counts=False,
              weather_data_filename=None,
              holiday_data_filename=None,
              tweet_count_data_filename=None,
              future_count_data_filename=None,
              past_count_data_filename=None,
              present_count_data_filename=None,
              positive_count_data_filename=None,
              negative_count_data_filename=None,
              aggregate_counts=False,
              tweet_lag=1, tweet_lead=0):
    assert(len_hour + len_day + len_week > 0)
    # Load the h5 file and retrieve data
    flow_data_path = os.path.join(datapath, flow_data_filename)
    if not os.path.exists(flow_data_path):
        raise Exception('input data path "%s" does not exists' % flow_data_path)
    logging.info('reading flow data...')
    stat(flow_data_path, T)
    f = h5py.File(flow_data_path, 'r')
    flow_data = f['data'].value
    timestamps = f['date'].value
    mask = f['mask'].value if use_mask else None
    f.close()

    # minmax_scale
    data_train = copy(flow_data)[:-len_test]
    logging.info('flows training data shape: ' + str(data_train.shape))
    mmn = MinMaxNormalization()
    logging.info('normalizing flows training data...')
    mmn.fit(data_train)
    data_mmn = np.array([mmn.transform(d) for d in flow_data])

    # Load tweets features count
    len_tweet = 0
    if len_lag_tweets is not None:
        len_tweet += len_lag_tweets
    if len_lead_tweets is not None:
        len_tweet += len_lead_tweets
    use_tweets = (use_tweet_counts or use_future_counts or use_present_counts) or \
                 (use_past_counts or use_positive_counts or use_negative_counts) or \
                 (use_positive_counts or use_negative_counts)
    if use_tweet_counts:
        tweet_counts = read_count_data(
            count_name='tweet_counts',
            datapath=os.path.join(datapath, tweet_count_data_filename),
            flow_timestamps=timestamps,
            aggregate_counts=aggregate_counts,
            len_lag=len_lag_tweets,
            len_lead=len_lead_tweets,
            data_lag=tweet_lag,
            data_lead=tweet_lead
        )
        # Insert the tweet counts dimension
        data_mmn = np.insert(data_mmn, 2, tweet_counts, axis=1)
    if use_future_counts:
        future_counts = read_count_data(
            count_name='future_counts',
            datapath=os.path.join(datapath, future_count_data_filename),
            flow_timestamps=timestamps,
            aggregate_counts=aggregate_counts,
            len_lag=len_lag_tweets,
            len_lead=len_lead_tweets,
            data_lag=tweet_lag,
            data_lead=tweet_lead
        )
        # Insert the future counts dimension
        data_mmn = np.insert(data_mmn, 2, future_counts, axis=1)
    if use_past_counts:
        past_counts = read_count_data(
            count_name='past_counts',
            datapath=os.path.join(datapath, past_count_data_filename),
            flow_timestamps=timestamps,
            aggregate_counts=aggregate_counts,
            len_lag=len_lag_tweets,
            len_lead=len_lead_tweets,
            data_lag=tweet_lag,
            data_lead=tweet_lead
        )
        # Insert the past counts dimension
        data_mmn = np.insert(data_mmn, 2, past_counts, axis=1)
    if use_present_counts:
        present_counts = read_count_data(
            count_name='present_counts',
            datapath=os.path.join(datapath, present_count_data_filename),
            flow_timestamps=timestamps,
            aggregate_counts=aggregate_counts,
            len_lag=len_lag_tweets,
            len_lead=len_lead_tweets,
            data_lag=tweet_lag,
            data_lead=tweet_lead
        )
        # Insert the present counts dimension
        data_mmn = np.insert(data_mmn, 2, present_counts, axis=1)
    if use_positive_counts:
        positive_counts = read_count_data(
            count_name='positive_counts',
            datapath=os.path.join(datapath, positive_count_data_filename),
            flow_timestamps=timestamps,
            aggregate_counts=aggregate_counts,
            len_lag=len_lag_tweets,
            len_lead=len_lead_tweets,
            data_lag=tweet_lag,
            data_lead=tweet_lead
        )
        # Insert the positive counts dimension
        data_mmn = np.insert(data_mmn, 2, positive_counts, axis=1)
    if use_negative_counts:
        negative_counts = read_count_data(
            count_name='negative_counts',
            datapath=os.path.join(datapath, negative_count_data_filename),
            flow_timestamps=timestamps,
            aggregate_counts=aggregate_counts,
            len_lag=len_lag_tweets,
            len_lead=len_lead_tweets,
            data_lag=tweet_lag,
            data_lead=tweet_lead
        )
        # Insert the negative counts dimension
        data_mmn = np.insert(data_mmn, 2, negative_counts, axis=1)

    # save preprocessing stats
    fpkl = open(norm_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    # instance-based dataset --> sequences with format as (X, Y) where X is
    # a sequence of images and Y is an image.
    st = STMatrix(data_mmn, timestamps, T, CheckComplete=False)
    logging.info('creating training and tests flow inputs...')
    XH, XD, XW, Y, timestamps_Y = st.create_dataset(
        len_hour=len_hour,
        len_day=len_day,
        len_week=len_week,
        len_tweet=len_tweet,
        day_interval=day_interval,
        week_interval=week_interval,
        use_tweet_features=use_tweets,
        aggregate_counts=aggregate_counts
    )
    logging.info("XH shape: " + str(XH.shape))
    logging.info("XD shape: " + str(XD.shape))
    logging.info("XW shape: " + str(XW.shape))
    logging.info("Y shape: " + str(Y.shape))

    # Segment the training set
    XH_train = XH[:-len_test]
    XD_train = XD[:-len_test]
    XW_train = XW[:-len_test]
    Y_train = Y[:-len_test]
    timestamp_train = timestamps_Y[:-len_test]
    logging.info('train set XH shape: ' + str(XH_train.shape))
    logging.info('train set XD shape: ' + str(XD_train.shape))
    logging.info('train set XW shape: ' + str(XW_train.shape))
    logging.info('train set Y shape' + str(Y_train.shape))

    # Segment the test set
    XH_test = XH[-len_test:]
    XD_test = XD[-len_test:]
    XW_test = XW[-len_test:]
    Y_test = Y[-len_test:]
    timestamp_test = timestamps_Y[-len_test:]
    logging.info('test set XH shape: ' + str(XH_test.shape))
    logging.info('test set XD shape: ' + str(XD_test.shape))
    logging.info('test set XW shape: ' + str(XW_test.shape))
    logging.info('test set Y shape: ' + str(Y_test.shape))

    # Prepare the external component
    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if use_holiday:
        # load holiday
        holiday_data_path = os.path.join(datapath, holiday_data_filename)
        holiday_feature = load_holiday(timestamps_Y, holiday_data_path)
        meta_feature.append(holiday_feature)
    if use_weather:
        # load weather data
        weather_data_path = os.path.join(datapath, weather_data_filename)
        weather_feature = load_weather(timestamps_Y, weather_data_path)
        meta_feature.append(weather_feature)
    if len(meta_feature) > 0:
        meta_feature = np.hstack(meta_feature)
    else:
        meta_feature = np.asarray(meta_feature)

    if len(meta_feature.shape) > 1:
        metadata_dim = meta_feature.shape[1]
    else:
        metadata_dim = None
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data:
        logging.info('time feature shape: ' + str(time_feature.shape))
    if meta_data and use_holiday:
        logging.info('holiday feature shape: ' + str(holiday_feature.shape))
    if meta_data and use_weather:
        logging.info('weather feature shape: ' + str(weather_feature.shape))
    logging.info('meta feature shape: ' + str(meta_feature.shape))

    # Combining the datasets into a list
    X_train = []
    X_test = []
    train_datasets_list = zip([len_hour, len_day, len_week],
                              [XH_train, XD_train, XW_train])
    test_datasets_list = zip([len_hour, len_day, len_week],
                             [XH_test, XD_test, XW_test])

    for l, X_ in train_datasets_list:
        if l > 0:
            X_train.append(X_)

    for l, X_ in test_datasets_list:
        if l > 0:
            X_test.append(X_)

    if metadata_dim is not None:
        meta_feature_train = meta_feature[:-len_test]
        meta_feature_test = meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    # type_map = {0: 'closeness', 1: 'period', 2: 'trend', 3: 'meta'}
    for i, _X in enumerate(X_train):
        logging.info('X train shape at index %s ' % i + ': ' + str(_X.shape))
    for i, _X in enumerate(X_test):
        logging.info('X test shape at index %s ' % i + ': ' + str(_X.shape))

    return (X_train, Y_train, X_test, Y_test, mmn, metadata_dim,
            timestamp_train, timestamp_test, mask)
