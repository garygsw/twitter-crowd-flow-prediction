import os
import time
import logging
import pickle
from copy import copy
import pandas as pd
import numpy as np
import h5py
from datetime import datetime
from preprocessing import MinMaxNormalization, timestamp2vec, string2timestamp


class STMatrix(object):
    """
    """

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

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7,
                       len_period=3, PeriodInterval=1, use_tweet_counts=False):
        """current version
        """
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        # Generate a list of all look back time stamps
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j
                    for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j
                    for j in range(1, len_trend + 1)]]

        # Finding maximum look back in time per step
        i = max(self.T * TrendInterval * len_trend,
                self.T * PeriodInterval * len_period,
                len_closeness)

        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame
                                      for j in depend])

            if Flag is False:
                i += 1
                continue

            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[2]]

            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))

            y = self.get_matrix(self.pd_timestamps[i])
            if use_tweet_counts:  # remove tweet counts in Y
                y = y[:2, :, :]
            Y.append(y)

            timestamps_Y.append(self.timestamps[i])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        return XC, XP, XT, Y, timestamps_Y


class TweetMatrix(STMatrix):
    ''' Class to prepare tweets rolling horizons indexes in the required sequences.
    '''

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        assert len(data) == len(timestamps)
        self.data = data[:-1]              # throw away last one
        self.timestamps = timestamps[1:]   # throw away lag timestamp
        self.T = T
        self.pd_timestamps = string2timestamp(self.timestamps)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7,
                       len_period=3, PeriodInterval=1):
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        TC = []
        TP = []
        TT = []
        timestamps_Y = []

        # Generate a list of all look back time stamps
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j
                    for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j
                    for j in range(1, len_trend + 1)]]

        # Finding maximum look back in time per step
        i = max(self.T * TrendInterval * len_trend,
                self.T * PeriodInterval * len_period,
                len_closeness)

        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame
                                      for j in depend])

            if Flag is False:
                i += 1
                continue

            t_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[0]]
            t_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[1]]
            t_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[2]]

            if len_closeness > 0:
                TC.append(t_c)
            if len_period > 0:
                TP.append(t_p)
            if len_trend > 0:
                TT.append(t_t)

            timestamps_Y.append(self.timestamps[i])
            i += 1

        TC = np.asarray(TC)
        TP = np.asarray(TP)
        TT = np.asarray(TT)
        return TC, TP, TT, timestamps_Y


def stat(fname):
    def get_nb_timeslot(f):
        s = f['date'][0]   # start date
        e = f['date'][-1]  # end date
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
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
    In real-world, we dont have the meteorol data in the predicted timeslot.
    Instead, we use the meteoral at previous timeslots
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


def load_data(datapath, flow_data_filename=None, T=48,
              len_closeness=None, len_period=None, len_trend=None,
              period_interval=1, trend_interval=7, use_mask=False,
              len_test=None, norm_name=None, meta_data=False,
              weather_data=False, holiday_data=False,
              tweet_count_data=False, tweet_index_data=False,
              tweet_count_data_filename=None,
              tweet_norm=None, tweet_index_data_filename=None,
              weather_data_filename=None, holiday_data_filename=None):
    assert(len_closeness + len_period + len_trend > 0)
    # Load the h5 file and retrieve data
    flow_data_path = os.path.join(datapath, flow_data_filename)
    stat(flow_data_path)
    f = h5py.File(flow_data_path, 'r')
    flow_data = f['data'].value
    timestamps = f['date'].value
    mask = f['mask'].value if use_mask else None
    f.close()

    # minmax_scale
    data_train = copy(flow_data)[:-len_test]
    logging.info('train_data shape: ' + str(data_train.shape))
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_mmn = np.array([mmn.transform(d) for d in flow_data])

    # Load tweet count data tile
    if tweet_count_data and tweet_norm is not None:
        tweet_count_path = os.path.join(datapath, tweet_count_data_filename)
        f = h5py.File(tweet_count_path, 'r')
        assert(timestamps[0] == f['date'].value[1])  # due to lag
        assert(timestamps[-1] == f['date'].value[-1])
        tweet_counts = f['count'].value
        tweet_counts = tweet_counts[:-1]  # throw away last
        f.close()

        # Normalize tweet counts
        # tweet_count: (# of timeslots, h, w)
        if tweet_norm == 'day+time':
            hist_seq = {i: {} for i in range(7)}
            for i, timestamp in enumerate(timestamps):
                date, timeslot = timestamp.split('_')
                weekday = datetime.strptime(date, '%Y%m%d').weekday()
                count_matrix = tweet_counts[i]
                hist_seq[weekday][timeslot] = hist_seq[weekday].get(
                    timeslot, []) + [count_matrix]
            hist_max = {i: {} for i in range(7)}
            for weekday, timeslots in hist_seq.iteritems():
                for t, matrix_seq in timeslots.iteritems():
                    hist_max[weekday][t] = np.array(matrix_seq).max(axis=0)
            for i, timestamp in enumerate(timestamps):
                date, timeslot = timestamp.split('_')
                weekday = datetime.strptime(date, '%Y%m%d').weekday()
                tweet_counts[i] = np.divide(
                    tweet_counts[i],
                    hist_max[weekday][timeslot],
                    out=np.zeros_like(tweet_counts[i]),
                    where=hist_max[weekday][timeslot] != 0
                )
        elif tweet_norm == 'all':
            max_counts = tweet_counts.max(axis=0)
            for i, timestamp in enumerate(timestamps):
                tweet_counts[i] = np.divide(
                    tweet_counts[i],
                    max_counts,
                    out=np.zeros_like(tweet_counts[i]),
                    where=max_counts != 0
                )

        # Insert the tweet counts dimension
        data_mmn = np.insert(data_mmn, 2, tweet_counts, axis=1)

    # save preprocessing stats
    fpkl = open(norm_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    # instance-based dataset --> sequences with format as (X, Y) where X is
    # a sequence of images and Y is an image.
    st = STMatrix(data_mmn, timestamps, T, CheckComplete=False)
    XC, XP, XT, Y, timestamps_Y = st.create_dataset(
        len_closeness=len_closeness,
        len_period=len_period,
        len_trend=len_trend,
        PeriodInterval=period_interval,
        TrendInterval=trend_interval,
        use_tweet_counts=tweet_count_data,
    )
    logging.info("XC shape: " + str(XC.shape))
    logging.info("XP shape: " + str(XP.shape))
    logging.info("XT shape: " + str(XT.shape))
    logging.info("Y shape: " + str(Y.shape))

    if tweet_index_data:
        tweet_index_data_path = os.path.join(datapath, tweet_index_data_filename)
        f = h5py.File(tweet_index_data_path, 'r')
        index_data = f['index'].value
        timestamps = f['date'].value
        # max_vocab = tweet_index.attrs['vocab_size']
        # m = tweet_index.attrs['m']
        f.close()

        tm = TweetMatrix(index_data, timestamps, T, CheckComplete=False)
        TC, TP, TT, T_timestamps = tm.create_dataset(
            len_closeness=len_closeness,
            len_period=len_period,
            len_trend=len_trend,
            PeriodInterval=period_interval,
            TrendInterval=trend_interval
        )
        assert(T_timestamps[0] == timestamps_Y[0])
        assert(T_timestamps[-1] == timestamps_Y[-1])
        logging.info("TC shape: " + str(TC.shape))
        logging.info("TP shape: " + str(TP.shape))
        logging.info("TT shape: " + str(TT.shape))

    # Segment the training set
    XC_train = XC[:-len_test]
    XP_train = XP[:-len_test]
    XT_train = XT[:-len_test]
    Y_train = Y[:-len_test]
    timestamp_train = timestamps_Y[:-len_test]
    logging.info('train set XC shape: ' + str(XC_train.shape))
    logging.info('train set XP shape: ' + str(XP_train.shape))
    logging.info('train set XT shape: ' + str(XT_train.shape))
    logging.info('train set Y shape' + str(Y_train.shape))
    if tweet_index_data:
        TC_train = TC[:-len_test]
        TP_train = TP[:-len_test]
        TT_train = TT[:-len_test]
        logging.info('train set TC shape: ' + str(TC_train.shape))
        logging.info('train set TP shape: ' + str(TP_train.shape))
        logging.info('train set TT shape: ' + str(TT_train.shape))

    # Segment the test set
    XC_test = XC[-len_test:]
    XP_test = XP[-len_test:]
    XT_test = XT[-len_test:]
    Y_test = Y[-len_test:]
    timestamp_test = timestamps_Y[-len_test:]
    logging.info('test set XC shape: ' + str(XC_test.shape))
    logging.info('test set XP shape: ' + str(XP_test.shape))
    logging.info('test set XT shape: ' + str(XT_test.shape))
    logging.info('test set Y shape: ' + str(Y_test.shape))
    if tweet_index_data:
        TC_test = TC[-len_test:]
        TP_test = TP[-len_test:]
        TT_test = TT[-len_test:]
        logging.info('test set TC shape: ' + str(TC_test.shape))
        logging.info('test set TP shape: ' + str(TP_test.shape))
        logging.info('test set TT shape: ' + str(TT_test.shape))

    # Prepare the external component
    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_data_path = os.path.join(datapath, holiday_data_filename)
        holiday_feature = load_holiday(timestamps_Y, holiday_data_path)
        meta_feature.append(holiday_feature)
    if weather_data:
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
    if meta_data and holiday_data:
        logging.info('holiday feature shape: ' + str(holiday_feature.shape))
    if meta_data and weather_data:
        logging.info('weather feature shape: ' + str(weather_feature.shape))
    logging.info('meta feature shape: ' + str(meta_feature.shape))

    # Combining the datasets into a list
    X_train = []
    for l, X_, T_ in zip([len_closeness, len_period, len_trend],
                         [XC_train, XP_train, XT_train],
                         [TC_train, TP_train, TT_train]):
        if l > 0:
            X_train.append(X_)
            X_train.append(T_)
    X_test = []
    for l, X_, T_ in zip([len_closeness, len_period, len_trend],
                         [XC_test, XP_test, XT_test],
                         [TC_test, TP_test, TT_test]):
        if l > 0:
            X_test.append(X_)
            X_test.append(T_)

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

    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test, mask
