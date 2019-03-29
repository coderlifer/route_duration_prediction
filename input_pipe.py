import tensorflow as tf

from feeder import VarFeeder
from enum import Enum
from typing import List, Iterable
# import numpy as np
import pandas as pd


class ModelMode(Enum):
    TRAIN = 0
    EVAL = 1,
    PREDICT = 2


class InputPipe:
    def __init__(self, inp: VarFeeder, features: Iterable[tf.Tensor], n_pages: int, mode: ModelMode, n_epoch=None,
                 batch_size=127, runs_in_burst=1, verbose=True, predict_window=60, train_window=500,
                 train_completeness_threshold=1, predict_completeness_threshold=1, back_offset=0,
                 train_skip_first=0, rand_seed=None):
        """
        Create data preprocessing pipeline
        :param inp: Raw input data
        :param features: Features tensors (subset of data in inp)
        :param n_pages: Total number of pages
        :param mode: Train/Predict/Eval mode selector
        :param n_epoch: Number of epochs. Generates endless data stream if None
        :param batch_size:
        :param runs_in_burst: How many batches can be consumed at short time interval (burst).
        Multiplicator for prefetch()
        :param verbose: Print additional information during graph construction
        :param predict_window: Number of days to predict
        :param train_window: Use train_window days for traning
        :param train_completeness_threshold: Percent of zero datapoints allowed in train timeseries.
        :param predict_completeness_threshold: Percent of zero datapoints allowed in test/predict timeseries.
        :param back_offset: Don't use back_offset days at the end of timeseries
        :param train_skip_first: Don't use train_skip_first days at the beginning of timeseries
        :param rand_seed:

        """
        self.n_pages = n_pages
        # self.inp = inp
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self.back_offset = back_offset

        # TODO: following attributes should match with that in `inp`.
        self.data_days = 1000
        self.data_start = 0
        self.data_end = 1000
        self.features_end = 1000

        if verbose:
            print("Mode:%s, data days:%d, Data start:%s, data end:%s, features end:%s " % (
                mode, self.data_days, self.data_start, self.data_end, self.features_end))

        if mode == ModelMode.TRAIN:
            # reserve predict_window at the end for validation
            assert self.data_days - predict_window > predict_window + train_window, \
                "Predict+train window length (+predict window for validation) is larger " \
                "than total number of days in dataset"
            self.start_offset = train_skip_first
        elif mode == ModelMode.EVAL or mode == ModelMode.PREDICT:
            self.start_offset = self.data_days - train_window - back_offset
            # if verbose:
            #     train_start = self.data_start + pd.Timedelta(self.start_offset, 'D')
            #     eval_start = train_start + pd.Timedelta(train_window, 'D')
            #     end = eval_start + pd.Timedelta(predict_window - 1, 'D')
            #     print("Train start %s, predict start %s, end %s" % (train_start, eval_start, end))
            assert self.start_offset >= 0

        self.train_window = train_window
        self.predict_window = predict_window
        self.attn_window = train_window - predict_window + 1
        self.max_train_empty = int(round(train_window * (1 - train_completeness_threshold)))
        self.max_predict_empty = int(round(predict_window * (1 - predict_completeness_threshold)))
        self.mode = mode
        self.verbose = verbose

        self.encoder_features_depth = 6
        self.norm_std = 1.0
        self.norm_mean = 1.0

        # self.global_active_power_mean = 1.091160489293829
        # self.global_active_power_std = 1.0568687550693088
        # self.global_reactive_power_mean = 0.1237263819487724
        # self.global_reactive_power_std = 0.11275797318457173
        # self.voltage_mean = 240.84178442549432
        # self.voltage_std = 3.2399792136846584
        # self.global_intensity_mean = 4.625819556137465
        # self.global_intensity_std = 4.442533951504559
        # self.sub_metering_1_mean = 1.1217970457916926
        # self.sub_metering_1_std = 6.153957511557926
        # self.sub_metering_2_mean = 1.3010404873906933
        # self.sub_metering_2_std = 5.831016625738528
        # self.sub_metering_3_mean = 6.45443521627108
        # self.sub_metering_3_std = 8.436417544355667

        # Reserve more processing threads for eval/predict because of larger batches
        # num_threads = 3 if mode == ModelMode.TRAIN else 6

        # Create dataset, transform features and assemble batches
        # root_ds = tf.data.Dataset.from_tensor_slices(tuple(features)).repeat(n_epoch)
        # batch = (root_ds
        #          # .map(cutter[mode])
        #          # .filter(self.reject_filter)
        #          # .map(self.make_features, num_parallel_calls=num_threads)
        #          .batch(batch_size * self.train_window)
        #          .prefetch(runs_in_burst * 2)
        #          )

        record_defaults = [tf.float32] * 7
        dataset = tf.data.experimental.CsvDataset(
            features, record_defaults, header=True, select_cols=[2, 3, 4, 5, 6, 7, 8])
        dataset = dataset.repeat(count=n_epoch)
        dataset = dataset.batch(batch_size=batch_size * self.train_window)
        dataset = dataset.prefetch(buffer_size=runs_in_burst * 2)

        self.iterator = dataset.make_initializable_iterator()
        it_tensors = self.iterator.get_next()

        # Assign all tensors to class variables
        global_active_power, global_reactive_power, voltage, global_intensity, \
            sub_metering_1, sub_metering_2, sub_metering_3 = it_tensors
        global_active_power = tf.expand_dims(global_active_power, axis=0)
        global_reactive_power = tf.expand_dims(global_reactive_power, axis=0)
        voltage = tf.expand_dims(voltage, axis=0)
        global_intensity = tf.expand_dims(global_intensity, axis=0)
        sub_metering_1 = tf.expand_dims(sub_metering_1, axis=0)
        sub_metering_2 = tf.expand_dims(sub_metering_2, axis=0)
        sub_metering_3 = tf.expand_dims(sub_metering_3, axis=0)

        x_features = tf.concat([global_reactive_power, voltage, global_intensity,
                                sub_metering_1, sub_metering_2, sub_metering_3], axis=0)
        x_features = tf.transpose(x_features)
        x_features = tf.reshape(x_features, [batch_size, self.train_window, self.encoder_features_depth])
        self.time_x = x_features[:, 0:self.train_window-1, :]
        self.time_y = x_features[:, self.train_window-1:, :]

        y_features = tf.transpose(global_active_power)
        y_features = tf.reshape(y_features, [batch_size, self.train_window])
        self.true_x = y_features[:, :self.train_window-1]
        self.true_y = y_features[:, self.train_window-1:]

    def init_iterator(self, session):
        session.run(self.iterator.initializer)
