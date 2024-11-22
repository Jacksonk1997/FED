import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings

warnings.filterwarnings('ignore')

from utils import LogConfig


class DataProcessor:
    def __init__(self,
                 log,
                 filepath,
                 filename,
                 use_features,
                 target_idx,
                 resolution,
                 sample_interval,
                 split_ratios,
                 history_seq_len,
                 predict_seq_len,
                 normalize_method='MinMax'
                 ):
        """
        Here is a simple implementation.

        In order to save memory,
        it can be modified as a generator through the __getitem__ method.

        :param filepath: str, csv file absolute path.
        :param filename: str, csv file name including suffix, such as 'data.csv'.
        :param use_features: list[str, ..., str] which features to use.
        :param target_idx: int, the target feature index.
        :param resolution: int, which dataset resolution, such as 5min, 15min, 1hour etc.
        :param sample_interval: int, sample interval indicates how many steps to skip.
        :param split_ratios: list[float, float, float], train, validation and test split ratios.
        :param history_seq_len: int, history sequence length.
        :param predict_seq_len: int, predict sequence length.
        :param normalize_method: str, which method to normalize dataset, support MinMax and Standard.
        :return: train set, validation set and test set.
        """
        super().__init__()

        self.log = log

        self.log.debug("Start loading dataset.")
        self.filepath = filepath
        self.filename = filename
        self.use_features = use_features
        self.df = pd.read_csv(self.filepath + self.filename,
                              encoding='utf-8')
        # this ensures that load original feature types!
        self.df = self.df[self.use_features]
        self.log.debug("Load dataset successfully.")
        self.log.info(f"Use features Names: {self.use_features}.")
        self.log.info(f"Use features Dtype: {self.df.dtypes}.")

        self.df_len = self.df.shape[0]
        self.target_idx = target_idx
        self.resolution = resolution
        self.sample_interval = sample_interval
        self.history_seq_len = history_seq_len
        self.predict_seq_len = predict_seq_len
        self.split_ratios = split_ratios

        # Normalize series method
        self.scaler = None
        self.normalize_method = normalize_method

        if self.predict_seq_len == 1:
            self.single_step_task = True
        else:
            self.single_step_task = False

    def split_dataset(self):
        """
        :param split_ratios: list [float, float, float]
        :return:
        """
        self.log.debug("Start splitting dataset.")
        tr_ratio, va_ratio, te_ratio = self.split_ratios
        self.tr_size = int(self.df_len * tr_ratio)
        self.va_size = int(self.df_len * va_ratio)
        self.te_size = int(self.df_len * te_ratio)
        self.log.info(f"Train size: {self.tr_size}, Valid size: {self.va_size} and Test size: {self.te_size}.")

        self.df_train = self.df.iloc[:self.tr_size, :]
        self.df_valid = self.df.iloc[self.tr_size:-self.te_size, :]
        self.df_test = self.df.iloc[self.tr_size + self.va_size:, :]

        # reset index
        self.df_train.reset_index(inplace=True, drop=True)
        self.df_valid.reset_index(inplace=True, drop=True)
        self.df_test.reset_index(inplace=True, drop=True)

        self.log.debug("Split dataset successfully.")

    def normalize_dataset(self):
        """
        Normalize individually for each column of features.
        :return:
        """
        self.log.debug(f"Start nomalizing dataset by {self.normalize_method}.")
        try:
            if self.normalize_method:
                if self.normalize_method == 'Standard':
                    # Use the training set to prevent data leakage
                    self.scaler = StandardScaler()
                    self.scaler_fit = self.scaler.fit(self.df_train)
                elif self.normalize_method == 'MinMax':
                    self.scaler = MinMaxScaler(feature_range=(0, 1))
                    self.scaler_fit = self.scaler.fit(self.df_train)

                self.normed_df_train = pd.DataFrame(data=self.scaler_fit.transform(self.df_train.values),
                                                    index=self.df_train.index,
                                                    columns=self.df_train.columns
                                                    )
                self.normed_df_valid = pd.DataFrame(data=self.scaler_fit.transform(self.df_valid.values),
                                                    index=self.df_valid.index,
                                                    columns=self.df_valid.columns
                                                    )
                self.normed_df_test = pd.DataFrame(self.scaler_fit.transform(self.df_test.values),
                                                   index=self.df_test.index,
                                                   columns=self.df_test.columns
                                                   )
                self.log.info(f"Normed df test head data: {self.normed_df_test.head()}")
            else:
                self.normed_df_train = self.df_train
                self.normed_df_valid = self.df_valid
                self.normed_df_test = self.df_test

        except NameError as e:
            self.log.error(f"{self.normalize_method} is only support Standard or MinMax!", e)

    def create_dataset(self, input_df, start_idx, end_idx) -> object:
        # construct samples (X)
        samples = []
        # need to predict feature (y)
        targets = []
        # target DataFrame
        # target_df = input_df.iloc[:, self.target_idx]
        target_df = input_df[:, self.target_idx]
        # self.log.info(f"Target feature name: {input_df.columns[-1]}")

        # in order to cut the first sample
        start_idx = start_idx + self.history_seq_len
        # in order to get the correct sample
        if end_idx is None:
            end_idx = len(input_df) - self.predict_seq_len

        self.log.debug("Start create samples.")
        for i in range(start_idx, end_idx):
            # in order to avoid the last sample index over the max index
            if i + self.predict_seq_len >= end_idx:
                pass
            else:
                # get indices to cut df dataset
                indices = range(i - self.history_seq_len, i, self.sample_interval)

                # samples.append(input_df.iloc[indices])
                samples.append(input_df[indices])
                # single step prediction task
                if self.single_step_task:
                    targets.append(target_df[i + self.predict_seq_len])
                # multiple step prediction task
                else:
                    targets.append(target_df[i:i + self.predict_seq_len])

        samples = np.array(samples)
        targets = np.array(targets)
        if self.predict_seq_len == 1:
            targets = np.reshape(targets, (-1, 1))

        self.log.info(f"The samples shape: {samples.shape}, the targets shape: {targets.shape}")

        return (samples, targets)

    # def main(self):
    #     train_X_npy = self.filepath + 'train_X.npy'
    #     train_y_npy = self.filepath + 'train_y.npy'
    #     valid_X_npy = self.filepath + 'valid_X.npy'
    #     valid_y_npy = self.filepath + 'valid_y.npy'
    #     test_X_npy = self.filepath + 'test_X.npy'
    #     test_y_npy = self.filepath + 'test_y.npy'
    #
    #     if not os.path.exists(train_X_npy):
    #         self.split_dataset()
    #         self.normalize_dataset()
    #         self.normed_train = self.create_dataset(self.normed_df_train.values,
    #                                                 start_idx=0, end_idx=self.tr_size)
    #         with open(train_X_npy, 'wb') as f:
    #             np.save(f, self.normed_train[0])
    #         with open(train_y_npy, 'wb') as f:
    #             np.save(f, self.normed_train[1])
    #     else:
    #         with open(train_X_npy, 'rb') as f:
    #             saved_train_X = np.load(f)
    #         with open(train_y_npy, 'rb') as f:
    #             save_train_y = np.load(f)
    #         self.normed_train = (saved_train_X, save_train_y)
    #
    #     if not os.path.exists(valid_X_npy):
    #         self.normed_valid = self.create_dataset(self.normed_df_valid.values,
    #                                                 start_idx=0, end_idx=self.va_size)
    #         with open(valid_X_npy, 'wb') as f:
    #             np.save(f, self.normed_valid[0])
    #         with open(valid_y_npy, 'wb') as f:
    #             np.save(f, self.normed_valid[1])
    #     else:
    #         with open(valid_X_npy, 'rb') as f:
    #             saved_valid_X = np.load(f)
    #         with open(valid_y_npy, 'rb') as f:
    #             save_valid_y = np.load(f)
    #         self.normed_valid = (saved_valid_X, save_valid_y)
    #
    #     if not os.path.exists(test_X_npy):
    #         self.normed_test = self.create_dataset(self.normed_df_test.values,
    #                                                start_idx=0, end_idx=self.te_size)
    #         with open(test_X_npy, 'wb') as f:
    #             np.save(f, self.normed_test[0])
    #         with open(test_y_npy, 'wb') as f:
    #             np.save(f, self.normed_test[1])
    #     else:
    #         with open(test_X_npy, 'rb') as f:
    #             save_test_X = np.load(f)
    #         with open(test_y_npy, 'rb') as f:
    #             save_test_y = np.load(f)
    #         self.normed_test = (save_test_X, save_test_y)
    #
    #     self.log.info("Create Samples Done!")

    def main(self):
        self.split_dataset()
        self.normalize_dataset()

        self.normed_train = self.create_dataset(self.normed_df_train.values,
                                                start_idx=0, end_idx=self.tr_size)

        self.normed_valid = self.create_dataset(self.normed_df_valid.values,
                                                start_idx=0, end_idx=self.va_size)

        self.normed_test = self.create_dataset(self.normed_df_test.values,
                                               start_idx=0, end_idx=self.te_size)

        self.log.info("Create Samples Done!")


if __name__ == '__main__':
    log = LogConfig('data_process').get_logger()
    use_features = ['Air_Pressure', 'Wind_Speed', 'Weather_Temperature_Celsius',
                    'Global_Horizontal_Radiation', 'Pyranometer_1',
                    'hour', 'month',
                    'class_a', 'class_b', 'class_c', 'class_d',
                    'Active_Power',
                    ]
    dpr = DataProcessor(log=log,
                        filepath="D:\\PaperCode\\SCI-01-TIA\\Code\\dataset\\",
                        filename="minute_data.csv",
                        use_features=use_features,
                        target_idx=-1,
                        resolution=15,
                        sample_interval=1,
                        split_ratios=[0.8, 0.1, 0.1],
                        history_seq_len=48,
                        predict_seq_len=1,
                        normalize_method='MinMax')
    dpr.main()

