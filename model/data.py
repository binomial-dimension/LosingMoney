import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Data:
    """a class to read, preprocess and split the data
    """    
    def __init__(self, config):
        """init function of the class
        get mean and std of the data, and normalize the data
        Args:
            config (Config): store all the config
        """        
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        # get the mean and std of the data
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)

        # normalize the data
        self.norm_data = (self.data - self.mean)/self.std

        # for the test data, the first start_num_in_test data is not enough for a time_step
        # so we need to drop them
        self.start_num_in_test = 40

    def read_data(self):
        """read the data from the csv file
        return: data, data_column_name
        """
        # if debug_mode is True, only read the first debug_num rows
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path, nrows=self.config.debug_num,
                                    usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()

    def get_train_and_valid_data(self):
        """get the train and valid data

        Returns: train_x, valid_x, train_y, valid_y:
        train_x and train_y are the feature and label data of the train data
        valid_x and valid_y are the feature and label data of the valid data
        """
        # feature data is the data in the first train_num days
        feature_data = self.norm_data[:self.train_num]
        # label data is the data in the next predict_day days
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.config.label_in_feature_index]

        if not self.config.do_continue_train:
            # in the non-continue train mode, each time_step rows data will be a sample
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            # in the continue train mode, each time_step rows data will be a sample, and the two samples are offset by time_step rows
            # for example: 1-20 rows, 21-40 rows, ... to the end of the data, and then it is 2-21 rows, 22-41 rows, ... to the end of the data, ...
            # so that the final_state of the previous sample can be used as the init_state of the next sample, and it cannot be shuffled
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        # split the train data to train data and valid data
        # shuffle the train data
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        """to get the test data

        Args:
            return_label_data (bool, optional): if to return the label data. Defaults to False.

        Returns:
            test_x: np.array, the test feature data
        """        
        feature_data = self.norm_data[self.train_num:]
        # ensure that the time_step is smaller than the test data
        sample_interval = min(feature_data.shape[0], self.config.time_step)
        # if the test data is not enough for a time_step, we need to drop them
        self.start_num_in_test = feature_data.shape[0] % sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # in the test data, each time_step rows data will be a sample, and the two samples are offset by time_step rows
        # for example: 1-20 rows, 21-40 rows, ... to the end of the data
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        
        # in real test, we do not have the label data, so we do not need to return it
        if return_label_data:
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)
    
    def get_noshuffle_train_data(self,return_label_data=True):
        feature_data = self.norm_data[:self.train_num]
        # ensure that the time_step is smaller than the test data
        sample_interval = min(feature_data.shape[0], self.config.time_step)
        # if the test data is not enough for a time_step, we need to drop them
        self.start_num_in_test = feature_data.shape[0] % sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # in the test data, each time_step rows data will be a sample, and the two samples are offset by time_step rows
        # for example: 1-20 rows, 21-40 rows, ... to the end of the data
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        
        # in real test, we do not have the label data, so we do not need to return it
        if return_label_data:
            label_data = self.norm_data[self.start_num_in_test:self.train_num, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)