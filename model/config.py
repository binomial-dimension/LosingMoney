import os
import time
from dataclasses import dataclass
frame = "pytorch"

"""
Config class
"""
@dataclass
class Config:
    # decide what features to use and what to predict
    # columns begin from 0
    feature_columns = list(range(1, 5))
    label_columns = [1, 2, 3, 4]

    # get label index in feature
    label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(
        feature_columns, label_columns)

    # how many days to predict
    predict_day = 1

    # net parameters
    input_size = len(feature_columns)
    output_size = len(label_columns)

    # LSTM hidden size
    hidden_size = 200
    # LSTM layers
    lstm_layers = 2
    # dropout rate
    dropout_rate = 0.2
    # time step use last n days to predict
    time_step = 40

    # train setting
    do_train = True
    do_predict = True

    # if do continue train
    add_train = False
    # if shuffle train data
    shuffle_train_data = True
    # if use cuda, is gpu available and cuda is installed
    use_cuda = True

    # train data rate, test data is 1-train_data_rate
    train_data_rate = 0.83
    # valid data rate, valid data is train_data_rate*valid_data_rate
    valid_data_rate = 0.05

    batch_size = 1024
    learning_rate = 0.001
    epoch = 1000
    # early stop patience
    patience = 40
    random_seed = 42

    # use last state as the initial state of the next batch
    do_continue_train = False
    continue_flag = ""

    # continue train setting
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # debug setting
    # to debug, set debug_mode to True
    debug_mode = False
    # in debug mode, use debug_num data to train
    debug_num = 500

    # frame setting
    used_frame = frame
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + \
        used_frame + model_postfix[used_frame]

    # path setting
    train_data_path = "../data/processed.csv"
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    # log setting
    do_log_print_to_screen = True
    do_log_save_to_file = True
    do_figure_save = False
    # if do train loss visualization with visdom
    do_train_visualized = False

    # path generate
    # model save path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # figure save path
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    # log save path
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)
