import sys
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt

def load_logger(config):
    """load logger

    Args:
        config (Config): stroe all the config

    Returns:
        logger: logger
    """    
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # save config
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger

def draw(config, origin_data, logger, predict_norm_data: np.ndarray):
    """draw the predict and label data

    Args:
        config (Config): store all the config
        origin_data (Data): a class store the origin and processed data
        logger (Logger): a logger
        predict_norm_data (numpy array): the normalized predicted data

    Returns:
        four predict and label data: lose_truth,close,open_truth,open,high_truth,high,low_truth,low
    """    
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    # denormalize the data
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]
    
    # check the element number in origin and predicted data
    assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # calculate the mean squared error
    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))
    print("The mean squared error of stock {} is ".format(label_name) + str(loss))

    # draw the predict and label data

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]

    for i in range(label_column_num):
        plt.figure(i+1)
        plt.plot(label_X, label_data[:, i], label='label')
        plt.plot(predict_X, predict_data[:, i], label='predict')
        plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        # calculate the mean squared percentage error
        mape = np.mean(np.abs((label_data[config.predict_day:, i] - predict_data[:-config.predict_day, i]) / label_data[config.predict_day:, i]))
        logger.info("The mean squared percentage error of stock {} is ".format(label_name[i]) + str(mape))
        print("The mean squared percentage error of stock {} is ".format(label_name[i]) + str(mape))

        logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                str(np.squeeze(predict_data[-config.predict_day:, i])))
        
        # save the figure
        if config.do_figure_save:
            plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))

    plt.show()
    return label_data[:, 0], predict_data[:, 0],label_data[:, 1], predict_data[:, 1],label_data[:, 2], predict_data[:, 2],label_data[:,3],predict_data[:,3]
