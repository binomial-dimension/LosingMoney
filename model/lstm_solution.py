import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from data import Data
from utils import load_logger
from lstm import train, predict

frame = 'pytorch'

def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
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


def main(config):
    """main function

    Args:
        config (Config): store all the config
    """    
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)

            # denormalized predict data
            pred_result = predict(config, test_X) 
            close_truth,close,open_truth,open,high_truth,high,low_truth,low = draw(config, data_gainer, logger, pred_result)

            # output the predict data to csv file
            output = pd.DataFrame({'close':close,'close_truth':close_truth,'open':open,'open_truth':open_truth,'high':high,'high_truth':high_truth,'low':low,'low_truth':low_truth})
            output.to_csv('../data/predict.csv', index=False)
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    con = Config()
    main(con)