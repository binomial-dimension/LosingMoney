import pandas as pd
import numpy as np

from config import Config
from data import Data
from utils import load_logger,draw
from lstm import train, predict

frame = 'pytorch'

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

            # get predict train data for future use
            train_X, train_Y = data_gainer.get_noshuffle_train_data()
            predict_norm_data = predict(config, train_X)
            train_X = train_X.reshape(-1,train_X.shape[2])
            origin_data = data_gainer

            # denormalize the data
            predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]
            label_data = train_Y * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]
            close_truth,close,open_truth,open,high_truth,high,low_truth,low =  label_data[:, 0], predict_data[:, 0],label_data[:, 1], predict_data[:, 1],label_data[:, 2], predict_data[:, 2],label_data[:,3],predict_data[:,3]
            
            # output the predict data to csv file
            output = pd.DataFrame({'close':close,'close_truth':close_truth,'open':open,'open_truth':open_truth,'high':high,'high_truth':high_truth,'low':low,'low_truth':low_truth})
            output.to_csv('../data/predict_train.csv', index=False)
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    con = Config()
    main(con)