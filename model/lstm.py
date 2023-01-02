import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

random_seed = 6666
torch.manual_seed(random_seed)

"""
LSTM model for time series prediction.

Args:
    Module (torch.nn): torch.nn
"""
class Net(Module):
    """
    Initialize the model.

    Args:
        config (Config): store all the config
    """
    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size,
                             out_features=config.output_size)


    """
    Forward function

    Args:
        x (tensor): input data
        hidden (tuple, optional): hidden state. Defaults to None.

    Returns: linear_out, hidden
        linear_out: output
        hidden: hidden state
    """
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden


"""
Train function

Args:
    config (Config): store all the config
    logger (Logger): logger
    train_and_valid_data (tuple): train and valid data
"""
def train(config, logger, train_and_valid_data):
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    # turn numpy to tensor
    train_X, train_Y = torch.from_numpy(
        train_X).float(), torch.from_numpy(train_Y).float()
    # Set DataLoader and batch_size
    train_loader = DataLoader(TensorDataset(
        train_X, train_Y), batch_size=config.batch_size)

    valid_X, valid_Y = torch.from_numpy(
        valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(
        valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device(
        "cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)

    # check if continue train
    if config.add_train:
        model.load_state_dict(torch.load(
            config.model_save_path + config.model_name))
        logger.info("Load model from {}".format(
            config.model_save_path + config.model_name))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=20, verbose=True)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(max_lr=0.005, steps_per_epoch=len(train_loader), epochs=config.epoch,optimizer=optimizer)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # main loss is MSE, and add MAPE loss
    criterion = torch.nn.MSELoss()

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0

    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()
        train_loss_array = []
        hidden_train = None

        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            optimizer.zero_grad()
            pred_Y, hidden_train = model(_train_X, hidden_train)

            # detach hidden state if not continue train
            if not config.do_continue_train:
                hidden_train = None
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()
                hidden_train = (h_0, c_0)

            # calculate loss which is the weighted sum of MSE and MAPE
            loss = criterion(pred_Y, _train_Y) + 0.3 * \
                torch.mean(torch.abs((pred_Y-_train_Y)/_train_Y))
            #loss = torch.mean(torch.abs((pred_Y-_train_Y)/_train_Y))

            loss.backward()
            optimizer.step()

            train_loss_array.append(loss.item())
            global_step += 1

            # if visualize the train loss per 100 steps
            if config.do_train_visualized and global_step % 100 == 0:
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # early stopping
        model.eval()
        valid_loss_array = []
        hidden_valid = None

        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_valid = model(_valid_X, hidden_valid)

            if not config.do_continue_train:
                hidden_valid = None

            # calculate loss which is the weighted sum of MSE and MAPE
            loss = criterion(pred_Y, _valid_Y) + 0.3 * \
                torch.mean(torch.abs((pred_Y-_valid_Y)/_valid_Y))
            #loss = torch.mean(torch.abs((pred_Y-_valid_Y)/_valid_Y))
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)

        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                    "The valid loss is {:.6f}.".format(valid_loss_cur))

        # if visualize the train loss per epoch
        if config.do_train_visualized:
            vis.line(X=np.array([epoch]), Y=np.array([train_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Train', opts=dict(showlegend=True))
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Eval', opts=dict(showlegend=True))

        # save the best model
        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(),
                       config.model_save_path + config.model_name)
        else:
            bad_epoch += 1
            # if bad epoch is larger than patience, stop training
            if bad_epoch >= config.patience:
                logger.info(
                    " The training stops early in epoch {}".format(epoch))
                break

        # update learning rate
        scheduler.step(valid_loss_cur)


"""
Predict the test data

Args:
    config (Config): the config of the model
    test_X (numpy array): the test data

Returns: result
    result (numpy array): the predict result
"""
def predict(config, test_X):
    # get the test data loader
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # load the model
    device = torch.device(
        "cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)
    model.load_state_dict(torch.load(
        config.model_save_path + config.model_name))

    # init the result
    result = torch.Tensor().to(device)

    # predict
    model.eval()
    hidden_predict = None

    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    # remember to detach the result from the graph
    return result.detach().cpu().numpy()
