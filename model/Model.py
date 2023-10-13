import os
import torch
import logging
import datetime
import builtins
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils.earlystopping import EarlyStopping
from model.Encode2Decode import Encode2Decode

class Model():
    def __init__(self, config, era5, model_save_folder, loading_path = None, set_device=4):
        self.device = torch.device(str(config['model']['device']))
        self.model = Encode2Decode(config).to(self.device)
        self.criterion = nn.MSELoss()
        self.epochs = int(config['model']['epochs'])
        self.patient = int(config['model']['patient'])
        self.output = int(config['model']['output_len'])
        self.out_channel = int(config['model']['out_channel'])
        self.height = int(config['data']['height'])
        self.width = int(config['data']['width'])
        self.optim = optim.Adam(self.model.parameters(), lr=float(config['model']['learning_rate']))
        self.era5 = era5
        self.model_save_folder = model_save_folder

    def setup_logger(self, model_save_folder):
        
        level =logging.INFO
    
        log_name = 'model.log'
    
        fileHandler = logging.FileHandler(os.path.join(model_save_folder, log_name), mode = 'a')
        fileHandler.setLevel(logging.INFO)
    
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)
    
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
    
        logger = logging.getLogger(model_save_folder + log_name)
        logger.setLevel(level)
    
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
    
        self.logger = logger

    def load(self, model_save_folder):
        model_save_path = '{}/best_val.pth'.format(model_save_folder)
        checkpoint = torch.load(model_save_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['min_loss']
    
    def train(self, train_dataset, valid_dataset, start_epoch=0, min_loss = 1e9):
        n_total_steps = len(train_dataset)
        train_losses = []
        early_stopping = EarlyStopping(patience=self.patient, verbose=True)
        avg_train_losses = []
        avg_valid_losses = []

        for i in range(start_epoch, self.epochs):
            self.logger.info('epochs [{}/{}]'.format(i + 1, self.epochs))
            losses, val_losses = [], []
            self.model.train()
            epoch_loss = 0.0
            val_epoch_loss = 0.0
            starttime = datetime.datetime.now()
            for data in tqdm(train_dataset):
                x_idx, y_idx = data
                x_train = torch.Tensor(self.era5.train_validate_data[x_idx],).to(self.device)
                y_train = torch.Tensor(self.era5.train_validate_data[y_idx],).to(self.device)
                pred_train = self.model(x_train, y_train)
                loss = self.criterion(pred_train[:,:,[0,1,3]], y_train[:,:,[0,1,3]])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_losses.append(loss.item())
                epoch_loss += loss.item()

            with torch.no_grad():
                self.model.eval()
                for data in tqdm(valid_dataset):
                    x_idx, y_idx = data
                    x_val = torch.Tensor(self.era5.train_validate_data[x_idx],).to(self.device)
                    y_val = torch.Tensor(self.era5.train_validate_data[y_idx],).to(self.device)
                    pred_val = self.model(x_val, y_val, teacher_forcing_rate = 0)
                    loss = self.criterion(pred_val[:,:,[0,1,3]], y_val[:,:,[0,1,3]])
                    val_losses.append(loss.item())
                    val_epoch_loss += loss.item()
            
            endtime = datetime.datetime.now()
            train_loss = np.average(train_losses)
            valid_loss = np.average(val_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            self.logger.info('cost:{} train loss {}, valid loss {}'.format(
                (endtime-starttime).seconds, np.mean(train_loss), np.mean(valid_loss)))
            
            if min_loss > valid_loss:
                checkpoint = {
                    'epoch': i+1, # next epoch
                    'model': self.model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                    'min_loss': valid_loss,
                }

                torch.save(checkpoint, self.model_save_folder + '/best_val.pth')
                save_model_message = "save model to {} successfully".format(self.model_save_folder)
                self.logger.info(save_model_message)
                min_loss = valid_loss
    
#             Uncomment for applying early stopping
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
    
    def evaluate_test(self, test_dataset, model_save_path, builtins_print=False):

        if builtins_print == False:
            print = self.logger.info
        else:
            print = builtins.print
        
        prediction = []
        test_groundtruth = []
        self.model.load_state_dict(torch.load(model_save_path)['model'])

        mae = np.zeros((self.output, self.out_channel, self.height, self.width))
        mse = np.zeros((self.output, self.out_channel, self.height, self.width))
        counter = 0
    
        with torch.no_grad():
            self.model.eval()
            for data in tqdm(test_dataset):
                x_idx, y_idx = data
                x = torch.Tensor(self.era5.test_data[x_idx],).to(self.device)
                y = torch.Tensor(self.era5.test_data[y_idx],).to(self.device)
                
                pred = self.model(x, y, teacher_forcing_rate = 0) 
                
                pred = pred.cpu().detach().numpy()
                target = y.cpu().detach().numpy()
                slice_data_pred = []
                slice_data_target = []
                for i in range(y.shape[2]):
                    if i!=2:
                        slice_data_pred.append(self.era5.train_nomalizer[i].reverse(pred[:, :, i]))
                        slice_data_target.append(self.era5.test_nomalizer[i].reverse(target[:, :, i]))
                        
                pred_reverse = np.stack(slice_data_pred, axis=2)
                target_reverse = np.stack(slice_data_target, axis=2)
                
                for _b in range(target_reverse.shape[0]):
                    for _t in range(target_reverse.shape[1]): 
                        for _c in range(target_reverse.shape[2]):
                            mae[_t, _c] += np.abs(target_reverse[_b, _t, _c].astype(np.float32) - pred_reverse[_b, _t, _c])
                            mse[_t, _c] += (target_reverse[_b, _t, _c].astype(np.float32) - pred_reverse[_b, _t, _c])**2

                counter += target.shape[0]
        
        timestep = target.shape[1]

        _rh_maes=[np.mean(mae[i, 0])/counter for i in range(0, timestep)]
        _2mt_maes=[np.mean(mae[i, 1])/counter for i in range(0, timestep)]
        _wind_maes=[np.mean(mae[i, 2])/counter for i in range(0, timestep)]

        _rh_mses=[np.mean(mse[i, 0])/counter for i in range(0, timestep)]
        _2mt_mses=[np.mean(mse[i, 1])/counter for i in range(0, timestep)]
        _wind_mses=[np.mean(mse[i, 2])/counter for i in range(0, timestep)]

        _rh_rmses=[np.sqrt(np.mean(mse[i, 0])/counter) for i in range(0, timestep)]
        _2mt_rmses=[np.sqrt(np.mean(mse[i, 1])/counter) for i in range(0, timestep)]
        _wind_rmses=[np.sqrt(np.mean(mse[i, 2])/counter) for i in range(0, timestep)]

        maes = [_rh_maes, _2mt_maes, _wind_maes]
        mses = [_rh_mses, _2mt_mses, _wind_mses]
        rmses = [_rh_rmses, _2mt_rmses, _wind_rmses]

        values = ['relative humidity', '2m temperature', '10m wind']

        timestep = len(maes[0])

        for i in range(len(values)):
            for j in range(timestep):
                evluate_message = 'preditive step {:d} for {} mae:{:.3f}'.format(j+1, values[i], maes[i][j])
                print(evluate_message)

        for i in range(len(values)):
            for j in range(timestep):
                evluate_message = 'preditive step {:d} for {} mse:{:.3f}'.format(j+1, values[i], mses[i][j])
                print(evluate_message)

        for i in range(len(rmses)):
            for j in range(timestep):
                evluate_message = 'preditive step {:d} for {} rmse:{:.3f}'.format(i+1, values[i], rmses[i][j])
                print(evluate_message)

        for i in range(len(values)):
            evluate_message = 'For mean {}, mae:{:.3f}, mse:{:.3f}, rmse:{:.3f}'.format(
                values[i],
                np.mean(maes[i]),
                np.mean(mses[i]),
                np.mean(rmses[i])
            )
            print(evluate_message)