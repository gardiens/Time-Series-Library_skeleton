"""Fichier qui a juste pour but d'éffectuer un feedForward sur un signal en FFT. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
import torch.nn.functional as F

"""
class Model(nn.Module):


    def __init__(self, configs, individual=False):

        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear= nn.ModuleList()


            for i in range(self.channels):
                self.Linear.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        pass 


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            X=torch.fft(x_enc)
            for i in range(self.channels):
                X[:,i,:]=self.Linear[i](X[:,i,:])
                X[:,i,:]=self.activation_function(X[:,i,:])
            
            X=torch.ifft(X)
            return X 
                
        return None"""


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.input_len*self.enc_in,120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Max pooling over a (2, 2) window
        x= torch.fft(x,dim=1,norm='forward')
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





