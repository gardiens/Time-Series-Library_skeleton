import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.FEDformer import *
#* Exemple de args possible pour débug et setting
# récupération d'un batch
from data_provider.data_factory import data_provider
if __name__ == '__main__':
    class Args:
        def __init__(self):
            # basic config
            self.task_name = 'long_term_forecast'
            self.is_training = 1
            self.model_id = 'testDEBUGAGGE'
            self.model = 'Autoformer'

            # data loader
            self.data = 'NTU'
            self.root_path = './dataset/NTU_RGB+D/'
            self.data_path = 'numpyed/'
            self.features = 'M'
            self.target = 'OT'
            self.freq = 'h'
            self.checkpoints = './checkpoints/'

            # forecasting task
            self.seq_len = 32
            self.label_len = 32
            self.pred_len = 32
            self.seasonal_patterns = 'Monthly'

            # inputation task
            self.mask_rate = 0.25

            # anomaly detection task
            self.anomaly_ratio = 0.25

            # model define
            self.top_k = 5
            self.num_kernels = 6
            self.enc_in = 75
            self.dec_in = 75
            self.c_out = 75
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 2048
            self.moving_avg = 25
            self.factor = 1
            self.distil = True
            self.dropout = 0.1
            self.embed = 'timeNTU'
            self.activation = 'gelu'
            self.output_attention = False

            # optimization
            self.num_workers = 10
            self.itr = 1
            self.train_epochs = 10
            self.batch_size = 2
            self.patience = 3
            self.learning_rate = 0.0001
            self.des = 'test'
            self.loss = 'MSE'
            self.lradj = 'type1'
            self.use_amp = False

            # GPU
            self.use_gpu = False
            self.gpu = 0
            self.use_multi_gpu = False
            self.devices = '0'

            # de-stationary projector params
            self.p_hidden_dims = [128, 128]
            self.p_hidden_layers = 2

            # NTU_RGB
            self.get_time_value = True
            self.get_cat_value = False
            self.ii=0
            self.preprocess=1
    args=Args()


    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_cv_{}_tvv{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, 
                args.ii,
                args.get_cat_value,
                args.get_time_value)

    setting
    from models.Metaformer import *
    data_set, data_loader=data_provider(args,flag="train") 
    sample_name:str="S001C001P001R001A001"
    entry=data_set.get_data_from_sample_name(sample_name)
    entry_model=data_set.get_input_model(entry)
    network=Modelee(args)
    print("recuperation donnée")
    (batch_x, batch_y, batch_x_mark, batch_y_mark)=enumerate(data_loader).__next__()[1]
    network.float()
    print("on rentre dans le dur")
    y=network(batch_x.float(), batch_x_mark.float(),None, batch_y_mark.float())
    print(y)