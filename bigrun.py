#*  args used for real script
from utils.constantes import get_settings
import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

import random
import time
import sys
from utils.constantes import get_settings
import copy
from utils.constante_skeleton import dict_set_membre
class Args:
    def __init__(self):
        # basic config
        self.task_name = 'long_term_forecast'
        self.is_training = 0
        self.model_id = 'AA'
        self.model = 'aaa'
        self.num_itr=0
        # data loader
        self.data = 'NTU'
        self.root_path = './dataset/NTU_RGB+D/'
        self.data_path = 'numpyed/'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'

        # forecasting task
        self.seq_len = 16
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
        self.factor = 3
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeNTU'
        self.activation = 'gelu'
        self.output_attention = False

        # optimization
        self.num_workers = 10
        self.itr = 1
        self.train_epochs = 14
        self.batch_size = 256
        self.patience = 3
        self.learning_rate = 10**(-3)
        self.des = 'Exp'
        self.loss = 'MSE'
        self.lradj = 'sem_constant'
        self.use_amp = False

        # GPU
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = True
        self.devices = '0,1'

        # de-stationary projector params
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2

        # NTU_RG
        self.get_time_value = 1
        self.get_cat_value = 0
        self.ii=0
        self.preprocess=1
        self.split_train_test="action"
        self.start_checkpoint=False
        self.augment=False
        self.no_test=True
args=Args()

setting=get_settings(args)


from exp.exp_basic import model_dict

#* Verification GPU/MultiGPU
print("use_gpu: ce quon demande ", args.use_gpu, "cuda est-il disponible:", torch.cuda.is_available(), flush=True)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print("use_gpu: selon l'ordi après ", args.use_gpu, flush=True)
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]



Exp = Exp_Long_Term_Forecast
for model in model_dict:
    print(" On s'occupe de ?")
    args.model=model
    args.model_id="15-08"+model
    if  model=='Meta':
        continue

    if args.is_training: # training
        for ii in range(args.itr):
            # setting record of experiments
            args.num_itr=ii
            setting = get_settings(args)
            print("batch_size",args.batch_size)   
            exp = Exp(args)  # set experiments
            
            
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting),flush=True)
         

        
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting),flush=True)
            if args.no_test:
                print(" No test")
                continue
            exp.test(setting)
            time.sleep(1800)
            torch.cuda.empty_cache()
            
    else:
        print(" On test directement les données:")
        ii = 0
        args.num_itr=ii
        setting = get_settings(args)
        print(setting)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting),flush=True)
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
