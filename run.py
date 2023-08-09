import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import sys
from utils.constantes import get_settings
import copy
from utils.constante_skeleton import dict_set_membre
if __name__ == '__main__':
    
    #print("version de cuda",torch.version.cuda)
    #print("version de cudnn",torch.backends.cudnn.version())
    #print("version de torch",torch.__version__)
    print("nombre de gpu disponible",torch.cuda.device_count())
    #print("version de python",sys.version)
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name,usually long_term_forecast, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status.if 1 the training is done, if 0 we load the checkpoint on setting')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id. The name of your experiment/id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name. See the list on Exp_basic,model_dict, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type. The data you want to use, type NTU for NTU_RGB+D')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file. For NTU it is ./dataset/NTU_RGB+D/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file. for NTU it is numpyed/')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task,for NTU it is M options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task. Do not change it')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding in the first layer of the model, didnt used them, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='length of  labels')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length,must be greater than label_len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size ( so the number of channels input of the model).For NTU it is 75')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size. to me , the same as encoder input size, For NTU it is 75')
    parser.add_argument('--c_out', type=int, default=7, help='output size, For NTU it is 75')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model between encoder layer')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers. Usually 2')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers. Usually 1')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn.')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding. Not used in NTU, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function. Chose wether MSE or MAE')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate. options:[constant, sem_constant, cosine, type1, type2]')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    #** NTU_RGB
    parser.add_argument('--get_time_value', type=int, default=0, help='get time value,0 if not, 1 if yes')
    parser.add_argument('--get_cat_value', type=int, default=0, help='get cat value,0 if not, 1 if yes')
    parser.add_argument('--preprocess', type=int, default=1, help='preprocess data,0 if 1 or more we do sth')
    #parser.add_argument('--sous_model', type=str, default='FEDformer', help='sous-Model pour metaformer')
    #parser.add_argument('--quel_membre', type=str, default='3_partie:', help='quel ensemble de membre pour Metaformer')
    parser.add_argument('--no_test', action='store_true', help='if used, we dont do the test after training', default=False)
    #* challenge test_hypothèse
    parser.add_argument('--split_train_test',type=str,default='action',help='split train test selon action ou au hasard. Possible value: [action,random]')
    #* Augmentation des données 
    parser.add_argument('--augment',default=False, action='store_true', help='use of data augmentation or not')
    parser.add_argument('--prop',type=str,default="1.05,0.05,0.05",help="proportion of dataset size that we will augment. separate the value by ,")
    
    #Si on rerun un modèle à partir d'un checkpoint
    parser.add_argument('--start_checkpoint',default=False,action='store_true',help='ask if we want to restart from a checkpoit')
    parser.add_argument('--setting_start_checkpoint',type=str,default="test",help="if start_checkpoint, ask the name of the last checkpoint settings ")

    args = parser.parse_args()
    #* Verification GPU/MultiGPU
    print("use_gpu: ce quon demande ", args.use_gpu, "cuda est-il disponible:", torch.cuda.is_available(), flush=True)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print("use_gpu: selon l'ordi après ", args.use_gpu, flush=True)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args,flush=True)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training: # training
        for ii in range(args.itr):
            # setting record of experiments
            args.num_itr=ii
            setting = get_settings(args)
            print("batch_size",args.batch_size)   
            exp = Exp(args)  # set experiments
            
            
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting),flush=True)
            if args.model=="Meta":
                ssargs=copy.deepcopy(args)
                ssargs.model=args.sous_model
                liste_sous_membre=dict_set_membre[args.quel_membre]
                for membre in liste_sous_membre:
                    n=len(liste_sous_membre[membre])
                    ssargs.enc_in=n*3
                    ssargs.enc_out=n*3
                    ssargs.c_out=n*3
                    ssargs.data=f"NTU_{membre}"
                    ssargs.model=args.sous_model
                    
                    settingprime=get_settings(ssargs)
                    exp.train(setting=settingprime)

            else:
                exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting),flush=True)
            if args.no_test:
                print(" No test")
                continue
            exp.test(setting)
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
