""" Le but est de recenser ici toutes les constantes dans les programmes :') """
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader,dataset_NTURGBD

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'NTU': dataset_NTURGBD
}
#used in data_factory
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM
from models import FEDformer_wavelet
#used in exp_basic.
model_dict = {
        'TimesNet': TimesNet,
        'Autoformer': Autoformer,
        'Transformer': Transformer,
        'Nonstationary_Transformer': Nonstationary_Transformer,
        'DLinear': DLinear,
        'FEDformer': FEDformer,
        'Informer': Informer,
        'LightTS': LightTS,
        'Reformer': Reformer,
        'ETSformer': ETSformer,
        'PatchTST': PatchTST,
        'Pyraformer': Pyraformer,
        'MICN': MICN,
        'Crossformer': Crossformer,
        'FiLM': FiLM,
        'NonstationaryTransformer':Nonstationary_Transformer,
        'NTS':Nonstationary_Transformer,
        'FEDWav':FEDformer_wavelet
}



#* Exemple de args_default possible pour débug et setting
class Args_default:
    def __init__(self):
        # basic config
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'test'
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
        self.seq_len = 30
        self.label_len = 42
        self.pred_len = 42
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
        self.batch_size = 4
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
args_default=Args_default()

setting_default = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
    args_default.task_name,
    args_default.model_id,
    args_default.model,
    args_default.data,
    args_default.features,
    args_default.seq_len,
    args_default.label_len,
    args_default.pred_len,
    args_default.d_model,
    args_default.n_heads,
    args_default.e_layers,
    args_default.d_layers,
    args_default.d_ff,
    args_default.factor,
    args_default.embed,
    args_default.distil,
    args_default.des, 0)



r"""
Contains a help *Joints* class which maps each Kinect v2 index with its name. Also provides a **connexion_tuples** np
array which contains all neighboring joints.

"""
from enum import IntEnum
import numpy as np


class Joints(IntEnum):
    r"""Maps each Kinect v2 joint name to its corresponding index. See
    https://medium.com/@lisajamhoury/understanding-kinect-v2-joints-and-coordinate-system-4f4b90b9df16 for joints infos.

    """
    SPINEBASE = 0
    SPINEMID = 1
    NECK = 2
    HEAD = 3
    
    SHOULDERLEFT = 4
    ELBOWLEFT = 5
    WRISTLEFT = 6
    HANDLEFT = 7

    SHOULDERRIGHT = 8
    ELBOWRIGHT = 9
    WRISTRIGHT = 10
    HANDRIGHT = 11

    HIPLEFT = 12
    KNEELEFT = 13
    ANKLELEFT = 14
    FOOTLEFT = 15

    HIPRIGHT = 16
    KNEERIGHT = 17
    ANKLERIGHT = 18
    FOOTRIGHT = 19

    SPINESHOULDER = 20

    HANDTIPLEFT = 21
    THUMBLEFT = 22

    HANDTIPRIGHT = 23
    THUMBRIGHT = 24


# shape (n_connexions, 2)
connexion_tuples = np.array([[Joints.SPINEBASE, Joints.SPINEMID],
                             [Joints.SPINEMID, Joints.SPINESHOULDER],
                             [Joints.SPINESHOULDER, Joints.NECK],
                             [Joints.NECK, Joints.HEAD],

                             [Joints.SPINESHOULDER, Joints.SHOULDERLEFT], # 4
                             [Joints.SHOULDERLEFT, Joints.ELBOWLEFT],
                             [Joints.ELBOWLEFT, Joints.WRISTLEFT],
                             [Joints.WRISTLEFT, Joints.HANDLEFT],
                             [Joints.HANDLEFT, Joints.HANDTIPLEFT],
                             [Joints.HANDLEFT, Joints.THUMBLEFT],

                             [Joints.SPINESHOULDER, Joints.SHOULDERRIGHT], # 10
                             [Joints.SHOULDERRIGHT, Joints.ELBOWRIGHT],
                             [Joints.ELBOWRIGHT, Joints.WRISTRIGHT],
                             [Joints.WRISTRIGHT, Joints.HANDRIGHT],
                             [Joints.HANDRIGHT, Joints.HANDTIPRIGHT],
                             [Joints.HANDRIGHT, Joints.THUMBRIGHT],

                             [Joints.SPINEBASE, Joints.HIPRIGHT], # 16
                             [Joints.HIPRIGHT, Joints.KNEERIGHT],
                             [Joints.KNEERIGHT, Joints.ANKLERIGHT],
                             [Joints.ANKLERIGHT, Joints.FOOTRIGHT],

                             [Joints.SPINEBASE, Joints.HIPLEFT], # 20
                             [Joints.HIPLEFT, Joints.KNEELEFT],
                             [Joints.KNEELEFT, Joints.ANKLELEFT],
                             [Joints.ANKLELEFT, Joints.FOOTLEFT]])



# Utilisé dans plot_skeleton je crois et c'est tous

# utilisé dans expr.basic et expr.long_term_forecast
def get_settings(args):
    """ Sachant des args, permets de renvoyer un setting de manière automatiser,
    REMPLACE AUTOMATQIEUEMNT _ PAR - """
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_cv{}_tvv{}'.format(
                args.task_name,
                args.model_id.replace("_","-"),
                args.model,
                args.data.replace("_","-"),
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
                args.des,args.num_itr,
                args.get_cat_value,
                args.get_time_value)

#Fonction ivnerse de get settings


def get_args_from_filename(file,args_inherit=None ):
    """ On va retourner le setting sachant le Filefait à la main
    POUR FED SI CEST NTU IL FAUT RAJOUTER DES TRUCS  """
    parser=file.split("_")
    if args_inherit is None:
        class Args1(Args_technique_CPU):
            def __init__(self):
                super().__init__()
                parser = file.split("_")
        
                self.task_name = "long_term_forecast"
                self.model_id = str(parser[3])
                self.model = str(parser[4])
                self.data = str(parser[5])
                self.features = str(parser[6][2:])
                self.seq_len = int(parser[7][2:])
                self.label_len = int(parser[8][2:])
                self.pred_len = int(parser[9][2:])
                self.d_model = int(parser[10][2:])
                self.n_heads = int(parser[11][2:])
                self.e_layers = int(parser[12][2:])
                self.d_layers = int(parser[13][2:])
                self.d_ff = int(parser[14][2:])
                self.factor = int(parser[15][2:])
                self.embed = str(parser[16][2:])
                self.distil = parser[17][2:] == "True"
                self.des = str(parser[18])
                self.ii = int(parser[19])
                print(parser[18])
                print(parser[20][2:])
                if len(parser[20][2:])==0:
                    #print("frérot t'as encore oublié cv_NOMBRe alors qu'il fallait faire cvN")
                    self.get_cat_value = int(parser[21])
                    self.get_time_value = int(parser[22][3:])
                else:

                    self.get_cat_value = int(parser[20][2:])

                    self.get_time_value = int(parser[21][3:])
    else:
        class Args1(Args_technique_GPU):
            def __init__(self):
                super().__init__()
                parser = file.split("_")
        
                self.task_name = "long_term_forecast"
                self.model_id = str(parser[3])
                self.model = str(parser[4])
                self.data = str(parser[5])
                self.features = str(parser[6][2:])
                self.seq_len = int(parser[7][2:])
                self.label_len = int(parser[8][2:])
                self.pred_len = int(parser[9][2:])
                self.d_model = int(parser[10][2:])
                self.n_heads = int(parser[11][2:])
                self.e_layers = int(parser[12][2:])
                self.d_layers = int(parser[13][2:])
                self.d_ff = int(parser[14][2:])
                self.factor = int(parser[15][2:])
                self.embed = str(parser[16][2:])
                self.distil = parser[17][2:] == "True"
                self.des = str(parser[18])
                self.ii = int(parser[19])
                print(parser[18])
                print(parser[20][2:])
                if len(parser[20][2:])==0:
                    #print("frérot t'as encore oublié cv_NOMBRe alors qu'il fallait faire cvN")
                    self.get_cat_value = int(parser[21])
                    self.get_time_value = int(parser[22][3:])
                else:

                    self.get_cat_value = int(parser[20][2:])

                    self.get_time_value = int(parser[21][3:])
    args=Args1()
    return args 
#* Configuration utilisé pour le test de certaines composition

class Args_technique_CPU():
    def __init__(self):
        # basic config

        self.is_training = 0

        # data loader
        self.data = 'NTU'
        self.root_path = './dataset/NTU_RGB+D/'
        self.data_path = 'numpyed/'

        self.target = 'OT'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'

        # forecasting task

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
        self.moving_avg = 25
        self.dropout = 0
        self.activation = 'gelu'
        self.output_attention = False

        # optimization
        self.num_workers = 10
        self.itr = 1
        self.train_epochs = 10
        self.batch_size = 4
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'test'
        self.loss = 'MSE'
        self.lradj = 'type1'
        self.use_amp = False

        # GPU
        self.use_gpu = False #! A METTRE TRUE POUR DGX 
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'

        # de-stationary projector params
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2
        self.num_itr=0

        # NTU_RGB
        self.ii=0

class Args_technique_GPU():
    def __init__(self):
        # basic config

        self.is_training = 0

        # data loader
        self.data = 'NTU'
        self.root_path = './dataset/NTU_RGB+D/'
        self.data_path = 'numpyed/'

        self.target = 'OT'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'

        # forecasting task

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
        self.moving_avg = 25
        self.dropout = 0
        self.activation = 'gelu'
        self.output_attention = False

        # optimization
        self.num_workers = 10
        self.itr = 1
        self.train_epochs = 10
        self.batch_size = 4
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'test'
        self.loss = 'MSE'
        self.lradj = 'type1'
        self.use_amp = False

        # GPU
        self.use_gpu = True #! A METTRE TRUE POUR DGX 
        self.gpu = 0
        self.use_multi_gpu = True
        self.devices = '0,1'

        # de-stationary projector params
        self.p_hidden_dims = [256,256]
        self.p_hidden_layers = 2
        self.num_itr=0

        # NTU_RGB
        self.ii=0
