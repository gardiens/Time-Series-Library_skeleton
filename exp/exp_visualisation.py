from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings

if torch.cuda.is_available(): 
    try:
        import cupy as np 
    except:
        import numpy as np 
        print("on importe numpy malgré un GPU?")
else:
    import numpy as np 
    
from setuptools import distutils
import distutils.version
from utils.NTU_RGB.plot_skeleton import plot_video_skeletons,plot_skeleton
from utils.constantes import get_settings,get_args_from_filename
from torch.utils.tensorboard import SummaryWriter
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.NTU_RGB.utils_checkpoint import load_checkpoint
from collections import OrderedDict
warnings.filterwarnings('ignore')

from utils.NTU_RGB.utils_dataset import show_grads
from exp.exp_long_term_forecasting import *


class Exp_viz(Exp_Long_Term_Forecast):
        def __init__(self, args):
            super(Exp_viz, self).__init__(args)
            setting = get_settings(args)
                    
            self.setting=setting
            self.writer=SummaryWriter(log_dir=f"runs/{setting}") #* Permets de setup tensorboard
            writer=self.writer
            #print("les args",vars(args))
            #add_hparams(self.writer,args) inutilisé car il ne semblait pas réussi àfaire fonctionner le hparmars mais dédouble les expériences.

            # Add in the writer every parameter which are float,int,str or bool
            #writer.add_hparams({k:v for k,v in vars(args).items() if type(v) in [float,int,str,bool]},{})
            #"writer.flush()
    

        