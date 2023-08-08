import os
import torch


from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM,Metaformer
from models import FEDformer_wavelet


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
        'FEDWav':FEDformer_wavelet,
        'Meta':Metaformer
}
#used in exp_basic. pour des raisons techniques, je n'ai pas pu le déplacer dans les constantes.
class Exp_Basic(object):
    def __init__(self, args):
        """initialise exp_basic qui est utilisé dans tous les exp. Il permet de récupérer les arguments et de construire le modèle

        Parameters
        ----------
        args : argparse ou classe
            ici ce qui est utilisé:
            - args.use_gpu: bool, si on utilise le gpu
            - args.gpu: int, numéro du gpu
            - args.use_multi_gpu: bool, si on utilise plusieurs gpu.
                Si oui, il faut indiquer les numéros de gpus ( essayer (0,1) ou plus si on veut plus de gpus)
                la seul implémentation réelle est nn.DataParallel, donc il faut que le modèle soit compatible avec nn.DataParallel et que pytorch ait bien été build avec le bon cuda
            - args.devices: str, numéros des gpus séparés par des virgules
            - args.model: str, nom du modèle. Il doit être dans le dictionnaire model_dict
        """
        self.args = args
        self.model_dict =model_dict
        self.device = self._acquire_device() 
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    #* Ces fonctions vont être spécifiés dans les autres expériences
    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
