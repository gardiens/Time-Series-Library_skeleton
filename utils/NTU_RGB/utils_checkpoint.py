""" Permet de load des checkpoints"""
import torch 
import os 
from collections import OrderedDict
from utils.constantes import get_settings,get_args_from_filename
def load_checkpoint(model,setting,args,checkpoint_path="./checkpoints/"):
    """load un model sachant un setting. Le checkpoint est sotcké dans checkpoint_path+setting.

    Parameters
    ----------
    model : Torch.module
        modèle dont on veut load le checkpoint
    setting : str
        settings d'un modèle
    args : args
        obtenu par run.py oupar classe
    checkpoint_path : str, optional
        path du checkpoint, à priori fixe dans toute l'implémentation, by default "./checkpoints/"
    """
    print("load",args.model_id)
    if "Meta" not in setting:
        if torch.cuda.is_available():
                    
                    state_dict = torch.load(os.path.join(checkpoint_path + setting, 'checkpoint.pth'))
        else:
            state_dict = torch.load(os.path.join(checkpoint_path+ setting, 'checkpoint.pth'),map_location=torch.device('cpu'))
        # create new OrderedDict that does not contain `module.`
        try:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items(): #* On va remove module, sinon on a une erreur de python
                
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            
            try: #* On load le checkpoint
                    model.load_state_dict(new_state_dict)
            except FileNotFoundError: #* Cas particulier où get_cat_value n'était pas encore défini
                args1=get_args_from_filename(setting,args)
                args1.get_cat_value="_"+str(args1.get_cat_value)
                setting1=get_settings(args1)
                if torch.cuda.is_available():
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(new_state_dict)
        except: 
            model.load_state_dict(state_dict)