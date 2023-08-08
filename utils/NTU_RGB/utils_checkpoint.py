""" Permet de load des checkpoints"""
import torch 
import os 
from collections import OrderedDict
from utils.constantes import get_settings,get_args_from_filename
def load_checkpoint(model,setting,checkpoint_path="./checkpoints/",args):
    
    if "Meta" not in setting:
        if torch.cuda.is_available():
                    
                    state_dict = torch.load(os.path.join(checkpoint_path + setting, 'checkpoint.pth'))
        else:
            state_dict = torch.load(os.path.join(checkpoint_path+ setting, 'checkpoint.pth'),map_location=torch.device('cpu'))
        # create new OrderedDict that does not contain `module.`

        new_state_dict = OrderedDict()
        for k, v in state_dict.items(): #* On va remove module, sinon on a une erreur de python
            
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        
        try: #* On load le checkpoint
                model.load_state_dict(new_state_dict)
        except FileNotFoundError: #* Cas particulier où get_cat_value n'était pas encore défini
            args1=get_args_from_filename(setting)
            args1.get_cat_value="_"+str(args1.get_cat_value)
            setting1=get_settings(args1)
            if torch.cuda.is_available():
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(new_state_dict)