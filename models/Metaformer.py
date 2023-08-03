import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constantes import model_dict,get_settings
from models.FEDformer import Model as FEDformer
from models import Autoformer 
import copy 
import os
from utils.constante_skeleton import Liste_set_membre
dict_membre={
    "buste":[0,1,2,3,20],
    "bras_gauche":[4,5,6,7,21,22],
    "bras_droit":[8,9,10,11,23,24],
    "jambe_gauche":[12,13,14,15],
    "jambe_droite":[16,17,18,19],
    
}
liste_membre=Liste_set_membre["3_partie:"]
dict_l_membre={
    "body":liste_membre[0],
    "arm":liste_membre[1],
    "leg":liste_membre[2],
}


def transfo_inverse_NTU(x,liste_membre):
    if x is None:
        return None
    
    xprime=torch.reshape(x,(x.shape[0],x.shape[1],25,3)) # Reshape de la bonne manière
    xprime=xprime[:,:,liste_membre,:] 
    xprime=torch.reshape(xprime,(xprime.shape[0],xprime.shape[1],xprime.shape[2]*xprime.shape[3]))
    return xprime
from collections import OrderedDict
class Model(nn.Module):
 

    def __init__(self, configs,liste_membre:list=liste_membre):
        """
        MON PREMIER NN MODULE
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.liste_membre=liste_membre
        self.modelmembre= FEDformer
        self.liste_modele_partie=nn.ModuleList()
        self.c_out=configs.c_out   
        try:

            self.nom_sous_model=configs.sous_model
        except:
            self.nom_sous_model="FEDformer"
    
        self.path="./checkpoints/"
        configprime=copy.deepcopy(configs)
        for key in dict_l_membre.keys():
            liste_membre=dict_l_membre[key]
            configprime.enc_in=len(liste_membre)*3 # A CHHANGER
            configprime.dec_in=len(liste_membre)*3 #A CHANGER
            configprime.dec_out=len(liste_membre)*3
            configprime.c_out=len(liste_membre)*3
            configprime.data=f"NTU-{key}"
            configprime.model=self.nom_sous_model
        
       
            ajouter=model_dict[self.nom_sous_model].Model(configprime)  
            settingprime=get_settings(configprime)
            sous_model_path=os.path.join(self.path,settingprime,'checkpoint.pth')
            # original saved file with DataParallel
        
            state_dict = torch.load(sous_model_path)
            # create new OrderedDict that does not contain `module.`

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            
            
            ajouter.load_state_dict(new_state_dict)
            self.liste_modele_partie.append(ajouter)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        y=torch.zeros(x_enc.shape[0],self.pred_len,25,3).to(x_enc.device)

        for k in range(len(self.liste_membre)):
            #* Transforme la série temporelle en ne gardant que celle qui nous intéresse 
            xprime=transfo_inverse_NTU(x_enc,liste_membre[k])
            x_mark_enc_prime=transfo_inverse_NTU(x_mark_enc,liste_membre[k])
            x_dec_prime=transfo_inverse_NTU(x_dec,liste_membre[k])
            x_mark_dec_prime=transfo_inverse_NTU(x_mark_dec,liste_membre[k])
            #* On fait tourner le modèle sur la partie qui nous intéresse
            yprime=self.liste_modele_partie[k](xprime,x_mark_enc_prime,x_dec_prime,x_mark_dec_prime)
            #* On remet la série temporelle à la bonne taille
            yprime=yprime.reshape(yprime.shape[0],yprime.shape[1],len(liste_membre[k]),3)
            y[:,:,liste_membre[k],:]=yprime
        y=y.reshape(y.shape[0],y.shape[1],self.c_out)
        return y 
