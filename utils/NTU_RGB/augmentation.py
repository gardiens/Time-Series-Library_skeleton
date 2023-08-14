""" Fichier qui va coder toutes les augmentations possibles
Les fonctiosn actuellement implémentés sont Randmix,augment,  et des rotations de points
 """

import os
from typing import Any
import torch
import torch as t 
if t.cuda.is_available():
    try:
        import cupy as np
    except:
        import numpy as np

import copy 

class load_data(object):
    def __init__(self,input_len:int=30,output_len:int=30,data_path='./dataset/NTU_RGB+D/numpyed/',file_extension='.skeleton.npy',get_cat_value=True,get_time_value=False,categorical_columns=['nbodys', 'actor', 'acti', 'camera', 'scene', 'repet']):
        pass
    def __call__(self, row):
         # data est de la frome [nb_frames,nb_joints,3] à la fin 
        mat_path=os.path.join(self.data_path,row['filename']+self.file_extension) #! WARNING ON THE EXTENSION OF THE .NPY
        #* On récupère la valeur du body intéressant
        num_body=row['num_body']
        data=np.load(mat_path,allow_pickle=True).item()[f'skel_body{int(num_body)}'] #* C'est une matrice de la forme [frames,nb_joints,3]
        ind_debut=debut_frame=int(row['debut_frame']) #! 
        return data, ind_debut 




class slice_data(object):
    def __init__(self,membre="all",liste_membre=[],nom_debut_frame="debut_frame",input_len=16,output_len=32):
        self.membre=membre
        self.liste_membre=liste_membre 
        pass

    def __call__(self,data,debut):
        #Debut est l'indice de début de la séquence.
        
        #* On récupère le début et la fin de la séquence
        # data est de la frome [nb_frames,nb_joints,3]
        begin=data[debut:debut+self.input_len]
        label=data[debut:debut+self.output_len]#* On prend les output_len suivantes
        #! A CHANGER ICI 
        reference=20 # Correspond à Spine shoulder ! 
        mean=np.mean(begin[:,reference,:],axis=0)
        begin=begin-mean # On recentre le squelette par rapport à la frame de référence
        label=label-mean
        #! Détail technique: à priori les données sont de la formes [nb_frames,nb_joints,3] mais les réseauxde neurones acceptent un format [nb_frames,nb_features] donc on va faire un reshape
        return begin,label
    


#!!!!
class rotate_data(object):
    """Fonction de base pour dataset_augmenter. Il va faire une rotation des données de +- 17 degrés sur l'axe x et y et z

    Parameters
    ----------
    object : _type_
        _description_

    cette opération peut être relativement longue
    """    
    """ S'occupe de la rotation des matrices d'entrée/sorties d'un angle théta qui est sur l'axe x/y
     Relativement long! """
    """ issu de https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7450165&casa_token=eZoq-Qw-lnsAAAAA:1W-2G3buqD3yElf70CnF8biljnTEKIL0af1x_O9Ujhd6Y0RAy0NDzjgxxMKmucjGVFkTfAVh5fqx&tag=1"""
    def __init__(self,up=-17,down=17) -> None:
        """définit rotate_data. Il va faire une rotation des données de entre up et down en degré sur l'axe x et y et z

        Parameters
        ----------
        up : int, optional
            borne supérieur de la rotation, by default -17
        down : int, optional
            borne inférieure de la rotation, by default 17
        """        
        self.up=up*np.pi/180
        self.down=down*np.pi/180
    def __call__(self,mat) -> Any:
        """fonction qui va faire la rotation des données

        Parameters
        ----------
        mat : np.array ou torch.tensor
            matrice en entrée auquel on va faire l'opération

        Returns
        -------
        array
            np.array. matrice de sortie auquel on a fait la rotation
        """        
        """ mat supposé de la forme (nb_frames,nb_joints,3)"""
        #theta,alpha,gamma= torch.rand(3)
        """ Calcul normalement RxRyRz(mat) """
        theta,alpha,gamma=torch.rand([3])
        if type(mat).__module__=='numpy':
            mat=torch.from_numpy(mat.copy())
        
        return torch.einsum("ab,deb -> dea",
                            self.Rx(theta),
                            torch.einsum("ab,deb -> dea",
                                         self.Ry(alpha),
                                         torch.einsum("ab,deb -> dea",
                                                      self.Rz(gamma),mat.float())),
                                         ).numpy() #!


    """ Les matrices sont donnée en radian et pas degré ! """
    def Rx(self,theta):
        return torch.from_numpy(np.array(
            [
                [1,0,0],
                [0,np.cos(theta),-np.sin(theta)],
                [0,np.sin(theta),np.cos(theta)],
            ]
        )
        ).float()
    def Ry(self,theta):
        return torch.from_numpy(np.array([
            [np.cos(theta),0,np.sin(theta)],
             [0,1,0],
             [-np.sin(theta),0,np.cos(theta)]
             
             ]
             )
        ).float()
    def Rz(self,theta):
        return torch.from_numpy(np.array([
            [np.cos(theta),-np.sin(theta),0],
            [np.sin(theta),np.cos(theta),0],
            [0,0,1]
        ]
            )
        ).float()
        


class Mixup(object):
    haut=np.arange(0,12,step=1)
    bas=np.arange(13,25,step=1) #!de 20 à 24 c'est le buste  ???
    def __init__(self) -> None:
        
        
        pass

    def __call__(self,mat1,mat2) -> Any:
        """ mixup sur les deux matrices, i.e il va prendre les duex matrices et renvoyer une nouvelle matrice avec lehaut de mat1 et le bas de mat2"""
        result= np.concatenate((mat1[:,:13,:],mat2[:,13:,:]),axis=1)
        
        return result  

class CutMix(object):
    """On prend deux matrices et on renvoie une autre matrice qui est une fusion des deux aléatoires! """
    def __init__(self,prob=0.5) -> None:
        self.prob=prob
        pass

    def __call__(self,mat1,mat2) -> Any:
        # Sélectionne  les indices qu'on va êchangés parmis 25 joints


        mask = torch.bernoulli(torch.full(size=(mat1.shape[1],),fill_value=self.prob)).int() #Full rempli une matrice de taille mat1.shape ( nb_joints) de prob
        # Bernouilli  renvoie 0 ou 1 en fonction de la proba d'entrée.
        reverse_mask = torch.ones((mat1.shape[1],)).int() - mask
        mat1,mat2=torch.from_numpy(mat1.copy()),torch.from_numpy(mat2.copy())
        return torch.einsum("abc,b-> abc",mat1,mask).numpy()+ torch.einsum("abc,b ->abc",mat2,reverse_mask).numpy()
    


class backward(object):
    def __init__(self) -> None:
        pass
    def __call__(self,mat):
        #print("oui?",type(mat))
        #print("on test sur backward",torch.from_numpy(mat[::-1,:,:]).shape)
        return np.flip(mat,axis=0).copy()