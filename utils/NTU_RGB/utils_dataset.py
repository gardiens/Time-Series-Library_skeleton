""" Everything usefull while parsing some NTU dataset ! """

import os
import numpy as np
import pandas as pd
from torch import tensor
import ruptures as rpt  # Package for changepoint detection


def normaliser_input_unpoint(x,y,point_ref:int=None):
    """ le but est de normaliser par rapport à un point le squelette pour être plus stable"""
    """ x,y sont de la forme [nb_frames,nb_joints,3]"""
    if point_ref is None:
        point_ref=20 #Spine Shoulder
    
    mean_x=np.mean(x[:,point_ref,1],axis=0)
    mean_y= np.mean(x[:,point_ref,2],axis=0)
    mean_z= np.mean(x[:,point_ref,2],axis=0)
    x=x[:,:,1]-mean_x
    x=x[:,:,2]-mean_y
    x=x[:,:,3]-mean_z
    y=y[:,:,1]-mean_x
    y=y[:,:,2]-mean_y
    y=y[:,:,3]-mean_z
    return x,y
class time_serie_NTU:

    """ Classe de Time_series qui va être la base de notre dataset"""
    """ File to display the lementary objects used in Dataset. """


    def __init__(self,input_len:int=30,output_len:int=30,data_path='./dataset/NTU_RGB+D/numpyed/',file_extension='.skeleton.npy',get_cat_value=True,get_time_value=False,categorical_columns=['nbodys', 'actor', 'acti', 'camera', 'scene', 'repet'],preproccess:int=1) -> None:
        """_summary_

        Parameters
        ----------
        row : df
            un row d'un pandas dataframe qui contient les informations d'un squelettes
         seq_len : int, optional
            longueur de la séquence d'entrée, by default 30
        out_len : int, optional
            longueur de la séquence de sortie, by default 30
        data_path : str, optional
            path qui contient tous les .npy des skeletons , by default './dataset/NTU_RGB+D/numpyed/'

        file_extension : str, optional
            _description_, by default '.skeleton.npy'
        get_cat_value : bool, optional
            true si on veut les valeurs catégoriques, by default True
        get_time_value : bool, optional
            renvoie en plus un array avec le différents temps, by default False
        categorical_columns : list, optional
            ensembles des données catégoriques que l'on veut garder ou non, by default ['nbodys', 'actor', 'acti', 'camera', 'scene', 'repet']
        """

        self.input_len=input_len
        self.output_len=output_len

        self.data_path=data_path
        self.file_extension=file_extension
        self.get_cat_value=get_cat_value
        self.categorical_columns=categorical_columns
        self.get_time_value=get_time_value
        self.intervalle_frame=1/30 #* On suppose que c'est du 30 fps
        self.preproccess=preproccess
    def __len__(self) -> int:
        return len(self.row)
    
    
    def get_data(self,row,preprocessing:int=1):
        ''' FONCTION UTILISE DANS LES DATASET/DATALOADER
        Renvoie une sortie de la forme :
        entry_data, label, cat_data, time_value si les valeurs sont bien bonne
        entry_data est de la forme (nb_frames,nb_joints nb_dim ( 3 ici))'''
        mat_path=os.path.join(self.data_path,row['filename']+self.file_extension) #! WARNING ON THE EXTENSION OF THE .NPY
        data=np.load(mat_path,allow_pickle=True).item()
        #* On récupère la valeur du body intéressant
        num_body=row['num_body']
        data=np.load(mat_path,allow_pickle=True).item()[f'skel_body{int(num_body)}'] #* C'est une matrice de la forme [frames,nb_joints,3]
        
        debut_frame=int(row['debut_frame'])
        #print(debut_frame)
        #* On récupère le début et la fin de la séquence
        # data est de la frome [nb_frames,nb_joints,3]
        debut=debut_frame  
        begin=data[debut:debut+self.input_len]
        label=data[debut:debut+self.output_len]#* On prend les output_len suivantes
        #! A CHANGER ICI 
        reference=20 # Correspond à Spine shoulder ! 
        mean=np.mean(begin[:,reference,:],axis=0)
        begin=begin-mean # On recentre le squelette par rapport à la frame de référence
        label=label-mean
        #! Détail technique: à priori les données sont de la formes [nb_frames,nb_joints,3] mais les réseauxde neurones acceptent un format [nb_frames,nb_features] donc on va faire un reshape
        begin=begin.reshape(begin.shape[0],-1)
        label=label.reshape(label.shape[0],-1)
        if preprocessing:
            #* On normalise les données par rapport à la première frame 
            begin=begin#-np.mean(begin,axis=0)
            label=label#-np.mean(begin,axis=0)
        #* Maintenant , begin est de la forme( nb_frames,nb_joints*3) et label est de la forme (nb_frames,nb_joints*3
        if self.get_time_value:
            time_value_enc=np.arange(debut_frame,debut_frame+self.input_len)*self.intervalle_frame #* Encoding du temps, représente x_mark_enc pour FEDformers
            time_value_dec=np.arange(debut_frame,debut_frame+self.output_len)*self.intervalle_frame #* Decoding du temps entre deux frames , représente x_mark_dec pour FedFormers
            time_value_enc=np.tile(time_value_enc,(begin.shape[1],1)).transpose((1,0)) #* On fait un tile pour avoir la même valeur pour chaque joint]))
            time_value_dec=np.tile(time_value_dec,(label.shape[1],1)).transpose((1,0)) #* On fait un tile pour avoir la même valeur pour chaque joint]))
        if self.get_cat_value:
            mat_cat_data=row[self.categorical_columns].values #* C'est un np.array avec les différentes 
            #* change the type to be float64 , MAY BE BUGGY HERE
            mat_cat_data=mat_cat_data.astype(np.float64)

        
        #*  renvoie la solution de la bonne forme ! 
        if self.get_time_value and self.get_cat_value:
            return begin,label,time_value_enc,time_value_dec,mat_cat_data
    
        if self.get_time_value:
            
                
              
            return begin,label,time_value_enc,time_value_dec
        if self.get_cat_value:
            return begin,label,mat_cat_data
        else:
            return begin,label
    def inverse_transform(self,x,entry=None,preprocessing=1):
        """Renvoie les données dans le bon format pour pouvoir les afficher
        renvoie de la forme [nb_frames,nb_joints,3]"""
        if not preprocessing:
            return x.reshape(x.shape[0],int(x.shape[1]//3),3)
        else:
            return (x).reshape(x.shape[0],int(x.shape[1]//3),3)

    def get_input_model(self,entry):
        """ depuis un X obtenu de get_data ou get_data_from_sample_name, renvoie un input de la bonne forme pour le modèle"""
        
        if self.get_time_value and self.get_cat_value:
            begin,label,time_value_enc,time_value_dec,mat_cat_data=entry
    
        elif self.get_time_value:
             begin,label,time_value_enc,time_value_dec=entry
        elif self.get_cat_value:
             begin,label,mat_cat_data=entry
        else:
             begin,label=entry 
        begin=tensor(np.expand_dims(begin,axis=0)).float()
        label=tensor(np.expand_dims(label,axis=0)).float()
        if self.get_time_value and self.get_cat_value:
            time_value_enc=tensor(np.expand_dims(time_value_enc,axis=0)).float()
            time_value_dec=tensor(np.expand_dims(time_value_dec,axis=0)).float()
            mat_cat_data=tensor(np.expand_dims(mat_cat_data,axis=0)).float()
            return begin,label,time_value_enc,time_value_dec,mat_cat_data
        if self.get_time_value:
            time_value_enc=tensor(np.expand_dims(time_value_enc,axis=0)).float()
            time_value_dec=tensor(np.expand_dims(time_value_dec,axis=0)).float()
            return begin,label,time_value_enc,time_value_dec
        if self.get_cat_value:
            mat_cat_data=tensor(np.expand_dims(mat_cat_data,axis=0)).float()
            return begin,label,mat_cat_data
        else:
            return begin,label




       
    
 
import re
def extract_integers(file_name):
    """ Extract the parameters from the scene file name."""
    file_name=file_name.split(".")[0]
    pattern = r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})'
    match = re.match(pattern, file_name)
    if match:
        integers = [int(match.group(i)) for i in range(1, 6)]
        return np.array(integers)
    else:
        return np.array([])




    

import ruptures as rpt
def summary_csv_NTU(path_data_npy:str='./dataset/NTU_RGB+D/numpyed/',path_csv:str='./dataset/NTU_RGB+D/summary_NTU/',name_csv="summary_NTU.csv",beta=3.6,preproccess=1):
    """créer un .csv qui permet à partir des differents .npy de résumer les valeurs importantes

    Parameters
    ----------
    path_data_npy : str, optional
       path vers le dossier où sont stocker les .npy, by default './dataset/NTU_RGB+D/numpyed/'
    path_csv : str, optional
        path où on va stocker le .csv, by default './dataset/NTU_RGB+D/summary_NTU/'
    name_csv : str, optional
        nom du .csv qu'on va sauvegarder, by default "summary_NTU.csv"

    Returns
    -------
    df  
        le dataframe correspondant au .csv
    path_csv
        le path du .csv
    """
    model=rpt.KernelCPD(kernel="linear",min_size=2)
    print('----création d un .csv résumant les données NTU RGB+D---',flush=True)
    if not  os.path.exists(path_csv):
        os.mkdir(path_csv)
    # Les listes qu'on va transférer ensutie en dataframe 
    result=[]
    result_nb_frames=[]
    result_debut_frame=[]
    result_mean=[]
    result_std=[]
    result_l_chp=[]
    result_n_chp=[]
    maxnbodys=0
    
    for file in os.listdir(path_data_npy): # Pour chaque fichier dans le dossier

        if file.endswith('.npy'): # On traite uniquement les .npy
            file_path=os.path.join(path_data_npy,file)
            data=np.load(file_path,allow_pickle=True).item()
            # dict avec comme colonnes dict_keys(['file_name', 'nbodys', 'njoints', 'skel_body0'])
            filename=data['file_name']
            nbodys=int(max(data['nbodys']))
            njoints=int(data['njoints'])
            liste=extract_integers(file.split('.')[0])
            scene,camera,actor,repet,acti=liste
            # On va faire un traitement sur chaque body
            nb_frames=[] # Nombre de frame de chaque body 
            debut_frame=[] # liste des débuts de frame de chaque body. A priori a sera à virer plus tard
            mean=[] # liste des moyennes de chaque body
            l_std=[] # liste des standards deviation de chaque body
            l_num_chp=[] # liste du nombre de changepoint de chaque body
            liste_chp=[] # liste des changepoints de chaque body 
            array=np.array(nbodys)
            reference=20 # Correspond à Spine shoulder ! 
            for k in range(nbodys):
                array_k=data[f'skel_body{k}'] #* c'est le déplacement du squelette k 
                array_k=array_k-np.mean(array_k[:,reference,:],axis=0) # On recentre le squelette par rapport à la frame de référence
                array_k=array_k.reshape(array_k.shape[0],array_k.shape[1]*array_k.shape[2])
                
                nb_frames.append(array_k.shape[0]) # Nombre de frame pour chaque body!
                debut_frame.append(np.argmax(array>k)) #! A CHANGER 
                # Calcul de la moyenne par body 
                moyenne=np.mean(array_k,axis=0)
                sum_moy=np.sum(moyenne,axis=0)
                mean.append(sum_moy)
                # Calcul de la std par body 
                std=np.std(array_k,axis=0)
                sum_std=np.sum(std,axis=0)
                l_std.append(sum_std)
                t=model.fit_predict(array_k,pen=beta)# Calcul du nombre de change point
                liste_chp.append(t[0] if len(t)>1 else 0) # le dernier correspond à la longueur de la liste ,détail technique
                l_num_chp.append(len(t)-1)
            # Mise à jour dans les listes pour l'incorporer au dataframe
            result.append([nbodys,filename,actor,acti,camera,scene,repet,njoints])
            result_nb_frames.append(nb_frames)
            result_debut_frame.append(debut_frame)
            result_mean.append(mean)
            result_std.append(l_std)
            result_l_chp.append(liste_chp)
            result_n_chp.append(l_num_chp)
            maxnbodys=max(maxnbodys,nbodys)
    L=['nbodys','filename','actor','acti','camera','scene','repet','njoints']
    dtype={k:int for k in L}
    dtype['filename']=str
    df1=pd.DataFrame(result,columns=['nbodys','filename','actor','acti','camera','scene','repet','njoints'])
    df2=pd.DataFrame(result_nb_frames,columns=[f'nb_frames_body_{k}' for k in range(maxnbodys)])
    df3=pd.DataFrame(result_debut_frame,columns=[f'debut_frame_body_{k}' for k in range(maxnbodys)])
    df4=pd.DataFrame(result_mean,columns=[f'sum_mean_body_{k}' for k in range(maxnbodys)])
    df5=pd.DataFrame(result_std,columns=[f'sum_std_body_{k}' for k in range(maxnbodys)])
    df6=pd.DataFrame(result_n_chp,columns=[f'nb_chp_body_{k}' for k in range(maxnbodys)])
    df7=pd.DataFrame(result_l_chp,columns=[f'chp_body_{k}' for k in range(maxnbodys)])
    dfaconcat=[]

    dfaconcat=[df1,df2,df3,df4,df5,df6,df7]
    df=pd.concat(dfaconcat,axis=1)
    df.infer_objects()
    df.to_csv(os.path.join(path_csv,name_csv),index=False )

    return df ,os.path.join(path_csv,name_csv)

def preprocess_csv_RGB_to_skeletondf(seq_len:int=30,out_len:int=30,path_csv="./dataset/NTU_RGB+D/summary_NTU/summary_NTU.csv",path_data_npy:str='./dataset/NTU_RGB+D/numpyed/',preprocess=1):
    """récupère un .csv plus ou moins preprocess pour le transformer en un autre .csv qui va être plus facile à utiliser . 
    il est de la forme [ des données catégorielles, num_body,nb_frames, des données de preprocessing]
    il ne fait aucune modification sur la séquence d'entrée ou de sortie
    Parameters
    ----------
    seq_len : int, optional
        longueur de la séquence d'entrée, by default 30
    out_len : int, optional
        longueur de la séquence de sortie, by default 30
    path_csv : str, optional
        path vers le csv initial qui contient bcp d'informations, by default "./dataset/NTU_RGB+D/summary_NTU/summary_NTU.csv"
    path_data_npy : str, optional
        path qui contient tous les .npy des skeletons , by default './dataset/NTU_RGB+D/numpyed/'

    Returns
    -------
    df  
        le dataframe correspondant au .csv
    path_csv
        le path du .csv
    """
    #TODO: we may have some issue with NaN, and on the os.path.exists IL FAUT RAJOUTER LA DATA_PATH je pense
    if not os.path.exists(path_csv):
        _,path= summary_csv_NTU(path_csv=os.path.dirname(path_csv),path_data_npy=path_data_npy,name_csv=os.path.basename(path_csv))
    df=pd.read_csv(path_csv,low_memory=False)
    #
    categorical=['nbodys', 'filename', 'actor', 'acti', 'camera', 'scene', 'repet']
    list_user=[]
    maxnbodys=max(df['nbodys'])
    for k in range(maxnbodys):
        if preprocess==0:
            list_user.append([f'nb_frames_body_{k}',f'debut_frame_body_{k}'])
        elif preprocess==1:
            list_user.append([f'nb_frames_body_{k}',f'debut_frame_body_{k}',f'sum_mean_body_{k}',f'sum_std_body_{k}',f'nb_chp_body_{k}',f'chp_body_{k}'])
        else:
            print("preprocess doit être 0,1 (utils_dataset_preprocess_csv) ")
            list_user.append([f'nb_frames_body_{k}',f'debut_frame_body_{k}',f'sum_mean_body_{k}',f'sum_std_body_{k}',f'nb_chp_body_{k}',f'chp_body_{k}'])

    list_df=[]
    for k in range(maxnbodys):
        list_df.append(df[categorical+list_user[k]].copy().dropna(subset=[f'nb_frames_body_{k}']))
        list_df[k]['num_body']=k # Set the numéro of body correspondant

        # On va changer les noms qui dépendent du bodys pour qu'ils aient une syntaxe commune
        arename={}
        for columnname in list_df[k].columns:
            if "_body_" in columnname:
                arename[columnname]=columnname.replace(f'_body_{k}','')
            list_df[k].rename(columns=arename,inplace=True)
    
    vrai_df=pd.concat(list_df)
    vrai_df.infer_objects()
    vrai_df.to_csv(os.path.join(os.path.dirname(path_csv),f'liste_NTU_skeleton_{maxnbodys}.csv'),index=False)
    return vrai_df,os.path.join(os.path.dirname(path_csv),f'liste_NTU_skeleton_{maxnbodys}.csv')



def data_rentrer_dans_DATASET_NTU(path_csv:str='./dataset/NTU_RGB+D/summary_NTU/liste_NTU_skeleton_4.csv',seq_len:int=30,out_len:int=30,path_data_npy:str='./dataset/NTU_RGB+D/numpyed/',preprocess=1,path_excel:str='./dataset/NTU_RGB+D/summary_NTU/data_quality.xlsx'):
    """Fonction qu'on va faire rentrer dans le torch Dataset de NTU RGB+D
    CEST ICI QUON FAIT NOS OPERATIONS DE PREPROCESSING :!!!!
    Parameters
    ----------
    path_csv : str, optional
        le path de liste_NTU_skeleton_maxnbodys_, by default './dataset/NTU_RGB+D/summary_NTU/liste_NTU_skeleton_3.csv'
    seq_len : int, optional
        longueur de la séquence d'entrée, by default 30
    out_len : int, optional
        longueur de la sortie à prédire, by default 30
    #TODO: Rajouter des conditions ici si on veut garder des donées plus spécifiques? 
    Returns
    -------
    un dataframe de la même forme que le précédent mais uniquement avec ceux qui vérifient les bonnes conditions 
    """
    if not os.path.exists(path_csv):
        _,path= preprocess_csv_RGB_to_skeletondf(seq_len,out_len,os.path.join(os.path.dirname(path_csv),'summary_NTU.csv'),path_data_npy=path_data_npy)
         
    df=pd.read_csv(path_csv,low_memory=False)

    #On va faire un filtre sur les données
    if preprocess==0: #* Pas de modification du dataset non particulière 
        nvdf=df[(df['nb_frames']>=seq_len+out_len+df['debut_frame']) & (df['num_body']<=2)].dropna() #* CEST ICI QUON RECUPERE SEULEMENT CEUX QUI ONT UN NOMBRE DE FRAME SUFFISANT, A CHANGER SI BESOIn!!
        return nvdf

    if preprocess ==1:
        #print("Attention on filtre le dataset")
        excel_sheet=pd.read_excel(path_excel)
        row_to_keep=excel_sheet["A"].where((excel_sheet["good_data"]=="x")|(excel_sheet["average_data"]=="x")).dropna()#* On récupère uniquement les rows labéllisés comme bonne ou moyens dans l'excel

        nvdf=df[(df['nb_frames']>=seq_len+out_len+df['debut_frame']) & (df['num_body']<=2)].dropna() #* CEST ICI QUON RECUPERE SEULEMENT CEUX QUI ONT UN NOMBRE DE FRAME SUFFISANT, A CHANGER SI BESOIn!!
        nvdf=nvdf.where((nvdf['acti'].isin(row_to_keep)) &(nvdf["nb_chp"]>0)).dropna() #! nb_chp doitbien être à 0 
        nvdf["debut_frame"] =np.where(nvdf["chp"]+seq_len+out_len<nvdf["nb_frames"],nvdf["chp"], nvdf["debut_frame"])
        #nvdf["debut_frame"]=nvdf["debut_frame"] if nvdf["chp"].str[1]+seq_len+out_len>int(nvdf["nb_frames"]) else nvdf["chp"].str[1]
    return nvdf 
