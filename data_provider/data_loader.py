import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
#from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
from utils.NTU_RGB.utils_dataset import time_serie_NTU,time_serie_NTU_particular_body, data_rentrer_dans_DATASET_NTU
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


#* DATA_LOADER A IMPLEMENTER JIMAGINE

class dataset_NTURGBD(Dataset):
    
    """Made by Phillipe rambaud and Pierrick Bournez 
    a bit debugged by Pierrick """
    """Dataset for the NTU RGB+D dataset, les items sont de la forme (time_serie, label) pas comme ETT """
    
    def __init__(self, root_path:str="./dataset/NTU_RGB+D/",data_path:str='numpyed/', flag='train', size:list=None,
                 features:str='MS',transform=None,item=time_serie_NTU,csv_path='./dataset/NTU_RGB+D/summary_NTU/liste_NTU_skeleton_4.csv',
                 get_time_value=False,get_cat_value=False,train_size=0.80,test_size=0.10,preprocess:int=1,split_train_test:str="action",quoi_pred="all") -> None:
        """Dataset de NTU_RGB+D120 et NTU_RGB+D60 implémenté par pierrick

        Parameters
        ----------
        root_path : str, optional
            path où les données vont être stockés, by default "./dataset/NTU_RGB+D/"
        data_path : str, optional
            path relatif à root_path où les .npy des squelettes sont stockés, by default 'numpyed/'
        flag : str, optional
            peut valoir [train,test,vali] et permets de récupérer le dataset correspondant, by default 'train'
        size : list, optional
            contient les tailles des longueurs à prédire.Il est de la forme [args.seq_len,args.pred_len,args.label_len], by default None
        features : str, optional
            non utilisé, by default 'MS'
        transform : _type_, optional
            non utilisé, by default None
        item : _type_, optional
            Une fois le dataframe des données construites, permets de récupérer la matrice avec les transformations adéquates, by default time_serie_NTU
        csv_path : str, optional
            path du csv qui contient pour chaque ligne le nom du fichier, le nom du squelette correspondant et le début du temps à prédire. Voir ??, by default './dataset/NTU_RGB+D/summary_NTU/liste_NTU_skeleton_4.csv'
        get_time_value : bool, optional
            indique si on veut obtenir en sortie un encodage temporel. Il n'a pas d'incidence sur le modèle et permets juste de reprendre l'architecture des autres datasets, by default False
        get_cat_value : bool, optional
            vaut True si on veut obtenir en sortie en plus un encodage des données catégoriques. voir ??? et n'a pas été implémenté sérieusement, by default False
        train_size : float, optional
            taille du dataset de train. N'est utilise que si args.split_train_test vaut random, by default 0.80
        test_size : float, optional
            taille du dataset de test. N'est utiliser que si args.split_train_test vaut random, by default 0.10
        preprocess : int, optional
            vaut 1 si on effectue des opérations sur le dataset. Il est recommandé de ne pas y toucher, by default 1
        split_train_test : str, optional
            indique si le partage entre le dataset de test et train est selon les actions où au hasard. Pour les actions, le dataset de train sont les actions <=100, pour la validation entre 100 et 110 et test sont ceux de plus de 110. Choix entre [action,random], by default "action"
        quoi_pred : str, optional
            indique si on veut prédire tous les membres ou seulement une proportion. Les choix possibles sont [all,body,arm,leg] utile que lorsque le modèle est Metaformer, by default "all"
        """
        if size==None: # Initialisation des tailles par défaut pour débugger
            
            #TODO: A MODIFIEr
            self.seq_len = 16
            self.label_len=32
            self.pred_len=32

        else:
            self.seq_len = size[0]

            self.label_len=size[1] #* longueur des labels
            self.pred_len=size[2] #* longueur de prédiction des times_series
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        if flag=='test':
            self.out_len=size[1] #!
        else:
            self.out_len=self.label_len #* Longueur des labels, paramètre le plus important de size
        #* les autres paras sont globalement inutiles
        self.data_path=os.path.join(root_path,data_path) #* path où sont stockés les .npy
        self.preprocess=preprocess 
        self.csv_path=csv_path
        self.split_train_test=split_train_test
        self.input_len=self.seq_len #* longueur de la time_series en entrée
        self.nb_joints=25 #* nombre de joints dans la time_series
        self.get_cat_value=get_cat_value #* si on veut les catégories
        self.get_time_value=get_time_value #* si on veut les time_values
        #* Ici, on va initialisater la classe qui sert à récupérer les données.
        if quoi_pred=="all":
            self.item=time_serie_NTU(data_path=self.data_path,input_len=self.input_len,output_len=self.out_len,get_cat_value=self.get_cat_value,get_time_value=self.get_time_value,preprocess=self.preprocess) #* Classe de base du data'set qui va renvoyer la time series voulu 
        if quoi_pred=="leg":
            self.item=time_serie_NTU_particular_body(input_len=self.input_len,output_len=self.out_len,get_cat_value=self.get_cat_value,get_time_value=self.get_time_value,preprocess=self.preprocess,quoi_pred="leg")

        if quoi_pred=="arm":
            self.item=time_serie_NTU_particular_body(input_len=self.input_len,output_len=self.out_len,get_cat_value=self.get_cat_value,get_time_value=self.get_time_value,preprocess=self.preprocess,quoi_pred="arm")

        if quoi_pred=="body":
            self.item=time_serie_NTU_particular_body(input_len=self.input_len,output_len=self.out_len,get_cat_value=self.get_cat_value,get_time_value=self.get_time_value,preprocess=self.preprocess,quoi_pred="body")
        # ne sert que lorsque split_train_test vaut random
        self.test_size=test_size
        self.train_size=train_size
        self.vali_size=1-self.train_size-self.test_size
        # attribut qui correspond au dataframe. Il est initialisé dans get_df
        self.liste_path=self.get_df() #* Ici liste_path correspond au dataset où on va prendre les données voulues! On va d'abord
        
    def __len__(self) -> int:
        """longueur du dataset

        Returns
        -------
        int
            longueur du dataset
        """
        return len(self.liste_path)
    

    def __getitem__(self, index: int):
        """fonction qui va permettre de récupérer les données importantes du dataset

        Parameters
        ----------
        index : int
            entier correspondant à la place dans le dataset

        Returns
        -------
        _type_
            _description_
        """
        time_series=self.item
        row=self.liste_path.iloc[index] #C'est la row du dataframe 

        if self.set_type==2:
            return time_series.get_data(row=row)
        else:
            return time_series.get_data(row=row)
        
    def get_df(self):
        """permet de récupérer le dataframe correspondant au dataset voulu

        Returns
        -------
        df
            dataframe issu de data_rentrer_dans_DATASET_NTU mais slicer en foncttion si on est en train/vali/test
        """     
        df=data_rentrer_dans_DATASET_NTU(path_csv=self.csv_path,seq_len=self.seq_len,out_len=self.out_len,path_data_npy=self.data_path,preprocess=self.preprocess) # c'est un pandas dataframe qui devriat tous avoir normalement

        #* On split entre les trois datasets
        if self.split_train_test=="action":#* le test_set correspond aux acti de 110 à 120 et le validation set correspond au set de 100 à 110
            
            if self.set_type==0: #* Training one
                return df.where(df['acti']<100).dropna()
            elif self.set_type==2: #* Validation one
                return df.where((df['acti']>=100) & (df['acti']<110)).dropna()
            else: #* Test one
                return df.where(df['acti']>=110).dropna()
            
  
        elif self.split_train_test=="random":
            if self.set_type==0:
                return pd.DataFrame(train_test_split(df,test_size=1-self.train_size)[self.set_type])
            else:
                df2=pd.DataFrame(train_test_split(df,train_size=self.train_size)[1])
                #print(self.test_size/self.train_size)
                return pd.DataFrame(train_test_split(df2,test_size=self.test_size/(1-self.train_size))[self.set_type-1])
   
    def get_data_from_sample_name(self,name_skeleton:str,num_body=0):
        """Fonction pratique de débuggage uqi permet de récupérer les données d'un squelette uniquement à partir de son nom

        Parameters
        ----------
        name_skeleton : str
            nom du skeleton ( sans le .npy ou .skeleton)
        num_body : int, optional
            numéro du body correspondant, by default 0

        Returns
        -------
        entry
            sortie  de la fonction get_data de la classe time_serie_NTU. généralement de la forme entrée,label,time_value_enc,time_value_dec

        Raises
        ------
        ValueError
            vérifie si le squelette est bien dans le dataset. Cela peut arriver si son nombre de frame est plus petit que la séquence à prédire ou si le numéro de body est incorrect
        """        
      
        """ name_skeleton doit être de la forme 'S001C001P001R001A001'
        permet de récupérer la data sachant uniquement le nom du skeleton."""
        if name_skeleton[-4:]=='.npy' or name_skeleton[-8:]=='.skeleton':
            name_skeleton=name_skeleton[:-4] # pour virer le .skeleton ou .npy
        try:

            row=self.liste_path.where( (self.liste_path['filename']==name_skeleton) & (self.liste_path["num_body"]==num_body)).dropna().iloc[0]
        except:
            print("Bugg sur le nom du skeleton et body",name_skeleton,num_body)
            print(self.liste_path.where( (self.liste_path['filename']==name_skeleton) & (self.liste_path["num_body"]==num_body)).dropna())
            raise ValueError("bug num_body ou name_skeleton dans get_data_from_sample_name")
        time_series=self.item
        return time_series.get_data(row=row)
    

    def inverse_transform_data(self,x,preprocessing=True):
        """ renvoie X de la bonne forme ( nb_frames,nb_joints,3) et effectue les potentielles effets inverses"""
        if preprocessing:
            return self.item.inverse_transform(x)
        else:
            return self.item.inverse_transform(x)

    def get_input_model(self,entry):
        """ depuis un X obtenu de get_data ou get_data_from_sample_name, renvoie un input de la bonne forme pour le modèle"""
        return self.item.get_input_model(entry)

def apply_transfo(entry,k,transfo):

    return transfo(entry[k].reshape(entry[k].shape[0],entry[k].shape[1]//3,3)).reshape(entry[k].shape[0],entry[k].shape[1])
def apply_transfo2(entry,k,transfo,entry2):
    return transfo(entry[k].reshape(entry[k].shape[0],entry[k].shape[1]//3,3),entry2.reshape(entry2.shape[0],entry2.shape[1]//3,3)).reshape(entry[k].shape[0],entry[k].shape[1])

class dataset_augmenter_augalone(Dataset):
    """brique de base pour augmenter un dataset initial. Il suffit de lui donner un dataset et une transformation et il va renvoyer un dataset augmenté. La transformation ne doit prendre en entrée qu'une seule entrée du dataset

    Parameters
    ----------
    Dataset : Dataset
        dataset d'entrée dans le modèle
    """
    def __init__(self,dataset_ini:dataset_NTURGBD,transfo,prop=0.05):
        """initialisationde dataset_augmenter

        Parameters
        ----------
        dataset_ini : dataset_NTURGBD
            dataset où on va faire la data augmentation.les entrées doivent être de la forme (nb_frames, nb_channels)
        transfo : fonction
            fonction sur une entrée begin ou label qui va renvoyer une nouvelle entrée . la fonction doit prendre en entrée une matrice de la forme (nb_frames,nb_joints,3) et renvoyer une matrice de la forme (nb_frames,nb_joints*3
        prop : float, optional
            nombre de nouvelles données crées parce processus, by default 0.05
        """        
        self.data_ini=dataset_ini #SUPPOSER NTU_RGB 
        self.transfo=transfo 
        self.l_ind=torch.randint(low=0,high=len(self.data_ini),size=(int(len(self.data_ini)*prop),)) #* les indices des données où on va appliquer la transformation
    
    def __getitem__(self, index) :
        entry=self.data_ini.__getitem__(int(self.l_ind[index])) #* Renvoie un élément de la forme :return begin,label,time_value_enc,time_value_dec
        #* apply_transfo reshape 
        return [apply_transfo(entry,0,self.transfo),apply_transfo(entry,1,self.transfo)]+list(entry[2:])
    def __len__(self):
        return len(self.l_ind)

class dataset_augmenter_augmix(Dataset):
    def __init__(self,dataset_ini:dataset_NTURGBD,transfo,prop=0.05):
        """initialisationde dataset_augmenter

        Parameters
        ----------
        dataset_ini : dataset_NTURGBD
            dataset où on va faire la data augmentation.les entrées doivent être de la forme (nb_frames, nb_channels)
        transfo : fonction
            fonction sur deux entrée begin ou label qui va renvoyer une nouvelle entrée . la fonction doit prendre en entrée une matrice de la forme (nb_frames,nb_joints,3) et renvoyer une matrice de la forme (nb_frames,nb_joints*3
        prop : float, optional
            nombre de nouvelles données crées parce processus, by default 0.05
        """    
        self.data_ini=dataset_ini #SUPPOSER NTU_RGB 
        self.transfo=transfo 
        self.l_ind=torch.randint(low=0,high=len(self.data_ini),size=(int(len(self.data_ini)*prop),))
        self.l_ind2=torch.randint(low=0,high=len(self.data_ini),size=(int(len(self.data_ini)*prop),))
    def __getitem__(self, index) :
        entry=self.data_ini.__getitem__(int(self.l_ind[index])) #* Renvoie un élément de la forme :return begin,label,time_value_enc,time_value_dec
        entry2,label2=self.data_ini.__getitem__(int(self.l_ind2[index]))[:2] #* Renvoie un élément de la forme :return begin,label,time_value_enc,time_value_dec
        return [apply_transfo2(entry,0,self.transfo,entry2),apply_transfo2(entry,1,self.transfo,label2)]+list(entry[2:])
    def __len__(self):
        return len(self.l_ind)

#* Ancienne classe de dataset déjà présente
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)