from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader,dataset_NTURGBD,dataset_augmenter_augalone,dataset_augmenter_augmix
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader,ConcatDataset

from functools import partial
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
    'NTU': dataset_NTURGBD,
    'NTU-leg':partial(dataset_NTURGBD,quoi_pred='leg'),
    'NTU-body':partial(dataset_NTURGBD,quoi_pred='body'),
    'NTU-arm':partial(dataset_NTURGBD,quoi_pred='arm')
}


def data_provider(args, flag):
    if "augment" in args.data:
        Data=data_dict[args.data.replace("-augment","")]
    else:
        
        Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print("flag:{lag} et {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else: #* CAS LONG TIME FORECAST
        if args.data == 'm4':
            drop_last = False
        #* Cas NTU_RGBD
        if  'NTU' in args.data:
            
            data_set= Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                get_time_value=args.get_time_value ,#!
                get_cat_value=args.get_cat_value, #!
                preprocess=args.preprocess, #! Ne pas changer pls 
                split_train_test=args.split_train_test #!
            )
        


        else:
            #* Cas générique
            print("hein")
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
            )
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    
       
    
    if args.augment and flag=='train':
        l_dataset = [data_set]
        l_transfo_a=[backward(),rotate_data()]
        l_transfo_2=[Mixup(),CutMix()]
        str_prop=args.prop.split(";") #* Les données sont stockés de la forme 1.0;0.05;0.05;0.05 avec la même logique que pour l_transfo 
        l_prop=[float(i) for i in str_prop] #! Technique
        for k in range(len(l_transfo_a)):
            try:
                transfo=l_transfo_a[k]
                prop= l_prop[k] #* On récupère la proportion de données à augmenter
            except:
                print(transfo,prop)
                raise Exception("Problème de taille entre l_transfo et l_prop,Attention à la syntaxe de args.prop")
            data=dataset_augmenter_augalone(dataset_ini=data_set, transfo=transfo, prop=prop) #* On récupère le dataset augmenter
            if len(data)>0:
                l_dataset.append(data)
        
        #* A débugger!  
        for k in range(len(l_transfo_2)):
            try:
                transfo=l_transfo_2[k]
                prop= l_prop[k+len(l_transfo_a)] #* On récupère la proportion de données à augmenter
            except:
                print(transfo,prop)
                raise Exception("Problème de taille entre l_transfo et l_prop,Attention à la syntaxe de args.prop")
            data=dataset_augmenter_augmix(dataset_ini=data_set, transfo=transfo, prop=prop)

            if len(data)>0:
                l_dataset.append(data)

        data_set=ConcatDataset(l_dataset)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        return data_set, data_loader
from utils.NTU_RGB.augmentation import *