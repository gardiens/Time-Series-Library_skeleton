""" Fichier pour build les csv pour refaire la documentation des données"""
from utils.NTU_RGB.utils_dataset import *

if __name__ == '__main__':
    print("on est parti")
    summary_csv_NTU()
    df,path= preprocess_csv_RGB_to_skeletondf(preprocess=1)
    print("fin de construction du csv, pour vérifier:",df.head())
         