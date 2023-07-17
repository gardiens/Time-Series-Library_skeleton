"""File to download the NTU dataset"""
"""
import os
import sys
import argparse
import zipfile
import shutil
import wget
import glob
from utils.NTU_RGB.txt2npy import *
root_path=os.getcwd()
import requests
import requests

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


if __name__ == "__main__":
    

    main_path=os.path.join(root_path, 'data/NTURGB+D')

    #* Créate a path /dataset/NTU if not exist
    if not os.path.exists(main_path):
        os.makedirs(main_path)


    # TAKE ID FROM SHAREABLE LINK
    file_id = "1CUZnBtYwifVXS21yVg62T-vrPVayso5H"
    # DESTINATION FILE ON YOUR DISK
    destination = os.path.join(main_path,'NTU_RGB+D60.zip')
    download_file_from_google_drive(file_id, destination)


    #* Unzip the dataset
    print("Unzip the dataset")
    directory_to_extract_to=os.path.join(main_path,'/raw/')
    if not os.path.exists(directory_to_extract_to):
        os.makedirs(directory_to_extract_to)
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


     # TAKE ID FROM SHAREABLE LINK
    file_id = "1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w"
    # DESTINATION FILE ON YOUR DISK
    destination = os.path.join(main_path,'NTU_RGB+120.zip')
    download_file_from_google_drive(file_id, destination)


    #* Unzip the dataset
    print("Unzip the dataset")
    directory_to_extract_to=os.path.join(main_path,'/raw/')
    if not os.path.exists(directory_to_extract_to):
        os.makedirs(directory_to_extract_to)
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    directory_data_traiter=os.path.join(main_path,"/numpyed/")
    if not os.path.exists(directory_data_traiter):
        os.makedirs(directory_data_traiter)
    #* Traitement des données
    missing_file_path=os.path.join(root_path,"/utils/NTU_RGB/NTU_RGBD120_samples_with_missing_skeletons.txt")
    save_npy_path=os.path.join(root_path,"/data/NTURGB+D/numpyed/")
    load_txt_path=directory_to_extract_to

    missing_files = _load_missing_file(missing_file_path)
    datalist = os.listdir(load_txt_path)
    alread_exist = os.listdir(save_npy_path)
    alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))
        
    for ind, each in enumerate(datalist):
        _print_toolbar(ind * 1.0 / len(datalist),
                        '({:>5}/{:<5})'.format(
                            ind + 1, len(datalist)
                        ))
        S = int(each[1:4])
        if S not in step_ranges:
            continue 
        if each+'.skeleton.npy' in alread_exist_dict:
            print('file already existed !')
            continue
        if each[:20] in missing_files:
            print('file missing')
            continue 
        loadname = load_txt_path+each
        print(each)
        mat = _read_skeleton(loadname,True,False,False) # Only load the skeleton
        mat = np.array(mat)
        save_path = save_npy_path+'{}.npy'.format(each)
        np.save(save_path, mat)
        # raise ValueError()
    _end_toolbar()
"""

from utils.NTU_RGB.txt2npy import _load_missing_file, _print_toolbar, _end_toolbar, _read_skeleton
import os
import numpy as np 
user_name = 'user'
save_npy_path = r'dataset\\NTU_RGB+D\\numpyed\\'
load_txt_path = r'dataset\\NTU_RGB+D\\raw\\'
missing_file_path =r'utils/NTU_RGB/NTU_RGBD120_samples_with_missing_skeletons.txt'
step_ranges = list(range(0,100)) # just parse range, for the purpose of paralle running. 

missing_files = _load_missing_file(missing_file_path)
datalist = os.listdir(load_txt_path)
alread_exist = os.listdir(save_npy_path)
alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))
    
for ind, each in enumerate(datalist):
    _print_toolbar(ind * 1.0 / len(datalist),
                    '({:>5}/{:<5})'.format(
                        ind + 1, len(datalist)
                    ))
    S = int(each[1:4])
    if S not in step_ranges:
        continue 
    if each+'.skeleton.npy' in alread_exist_dict:
        print('file already existed !')
        continue
    if each[:20] in missing_files:
        print('file missing')
        continue 
    loadname = load_txt_path+each
    print(each)
    mat = _read_skeleton(loadname,True,False,False) # Only load the skeleton
    mat = np.array(mat)
    save_path = save_npy_path+'{}.npy'.format(each)
    np.save(save_path, mat)
    # raise ValueError()
_end_toolbar()
    