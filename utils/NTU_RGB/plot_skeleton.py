import matplotlib.pyplot as plt
import numpy as np
import os
from utils.NTU_RGB.utils_NTU import read_xyz
from utils.NTU_RGB.joints import connexion_tuples
import matplotlib.animation as animation
import mpl_toolkits.mplot3d as plt3d
""" Tuto clair ici https://acme.byu.edu/00000179-d3f1-d7a6-a5fb-ffff6a210001/animation-pdf"""
"""path="./raw/S018C001P045R001A073.skeleton"
skeleton=read_xyz(path)
"""

from utils.constantes import model_dict,data_dict
def check_nombre_frames(L):
    """ Check si toutes les skeletons ont le même nombre de frames, sinon renvoie une erreur.
    supposé de la forme (nb_joints,3,nb_frames)"""
    nb_frames=L[0].shape[2]
    for i in range(len(L)):
        if L[i].shape[2]!=nb_frames:
            raise ValueError("Les deux np.array n'ont pas le même nombre de frames")
    

def plot_video_skeletons(mat_skeletons,title=None,write=True,animate=False,path_folder_save='./videos/',save_name='skeleton',num_body=0):
    """Plot the videos for several skeletons
    Input: les mat_skeletons est une liste d'arrays qui sont supposés être des sorties de FEDFormers de la forme (nb_joints,3,nb_frames)
    la couleur est preset et ne peut pas dépasser 11 individus...
     Parameters
    ----------
    mat_skeletons : list of np.array
        liste de np.array de la forme (nb_joints,3,nb_frames)
    save_name : str, 
        le nom du fichier qu'on va sauvegarder ( sans le .MP4), by default 'skeleton'
    title : str, optional
        nom du titre visible dans le .mp4 à côté de l'action, by default None
    write : bool, optional
        si True, renvoie un fichier .mp4 dans le bon dossier, by default True
    animate : bool, optional
        si True, renvoie la version non animé mais buggé de base, by default False
    path_folder_save : str, optional
        folder where we store the videos, by default './videos/'

    Returns
    -------
    ani the animation but useless in most of the case
    """
    liste_color_skeleton=["b","r","c","m","y","k","w"]
    #* définit les constantes globales de cette fonction
    nb_framestotal= mat_skeletons[0].shape[2]
    check_nombre_frames(mat_skeletons) # Check si toutes les skeletons ont le même nombre de frames, sinon renvoie une erreur.
    nb_skeletons=len(mat_skeletons)
    x_coord = int(0)
    y_coord = int(2)
    z_coord = int(1) # true coord?
    #COLOR skeletons??
    axis_length = 0.2
    # Limit of plot 
    xlimp= max(max([ np.amax(mat_skeletons[i][:,x_coord,:]) for i in range(nb_skeletons)]),axis_length)
    ylimp=max(max([ np.amax(mat_skeletons[i][:,y_coord,:]) for i in range(nb_skeletons)]),axis_length)
    zlimp=max(max([ np.amax(mat_skeletons[i][:,z_coord,:]) for i in range(nb_skeletons)]),axis_length)
    xlimm=min(min([ np.amin(mat_skeletons[i][:,x_coord,:]) for i in range(nb_skeletons)]),-axis_length)
    ylimm=min(min([ np.amin(mat_skeletons[i][:,y_coord,:]) for i in range(nb_skeletons)]),-axis_length)
    zlimm=min(min([ np.amin(mat_skeletons[i][:,z_coord,:]) for i in range(nb_skeletons)]),-axis_length)
    #* Défine la figure overall, c'est que du statique 
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #for nice view
    ax.view_init(10,50)

    #* Plot des points de références
    ax.scatter([0], [0], [0], color="red")
    ax.scatter([axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], marker="v", color="red")

    x_axis = plt3d.art3d.Line3D([0, axis_length], [0, 0], [0, 0])
    x_axis.set_color("red")
    y_axis = plt3d.art3d.Line3D([0, 0], [0, axis_length], [0, 0])
    y_axis.set_color("red")
    z_axis = plt3d.art3d.Line3D([0, 0], [0, 0], [0, axis_length])
    z_axis.set_color("red")
    ax.add_line(x_axis)
    ax.add_line(y_axis)
    ax.add_line(z_axis)
    #* Ici on a les trois points, les axes 
    ax.set_xlim3d(xlimm, xlimp)
    ax.set_ylim3d(ylimm, ylimp)
    ax.set_zlim3d(zlimm, zlimp)
    if title!=None: #* set a title.
        ax.text2D(0.05, 0.95,title, transform=ax.transAxes)
    else:
        ax.text2D(0.05, 0.95,save_name, transform=ax.transAxes)
  
    #* On définit les lignes et points à animer
    liste_line=[ [ax.plot([],[],[],linestyle=':',color=liste_color_skeleton[k])[0] for _ in connexion_tuples] for k in range(nb_skeletons) ]
    liste_point=[[ax.plot([],[],[],color=liste_color_skeleton[k],marker='o',markersize=3)[0] for _ in range(25)] for k in range(nb_skeletons)]
    liste_invariant=[ [ax.plot([0],[0],[0],color='green')[0] for _ in range(4)] for _ in range(nb_skeletons) ]
    #* On définit les fonctions d'animation
    def update(iteration,mat_skeletons,ligneaplots,pointss,invariants):
        """ On mets a jour le plot, c'est sombre mais fonctionnel théoriquement """
        for k in range(nb_skeletons):
            ligneaplot=ligneaplots[k]
            point=pointss[k]
            invariant=invariants[k]
            data=mat_skeletons[k]
            for i,line in enumerate(ligneaplot):

                x1,z1,y1=data[connexion_tuples[i][0]][:,iteration]
                x2,z2,y2=data[connexion_tuples[i][1]][:,iteration]
                #print(y1,z1)
                x1,x2=min(x1,x2),max(x1,x2)
                line.set_data(np.linspace(x1,x2,num=20),np.linspace(y1,y2,num=20))
                line.set_3d_properties(np.linspace(z1,z2,num=20))
            for i,point in enumerate(point): #plot les points
                x,z,y=data[i][:,iteration]
                point.set_data(x,y)
                point.set_3d_properties(z)
            
            x_spine_mid = data[1][x_coord][iteration] # 1 correspond au spine mid exactement
            y_spine_mid = data[1][y_coord][iteration]
            z_spine_mid = data[1][z_coord][iteration]
            invariant[0].set_data(np.linspace(x_spine_mid,x_spine_mid+axis_length,num=20),np.linspace(y_spine_mid,y_spine_mid,num=20))
            invariant[0].set_3d_properties(np.linspace(z_spine_mid,z_spine_mid,num=20))
            invariant[1].set_data(np.linspace(x_spine_mid,x_spine_mid,num=20),np.linspace(y_spine_mid,y_spine_mid+axis_length,num=20))
            invariant[1].set_3d_properties(np.linspace(z_spine_mid,z_spine_mid,num=20))
            invariant[2].set_data(np.linspace(x_spine_mid,x_spine_mid,num=20),np.linspace(y_spine_mid,y_spine_mid,num=20))
            invariant[2].set_3d_properties(np.linspace(z_spine_mid,z_spine_mid+axis_length,num=20))
            


            invariant[3].set_data(np.linspace(0,x_spine_mid,num=20),np.linspace(0,y_spine_mid,num=20))
            
            invariant[3].set_3d_properties(np.linspace(0,z_spine_mid,num=20))
            
            #* des points qui ne bougent pas 
        
        return ligneaplots,pointss,invariants
    
    #* Fin on met tous en forme pour l'animation

    if not os.path.exists(path_folder_save):
        os.makedirs(path_folder_save)
    ani=animation.FuncAnimation(fig,update,frames=range(nb_framestotal), fargs=(mat_skeletons,liste_line,liste_point,liste_invariant))
    if write:
        animation.writer=animation.writers['ffmpeg']
        plt.ioff() # Turn off interactive mode just in case
        
        ani.save( path_folder_save+save_name+'.mp4',fps=15) # A FAIRE PASSER EN MP4
         
    if animate:
        #u may expect strange behavior, only used write
        # it may be really laggy if u dont have a good computer
        plt.ion()
        plt.show()
    return ani


def plot_skeleton(path_skeleton:str=None,save_name='skeleton',title=None,write=True,animate=False,path_folder_save='./videos/',num_body=0):
    """plot le skeleton d'une personne du dataset NTU RGB+D sachant le .skeleton 

    Parameters
    ----------
    path_skeleton : str
        le path complet du skeleton
    save_name : str, 
        le nom du fichier qu'on va sauvegarder ( sans le .MP4), by default 'skeleton'
    title : str, optional
        nom du titre visible dans le .mp4 à côté de l'action, by default None
    write : bool, optional
        si True, renvoie un fichier .mp4 dans le bon dossier, by default True
    animate : bool, optional
        si True, renvoie la version non animé mais buggé de base, by default False
    path_folder_save : str, optional
        folder where we store the videos, by default './videos/'

    Returns
    -------
    ani the animation but useless in most of the case
    """    
    if path_skeleton==None:
        print("ATTENTION LE PATH EST NON INDIQUER, ON VA EN METTRE UN PAR DEFAUT")
        path_skeleton="dataset/NTU_RGB+D/raw/S001C001P001R001A001.skeleton"
    # if path_folder is not in the directory, create it
    if not os.path.exists(path_folder_save):
        os.makedirs(path_folder_save)
    #* Parsing the skeleton
    if not path_skeleton.strip().endswith('.skeleton') and not path_skeleton.strip().endswith('.npy'):
        path_skeleton='./dataset/NTU_RGB+D/raw/'+path_skeleton+'.skeleton'

    if path_skeleton.strip().endswith('.skeleton'):
        skeleton = read_xyz(path_skeleton) # l'ouput est `(3 {x, y, z}, max_frame, num_joint, 2 {n_subjects})
        print(skeleton.transpose(3, 2, 0, 1).shape)
        nv_skeleton=skeleton.transpose(3, 2, 0, 1)[num_body] #! Il st de la forme (nb_joints,3,nb_frames)

    elif path_skeleton.strip().endswith('.npy'):
        skeleton=np.load(path_skeleton,allow_pickle=True).item()
        print(' le plot de skeleton en .npy nest pas encore supporté,unexpected behavior EXPECTED !')
        skeleton=skeleton['b0']
        nv_skeleton=skeleton.transpose(3, 2, 0, 1)[0] #! Il st de la forme (nb_joints,3,nb_frames)

    else:
        assert ValueError('Unexpected file format, expected .skeleton or .npy')
    print("On sauvegarde le skeleton au nom de ",save_name)
    return plot_video_skeletons(mat_skeletons=[nv_skeleton],save_name=save_name,title=title,write=write,animate=animate,path_folder_save=path_folder_save)




from data_provider.data_loader import dataset_NTURGBD

def plot_prediction_modele(model,data_set:dataset_NTURGBD,args=None,checkpoint=None,sample_name:str="S001C001P001R001A001",settings=None,save_name='skeleton',title=None,write=True,animate=False,path_folder_save='./videos/'):
    """ Plot la vidéo de prédiction d'un modèle sur un sample donné"""
    print("on va plot:",sample_name)
    print(data_set.liste_path.where(data_set.liste_path["filename"]==sample_name).dropna())

    entry=data_set.get_data_from_sample_name(sample_name)
    entry_model=data_set.get_input_model(entry)
    X=entry[0]
    y_true=entry[1]
    
    if  checkpoint !=None:
        print("on load un checkpoint, ne devrait pas arriver dans la prédiction de modèle")
        if args==None:
            print("pas de args, on en définit un par défaut")
            #* Exemple de args possible pour débug et setting
            class Args:
                def __init__(self):
                    # basic config
                    self.task_name = 'long_term_forecast'
                    self.is_training = 1
                    self.model_id = 'test'
                    self.model = 'Autoformer'

                    # data loader
                    self.data = 'NTU'
                    self.root_path = './dataset/NTU_RGB+D/'
                    self.data_path = 'numpyed/'
                    self.features = 'M'
                    self.target = 'OT'
                    self.freq = 'h'
                    self.checkpoints = './checkpoints/'

                    # forecasting task
                    self.seq_len = 30
                    self.label_len = 42
                    self.pred_len = 42
                    self.seasonal_patterns = 'Monthly'

                    # inputation task
                    self.mask_rate = 0.25

                    # anomaly detection task
                    self.anomaly_ratio = 0.25

                    # model define
                    self.top_k = 5
                    self.num_kernels = 6
                    self.enc_in = 75
                    self.dec_in = 75
                    self.c_out = 75
                    self.d_model = 512
                    self.n_heads = 8
                    self.e_layers = 2
                    self.d_layers = 1
                    self.d_ff = 2048
                    self.moving_avg = 25
                    self.factor = 1
                    self.distil = True
                    self.dropout = 0.1
                    self.embed = 'timeNTU'
                    self.activation = 'gelu'
                    self.output_attention = False

                    # optimization
                    self.num_workers = 10
                    self.itr = 1
                    self.train_epochs = 10
                    self.batch_size = 4
                    self.patience = 3
                    self.learning_rate = 0.0001
                    self.des = 'test'
                    self.loss = 'MSE'
                    self.lradj = 'type1'
                    self.use_amp = False

                    # GPU
                    self.use_gpu = False
                    self.gpu = 0
                    self.use_multi_gpu = False
                    self.devices = '0'

                    # de-stationary projector params
                    self.p_hidden_dims = [128, 128]
                    self.p_hidden_layers = 2

                    # NTU_RGB
                    self.get_time_value = True
                    self.get_cat_value = False
            args=Args()

            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, 0)
                
                #* Load the model from the checkpoints
        print("dans plot prédiction on load depuis un checkpoint")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.float() # Just in case? 
    #* On va prédire la sortie du modèle
    #! Attention X est sensé être sous la forme d'un batch de taille 1, donc on va prendre le premier échantillon
    y_out=model(entry_model[0],entry_model[1],entry_model[2],entry_model[3])
    
        
    # y_out est de la forme (1,seq_len, nb_channel) si sortie de FEDFORMERS
    y_out=y_out[0].detach().numpy() # on prend le premier échantillon
    #* On va plot les résultats

    #* On retransforme dans un format utilisable 
    #print(mean)
    y_out=data_set.inverse_transform_data(y_out)
    y_true=data_set.inverse_transform_data(y_true)
    X=data_set.inverse_transform_data(X)
    #print("y_out",y_out)
    X_pred=y_out #np.concatenate((X,y_out),axis=0) # Quelle axis?
    X_true=y_true #np.concatenate((X,y_true),axis=0)

    #* plot_video_skeletons demande :  (nb_joints,3,nb_frames) et pour l'instant on a ( nb_frames,nb_joints,3) 
    X_pred=X_pred.transpose(1,2,0)
    X_true=X_true.transpose(1,2,0)
    X=X.transpose(1,2,0)
    #* On va plot les résultats
    plot_video_skeletons(mat_skeletons=[X_true,X_pred],save_name=save_name,title=title,write=write,animate=animate,path_folder_save=path_folder_save)
    plot_video_skeletons(mat_skeletons=[X],save_name=save_name+'_input',title=title,write=write,animate=animate,path_folder_save=path_folder_save)
import os
from utils.NTU_RGB.utils_dataset import extract_integers
def plot_plusieurs_skeletons_selon_A(model,data_set,checkpoints=None,nb_sample=1,video_save_path="./videos",root_path='./dataset/NTU_RGB+D',data_path_npy="numpyed/"):
    #* Récupère tous les skeletons selon certaines activités  et on sauvegarde leurs noms    
    nb_acti=120
    L=np.zeros(nb_acti)
    result=[[] for _ in range(nb_acti)]
    file_path=os.path.join(root_path,data_path_npy)
    
    for file in os.listdir(file_path):
      
        file_id=file.split('.')[0]
        array=extract_integers(file_id)

        k=array[-1]

        if  L[k-1]<nb_sample:
            if L[k-1]==0 or np.count_nonzero(np.array(extract_integers(result[k-1][-1]))-array)>2:
                #print('oui oui')
                if L[k-1] !=0:
                    print(np.count_nonzero(np.array(extract_integers(result[k-1][-1]))-array))
                result[k-1].append(file)
                L[k-1]+=1 
        
        if np.sum(L)==nb_acti*nb_sample:

            break
    print("On a fini de  récupérer les samples qu'on voulait calculer")
    file_path_npy=os.path.join(root_path,"raw/") 
     #Create folder Anumber of action and put each video in it
    for i in range(nb_acti):
        if not os.path.exists(os.path.join(video_save_path,'A'+str(i+1))):
            os.makedirs(os.path.join(video_save_path,'A'+str(i+1)))
    for l in result:
        for file in l:
            #print(file.split('.')[0])
            print('on soccupe de',file)
            id_file=file.split('.')[0]
            array=np.array(extract_integers(file_id))
            k=array[-1]
            path_skeleton=os.path.join(file_path_npy,id_file+'.skeleton')
            save_name='skeleton_'+id_file
            directory_k='A'+str(k)
            if not os.path.exists(os.path.join(video_save_path,directory_k,save_name+'.mp4')):
                
                plot_prediction_modele(model=model,data_set=data_set,checkpoint=checkpoints,sample_name=id_file,save_name=save_name)
    #* Peut être utile pour récupérer les noms des vidéos