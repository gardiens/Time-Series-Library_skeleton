import matplotlib.pyplot as plt
import numpy as np
import os
from utils.NTU_RGB.utils_NTU import read_xyz
from utils.NTU_RGB.joints import connexion_tuples
import matplotlib.animation as animation
import mpl_toolkits.mplot3d as plt3d
""" Tuto clair ici pour le fonctionnement intrinsèque du plot https://acme.byu.edu/00000179-d3f1-d7a6-a5fb-ffff6a210001/animation-pdf"""


def check_nombre_frames(L):
    """ Check si toutes les skeletons ont le même nombre de frames, sinon renvoie une erreur.
    supposé de la forme (nb_joints,3,nb_frames)"""
    nb_frames=L[0].shape[2]
    for i in range(len(L)):
        if L[i].shape[2]!=nb_frames:
            raise ValueError("Les deux np.array n'ont pas le même nombre de frames")
    

def plot_video_skeletons(list_mat_skeletons,title=None,write=True,animate=False,path_folder_save='./videos/',save_name='skeleton',num_body=0):
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
    liste_color_skeleton=["b","r","c","m","y","k","w"] # les couleurs utilisés pour les différents skeleton.
    #* définit les constantes globales de cette fonction
    liste_nb_framestotal=[mat_skeleton.shape[2] for mat_skeleton in list_mat_skeletons]
    nb_framestotal= max(liste_nb_framestotal)
    #check_nombre_frames(list_mat_skeletons) # Check si toutes les skeletons ont le même nombre de frames, sinon renvoie une erreur. Updated the 08/08
    nb_skeletons=len(list_mat_skeletons)
    x_coord = int(0)
    y_coord = int(2)
    z_coord = int(1) # true coord?
    axis_length = 0.2
    # Limit of plot 
    xlimp= max(max([ np.amax(list_mat_skeletons[i][:,x_coord,:]) for i in range(nb_skeletons)]),axis_length)
    ylimp=max(max([ np.amax(list_mat_skeletons[i][:,y_coord,:]) for i in range(nb_skeletons)]),axis_length)
    zlimp=max(max([ np.amax(list_mat_skeletons[i][:,z_coord,:]) for i in range(nb_skeletons)]),axis_length)
    xlimm=min(min([ np.amin(list_mat_skeletons[i][:,x_coord,:]) for i in range(nb_skeletons)]),-axis_length)
    ylimm=min(min([ np.amin(list_mat_skeletons[i][:,y_coord,:]) for i in range(nb_skeletons)]),-axis_length)
    zlimm=min(min([ np.amin(list_mat_skeletons[i][:,z_coord,:]) for i in range(nb_skeletons)]),-axis_length)
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
    # plot des lignes de références 
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
        if nb_skeletons==1:
            ax.text2D(0.1,0.90,"the skeleton",color=liste_color_skeleton[0], transform=ax.transAxes)

        if nb_skeletons==2:
            ax.text2D(0.1,0.90,"ground skeleton",color=liste_color_skeleton[0], transform=ax.transAxes)
            ax.text2D(0.1,0.85,"predicted skeleton",color=liste_color_skeleton[1], transform=ax.transAxes)
    else:
        ax.text2D(0.05, 0.95,save_name, transform=ax.transAxes)
        if nb_skeletons==1:
            ax.text2D(0.05,0.90,"skeleton",color=liste_color_skeleton[0], transform=ax.transAxes)

        if nb_skeletons==2:
            ax.text2D(0.05,0.90,"ground skeleton",color=liste_color_skeleton[0], transform=ax.transAxes)
            ax.text2D(0.05,0.85,"predicted skeleton",color=liste_color_skeleton[1], transform=ax.transAxes)

  
    #* On définit les lignes et points à animer
    liste_line=[ [ax.plot([],[],[],linestyle=':',color=liste_color_skeleton[k])[0] for _ in connexion_tuples] for k in range(nb_skeletons) ]
    liste_point=[[ax.plot([],[],[],color=liste_color_skeleton[k],marker='o',markersize=3)[0] for _ in range(25)] for k in range(nb_skeletons)]
    liste_invariant=[ [ax.plot([0],[0],[0],color='green')[0] for _ in range(4)] for _ in range(nb_skeletons) ]
    #* On définit les fonctions d'animation
    def update(iteration,list_mat_skeletons,ligneaplots,pointss,invariants):
        """ On mets a jour le plot, c'est sombre mais fonctionnel théoriquement """
        for k in range(nb_skeletons):
            ligneaplot=ligneaplots[k]
            point=pointss[k]
            invariant=invariants[k]
            data=list_mat_skeletons[k]
            if iteration>=liste_nb_framestotal[k]: #!!!
                continue
            for i,line in enumerate(ligneaplot):
                x1,z1,y1=data[connexion_tuples[i][0]][:,iteration]
                x2,z2,y2=data[connexion_tuples[i][1]][:,iteration]
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
    ani=animation.FuncAnimation(fig,update,frames=range(nb_framestotal), fargs=(list_mat_skeletons,liste_line,liste_point,liste_invariant))
    if write:
        animation.writer=animation.writers['ffmpeg']
        plt.ioff() # Turn off interactive mode just in case
        print("sauvegarde du fichier")
        ani.save( path_folder_save+save_name+'.mp4',fps=15) # A FAIRE PASSER EN MP4
        print("fin de sauvegarde")
    if animate:
        #u may expect strange behavior, only used write
        # it may be really laggy if u dont have a good computer
        plt.ion()
        plt.show()
    return ani


def plot_skeleton(path_skeleton:str=None,save_name='skeleton',title=None,write=True,animate=False,path_folder_save='./videos/',num_body=0):
    """plot le skeleton d'une personne du dataset NTU RGB+D sachant le .skeleton.
    outdated

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
        nv_skeleton=skeleton.transpose(3, 2, 0, 1)[num_body] #! Il st de la forme (nb_joints,3,nb_frames)

    elif path_skeleton.strip().endswith('.npy'):
        skeleton=np.load(path_skeleton,allow_pickle=True).item()
        print(' le plot de skeleton en .npy nest pas encore supporté,unexpected behavior EXPECTED !')
        skeleton=skeleton['b0']
        nv_skeleton=skeleton.transpose(3, 2, 0, 1)[0] #! Il st de la forme (nb_joints,3,nb_frames)

    else:
        assert ValueError('Unexpected file format, expected .skeleton or .npy')
    print("On sauvegarde le skeleton au nom de ",save_name)
    return plot_video_skeletons(list_mat_skeletons=[nv_skeleton],save_name=save_name,title=title,write=write,animate=animate,path_folder_save=path_folder_save)
