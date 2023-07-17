import matplotlib.pyplot as plt
import numpy as np
import os
from utils_NTU import read_xyz
from joints import connexion_tuples
import matplotlib.animation as animation
import mpl_toolkits.mplot3d as plt3d
""" Tuto clair ici https://acme.byu.edu/00000179-d3f1-d7a6-a5fb-ffff6a210001/animation-pdf"""
"""path="./raw/S018C001P045R001A073.skeleton"
skeleton=read_xyz(path)
"""

def plot_skeleton(path_skeleton:str,save_name='skeleton',title=None,write=True,animate=False,path_folder_save='./videos/'):
    """plot le skeleton d'une personne du dataset NTU RGB+D

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

    # if path_folder is not in the directory, create it
    if not os.path.exists(path_folder_save):
        os.makedirs(path_folder_save)
    #* Parsing the skeleton
    if path_skeleton.strip().endswith('.skeleton'):
        skeleton = read_xyz(path_skeleton)
    elif path_skeleton.strip().endswith('.npy'):
        skeleton=np.load(path,allow_pickle=True).item()
        print(' le plot de skeleton en .npy nest pas encore supporté,unexpected behavior EXPECTED !')
        skeleton=skeleton['data0']
    else:
        assert ValueError('Unexpected file format, expected .skeleton or .npy')
       
    
    nv_skeleton=skeleton.transpose(3, 2, 0, 1)[0]
    def update(iteration,data,ligneaplot,points,invariant):
        """ On mets a jour le plot"""
        for i,line in enumerate(ligneaplot):
            #print(i,line)
            x1,z1,y1=data[connexion_tuples[i][0]][:,iteration]
            x2,z2,y2=data[connexion_tuples[i][1]][:,iteration]
            #print("oui")
            x1,x2=min(x1,x2),max(x1,x2)
            line.set_data(np.linspace(x1,x2,num=20),np.linspace(y1,y2,num=20))
            #print("ici")
            line.set_3d_properties(np.linspace(z1,z2,num=20))
            #print("et non")
        for i,point in enumerate(points): #plot les points
            x,z,y=data[i][:,iteration]
            point.set_data(x,y)
            point.set_3d_properties(z)
        
        x_spine_mid = data[1][x_coord][iteration] # 1 correspond au spine mid exactement
        y_spine_mid = data[1][y_coord][iteration]
        z_spine_mid = data[1][z_coord][iteration]
        #print(x_spine_mid,y_spine_mid,z_spine_mid)
        invariant[0].set_data(np.linspace(x_spine_mid,x_spine_mid+axis_length,num=20),np.linspace(y_spine_mid,y_spine_mid,num=20))
        invariant[0].set_3d_properties(np.linspace(z_spine_mid,z_spine_mid,num=20))
        invariant[1].set_data(np.linspace(x_spine_mid,x_spine_mid,num=20),np.linspace(y_spine_mid,y_spine_mid+axis_length,num=20))
        invariant[1].set_3d_properties(np.linspace(z_spine_mid,z_spine_mid,num=20))
        invariant[2].set_data(np.linspace(x_spine_mid,x_spine_mid,num=20),np.linspace(y_spine_mid,y_spine_mid,num=20))
        invariant[2].set_3d_properties(np.linspace(z_spine_mid,z_spine_mid+axis_length,num=20))
        


        invariant[3].set_data(np.linspace(0,x_spine_mid,num=20),np.linspace(0,y_spine_mid,num=20))
        
        invariant[3].set_3d_properties(np.linspace(0,z_spine_mid,num=20))
        
        #* des points qui ne bougent pas 
    
        #print("ON A FINI? ",ligneaplot)
        #plt.plot(ligneaplot)
        return ligneaplot,points,invariant
    #*Define the overall figure
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.set_xlim3d(np.amin(nv_skeleton[:,0,:]),np.amax(nv_skeleton[:,0,:]))
    ax.set_xlabel('X')
    ax.set_ylim3d(np.amin(nv_skeleton[:,1,:]),np.amax(nv_skeleton[:,1,:]))
    ax.set_ylabel('Y')
    ax.set_zlim3d(np.amin(nv_skeleton[:,2,:]),np.amax(nv_skeleton[:,2,:]))
    ax.set_zlabel('Z')

    #for nice view
    ax.view_init(10,50)
    axis_length = 0.2
        
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
    ax.set_xlim3d(min(np.amin(nv_skeleton[:,0,:]),-axis_length), max(np.amax(nv_skeleton[:, 0, :]), axis_length))
    ax.set_ylim3d(min(np.amin(nv_skeleton[:, 2, :]),-axis_length), max(np.amax(nv_skeleton[:, 2, :]), axis_length))
    ax.set_zlim3d(min(np.amin(nv_skeleton[:, 1, :]),-axis_length), max(np.amax(nv_skeleton[:, 1, :]), axis_length))
    if title!=None: #* set a title.
        ax.text2D(0.05, 0.95,title, transform=ax.transAxes)
    else:
        ax.text2D(0.05, 0.95,save_name, transform=ax.transAxes)
    x_coord = int(0)
    y_coord = int(2)
    z_coord = int(1) # true coord?
    """ Tuto clair ici https://acme.byu.edu/00000179-d3f1-d7a6-a5fb-ffff6a210001/animation-pdf"""
    #*Define the lines and points to animate

    lines= [ax.plot([],[],[],linestyle=':',color='blue')[0] for _ in connexion_tuples]
    points=[ax.plot([],[],[],color='blue',marker='o',markersize=3)[0] for _ in range(25)]
    invariant=[ax.plot([0],[0],[0],color='green')[0] for k in range(4)]
    invariant[3].set_color('black')
    invariant[3].set_linestyle(  (0, (1, 10)))
    ani=animation.FuncAnimation(fig,update,frames=range(nv_skeleton.shape[2]), fargs=(nv_skeleton,lines,points,invariant))
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
