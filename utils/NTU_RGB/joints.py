r"""
Contains a help *Joints* class which maps each Kinect v2 index with its name. Also provides a **connexion_tuples** np
array which contains all neighboring joints.

"""
from enum import IntEnum
import numpy as np


class Joints(IntEnum):
    r"""Maps each Kinect v2 joint name to its corresponding index. See
    https://medium.com/@lisajamhoury/understanding-kinect-v2-joints-and-coordinate-system-4f4b90b9df16 for joints infos.

    """
    SPINEBASE = 0
    SPINEMID = 1
    NECK = 2
    HEAD = 3
    
    SHOULDERLEFT = 4
    ELBOWLEFT = 5
    WRISTLEFT = 6
    HANDLEFT = 7

    SHOULDERRIGHT = 8
    ELBOWRIGHT = 9
    WRISTRIGHT = 10
    HANDRIGHT = 11

    HIPLEFT = 12
    KNEELEFT = 13
    ANKLELEFT = 14
    FOOTLEFT = 15

    HIPRIGHT = 16
    KNEERIGHT = 17
    ANKLERIGHT = 18
    FOOTRIGHT = 19

    SPINESHOULDER = 20

    HANDTIPLEFT = 21
    THUMBLEFT = 22

    HANDTIPRIGHT = 23
    THUMBRIGHT = 24


# shape (n_connexions, 2)
connexion_tuples = np.array([[Joints.SPINEBASE, Joints.SPINEMID],
                             [Joints.SPINEMID, Joints.SPINESHOULDER],
                             [Joints.SPINESHOULDER, Joints.NECK],
                             [Joints.NECK, Joints.HEAD],

                             [Joints.SPINESHOULDER, Joints.SHOULDERLEFT], # 4
                             [Joints.SHOULDERLEFT, Joints.ELBOWLEFT],
                             [Joints.ELBOWLEFT, Joints.WRISTLEFT],
                             [Joints.WRISTLEFT, Joints.HANDLEFT],
                             [Joints.HANDLEFT, Joints.HANDTIPLEFT],
                             [Joints.HANDLEFT, Joints.THUMBLEFT],

                             [Joints.SPINESHOULDER, Joints.SHOULDERRIGHT], # 10
                             [Joints.SHOULDERRIGHT, Joints.ELBOWRIGHT],
                             [Joints.ELBOWRIGHT, Joints.WRISTRIGHT],
                             [Joints.WRISTRIGHT, Joints.HANDRIGHT],
                             [Joints.HANDRIGHT, Joints.HANDTIPRIGHT],
                             [Joints.HANDRIGHT, Joints.THUMBRIGHT],

                             [Joints.SPINEBASE, Joints.HIPRIGHT], # 16
                             [Joints.HIPRIGHT, Joints.KNEERIGHT],
                             [Joints.KNEERIGHT, Joints.ANKLERIGHT],
                             [Joints.ANKLERIGHT, Joints.FOOTRIGHT],

                             [Joints.SPINEBASE, Joints.HIPLEFT], # 20
                             [Joints.HIPLEFT, Joints.KNEELEFT],
                             [Joints.KNEELEFT, Joints.ANKLELEFT],
                             [Joints.ANKLELEFT, Joints.FOOTLEFT]])




""" 
import matplotlib.pyplot as plt
import numpy as np
from joints import Joints
from utils_NTU import read_xyz
from joints import connexion_tuples
import matplotlib.animation as animation

path="./raw/S018C001P008R001A061.skeleton"
skeleton=read_xyz(path)
nv_skeleton=skeleton.transpose(3, 2, 0, 1)[0]

#* Définir la figure
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
#On définit les ax properties
ax.set_xlim3d([0,10])
ax.set_xlabel('X')
ax.set_ylim3d([0,10])
ax.set_ylabel('Y')
ax.set_zlim3d([0,10])
ax.set_zlabel('Z')


from joints import connexion_tuples
#Connexion tuple est une liste de liste...
# On définit les lignes qu'on veut plot
 Tuto clair ici https://acme.byu.edu/00000179-d3f1-d7a6-a5fb-ffff6a210001/animation-pdf
lines= [ax.plot([],[],[])[0] for _ in connexion_tuples]

def update(iteration,data,ligneaplot):
     On mets a jour le plot
    for i,line in enumerate(ligneaplot):
        #print(i,line)
        x1,y1,z1=data[connexion_tuples[i][0]][:,iteration]
        x2,y2,z2=data[connexion_tuples[i][1]][:,iteration]
        #print("oui")
        x1,x2=min(x1,x2),max(x1,x2)
        line.set_data(np.linspace(x1,x2,num=20),np.linspace(y1,y2,num=20))
        #print("ici")
        line.set_3d_properties(np.linspace(z1,z2,num=20))
        #print("et non")
        
    print("ON A FINI? ",ligneaplot)
    return ligneaplot
animation.writer=animation.writers['ffmpeg']
plt.ioff() # Turn off interactive mode just in case
ani=animation.FuncAnimation(fig,update,frames=range(nv_skeleton.shape[2]), fargs=(nv_skeleton,lines),interval=1/30,)
ani.save('./test_ani.mp4') # A FAIRE PASSER EN MP4
"""