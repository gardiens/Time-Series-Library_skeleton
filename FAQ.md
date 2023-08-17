# FAQ: questions techniques et autres

### Comment tester rapidement un modèle?
Voir commande_utile  Run of the model:FED pour plus d'explications. Il ne faut pas oublier de choisir au début un arg.


### Quels modèles recommandez vous d'essayer?

Nous avons testé  lors de stage principalement AutoFormer,FEDFormer et légèrement NTS. Nous recommandons FEDformer qui donne des résultats décents.  Concernant les autres modèles, nous tenons à signaler qu'ils semblent relativement peu optimiser car la compléxité temporelle semble non néggligeable...
###  Comment rajouter rapidement un modèle à l'implémentation? 
Il suffit d'implémenter un torch.nn module avec une méthode forward. Voir [ici](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)  pour plus d'informations

# Dataset 
###  Comment faire de la Data Augmentation?
Nous proposons 4 type de data augmentation possible: 
1. Une prédiction type « backward », sachant une séquence à prédire et son label, nous inversons l’échelle du temps et le modèle doit alors prédire lors du training la séquence d’entrée sachant le label. 
2.  Une prédiction type « random flip », nous allons faire des rotations de bodys sur des angles faibles [+-17 degré] pour rajouter de nouvelles « scènes » où sont pris les vidéos. Cette idée est motivée par [1]
3. Une prédiction type « Mixup ». Inspiré par [2], l’idée est de prendre deux squelettes d’humains et de concaténer le haut du premier squelette avec le bas du second squelette
4. Une prédiction type "Cutmix " [3] Cette prédiction est un raffinement de Mixup : On va prendre cette fois non une séparation (haut/bas) mais on va à la place prendre deux échantillons puis prendre aléatoirement des membres des deux personnes pour en faire un nouveaux sample

Pour pouvoir les utiliser, il suffit d'ajouter en argument à main.py --augment et de mettre "--prop 0.05,0.05,0.05,0.05". le premier est backward, le second flip, le troisième cutmix et le dernier Randmix. Il est à noter que notre implémentation <b> ajoute</b> les données augmentées au dataset et non les transforme.

### J'aimerais avoir un nom de channel différent en entrée/sortie
Je n'ai pas compris exactement la relation entre enc_in,dec_in et c_out. Pour simplifier le problème, je crois avoir compris que enc_in et dec_in doivent toujours avoir la même valeur pour FEDFormer et j'ai fait à la fin une projection linéaire dans le cas où c_out est plus petit ( ou grand).


### Pourquoi stocker un dataframe intermédiaire des données?
On a besoin de récupérer des informations intermédiaires sur les skeletons comme la longueur totale de la vidéo, le nombre de personne sur chaque vidéos... plutôt que de  calculer ça à chaque fois, nous avons fait un dataframe qui permet en quelque seconde d'obtenir toutes ces informations.

###  Comment rajouter rapidement un Dataset à l'implémentation? 
Pour rajouter son propre dataset, il faut créer un dataset pytorch  qui va posséder les deux attributs fondamentaux __getitem__ et __len__. Voir [ceci](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) pour plus d'explication.
Dans la fonction __getitem__ , il doit produire en sortie au moins 4 matrices qui sont:
- entry, une matrice de la série temporelle de départ. Elle est de la forme (seq_len,nb_channel)
- label, la matrice de la série temporelle  à prédire. Elle est de la forme (pred_len,nb_channel)
- time_value_enc, matrice de la série temporelle qui encode le temps en entrée.Il est de la forme (seq_len,)
- time_value_dec, matrice de la série temporelle qui encode le temps pour le label. Il est de la forme (pred_len,).



# Modèle 

### Quel est la différence entre seq_len, pred_len et label_len?
seq_len correspond à la longueur à prédire, pred_len à la longueur à prédire du modèle et label_len la longueure effectivement prédite des labels issus du dataset. Contrairement à l'implémentation initiale, nous supposons toujours que pred_len>label_len.
###  Que contient le string setting? Que correspondent chaque entier? 
 les modèles sont sauvegardé selon leur nom de settings, voici ce que chaque caractère représente: 
 ```console
 long_term_forecast_model-id_FEDformer_NTU_ftM_sl16_ll32_pl32_dm512_nh8_el2_dl1_df2048_fc3_ebtimeNTU_dtTrue_Exp_0_cv0_tvv1_p3
 ```
- long_term_forecast:  le nom de l'expérience qu'on utilise. Normalement ce sera toujours long_term_forecast (sinon il va y avoir des bugs dans get_settings)
- model-id: l'id du modèle qu'on expérimente. C'est le nom auquel on va faire référence
- FEDformer: nom en string du modèle
-NTU: nom du dataset
-ftM: indique si on prédit en varié ou multivarié, noramlement on ne touchera pas à ça
-sl: séquence len ,longueur de l'input en entrée
-ll: label len, longueur du label
-pl:longueur de prédiction
-dm: dimension du modèle, je crois que c'est la dimension de l'espace intermédiaire entre les layers
-nh:  je ne sais pas
-el: nombre de layers de l'encodeur
dl: nombre de layer du décodeur
-df: dimension du modèle, je crois que c'est la dimension de l'espace intermédiaire entre les layers
-fc: je crois que cela représente la window average utilisé pour le serie decompositionde certains modèle
-ebtime: comment on encode le temps, NTU signifie qu'on nefait rien 
-dt: 
-EXP: nom de l'expérience
-0: nombre d'itération, globalement un boulet
-cv: 1 si on prend les categorical value, 0 sinon
-tv: 1 si on récupère dans le dataset l'encodage du temps , 0 sinon
- p: preprocesisng. 1 est le preprocessing recomandé pour FEfformer et cela modifie juste comment on preprocess le skeleton en entrée.
### Tensorboard comment c'est implémenté?
Tensorboad est initialisé au début de Exp et nous ajoutons au modèle les loss de train/vali/test à chaque epoch.

# Bugs ou techniques utiles.


### cannot import distutils.version 
Prends tes jambes à ton cou et  inverse ton update de version de cuda :
```console
conda list --revisions
conda --revision N 
```
Sinon il faut chercher sur le net mais il s'agit d'un bug présent sur une version de pytorch mais patcher ensuite.




### C'est qui ce cupy dans les import?

 Numpy ne marche pas sur GPU, donc pour accélerer le training il fallait soit passer en torch soit utiliser une autre librairie. Cupy a l'avantage d'avoir une interface proche de Numpy et d'être GPU friendly sans aucune réecriture, nous avons donc adopté cette solution.




### Mon modèle sort " double precision and not float precision" ou un truc du genre
Il faut faire attention à la précision du modèle. dans la boucle de train nous ne gardons q'une précision en float, il faut alors transformer les entrées en float
```py 
#from commande_utile
from models.FEDformer import *
network=Model(args)
network.float()
# Batch recovery
from data_provider.data_factory import data_provider
args=Args()
data_set, data_loader=data_provider(args,flag="train") 

(batch_x, batch_y, batch_x_mark, batch_y_mark)=enumerate(data_loader).__next__()[1]
y=network(batch_x.float(), batch_x_mark.float(),None, batch_y_mark.float())
``` 



###  Je veux plus d'info sur une fonction?
toutes les fonctions utilisées pour NTU_RGB sont normalement commentées, contacte moi sinon pour toute précision supplémentaire.
###  Je veux plus d'info sur un dossier
clique sur le dossier et il y a normalement un readme explicatif
### Je veux rentrer plus profondément dans le code rapidement comment faire?
Regarde commande_utile.ipynb qui résume pas mal d'opérations qu'on peut faire rapidement. Pour aller plus loin, toutes les fonctions sont commentées.


###  Quels sont les sorties d'un modèle?
Les sorties d'une boucle de train et de test sont de trois natures:
1. les runs du training sont stockés dans le dossier runs et sont facilement accessible avec Tensorboard
2. des vidéos à prédire sont stockés dans le fichier test_results. Le squelette bleu est le ground truth et le squelette rouge est la prédiction.
3.  un dataframe avec les loss pour chaque sample est disponible dans le dossier results.

### Comment marche plot_skeleton
 Si on veut plot les squelettes deux possibilités s'offre à nous:
 - On veut plot par rapport à un nom de squelette,dans ce cas on utilise la fonction plot_skeleton de utils/NTU_RGB/plot_skeleton
 ```py 
 plot_skeleton(path_skeleton=Nom_du_skeleton,save_name=nom du fichier sauvegardé,num_body: numéro du corps correspondant, path_folder_save=quel dossier on va sauvegardé)
 ```
 - Si on veut faire en sortie du modèle il preprocess un peu les données et utiliser plot_video_skeleton
 ```py 
 plot_video_skeletons(list_mat_skeleton= liste des array desskeleton qu'on veut plot de la bonne manière, save_name=nom du fichier sauvegardé, path_folder_save= nom du dossier où on veut sauvegardé.)
```

### Pourquoi c'est aussi mal codé?
Je ne suis pas encore employé chez google et il y a peut-être une raison....