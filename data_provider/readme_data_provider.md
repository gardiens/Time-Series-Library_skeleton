# Data_provider 

Ce dossier contient 4 fichiers: data_factory ,data_loade, m4, et uea.py 
Pour la  construction du dataset seulement  data_factory et data_loader sont nécessaires.

## Démarrer: les commandes utiles 
Pour toutes les commandes suivantes, on suppose que vous ayez initialiser une classe args. 
Cela peut se faire en créeant une classe args ou en s'inspirant des nombreux scripts .sh 
# récupérer un data_set 

```python
# main.py
from data_provider.data_factory import data_provider
args=Args()
data_set, data_loader=data_provider(args,flag="train") 
```
# récupérer un échantillon
```python
sample_name="S001C001P001R001A001"
entry=data_set.get_data_from_sample_name()
entry_batch=data_set.get_input_model(entry)
``` 

##  Organisation des fichiers 

# data_factory 
data_factory contient data_provider  qui permets de récupérer le dataset et loader de torch. 
Il contient aussi le data_dict qui énumère tous les datasets possibles pour cette librairie. 
# Data_loader 
data_loader contient toutes les implémentations de dataset disponibles
- NTURGB l'implémentation du dataset NTU_RGBD. Il est à noter qu'il existe des extensions de ce dataset sous le nom NTU_leg, NTU_arm, NTU_body qui permettent de récupérer uniquement certains membres des squelettes. Ce dataset utilise des fonctions présente dans utils/NTU_RGB  surtout pour le preprocessing et l'obtention des items du dataset.
- Il existe aussi les datasets initiaux: ETT,PMSEG,MSLeg,SWATSegLoader,UEAloader


##Comment rajouter mon propre dataset?
Pour rajouter son propre dataset, il faut créer un dataset pytorch  qui va posséder les deux attributs fondamentaux __getitem__ et __len__. Voir [ceci](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) pour plus d'explication.
Dans la fonction __getitem__ , il doit produire en sortie au moins 4 matrices qui sont:
- entry, une matrice de la série temporelle de départ. Elle est de la forme (seq_len,nb_channel)
- label, la matrice de la série temporelle  à prédire. Elle est de la forme (pred_len,nb_channel)
- time_value_enc, matrice de la série temporelle qui encode le temps en entrée.Il est de la forme (seq_len)
- time_value_dec, matrice de la série temporelle qui encode le temps pour le label. Il est de la forme (pred_len).


## Quels sont les data augmentation disponible?
Nous proposons 4 type de data augmentation possible: 
1. Une prédiction type « backward », sachant une séquence à prédire et son label, nous inversons l’échelle du temps et le modèle doit alors prédire lors du training la séquence d’entrée sachant le label. Cela revient à utiliser la « réciprocité du mouvement ».
2.  Une prédiction type « random flip », nous allons faire des rotations de bodys sur des angles faibles [+-17 degré] pour rajouter de nouvelles « scènes » où sont pris les vidéos. Cette idée est motivée par [1]
3. Une prédiction type « Mixup ». Inspiré par [2], l’idée est de prendre deux squelettes d’humains et de concaténer le haut du premier squelette avec le bas du second squelette. Cela va forcer la prédiction à être plus riche et peut créer des mouvements plus complexes.
4. Une prédiction type "Cutmix " [3] Cette prédiction est un raffinement de Mixup : On va prendre cette fois non une séparation (haut/bas) mais on va à la place prendre deux échantillons puis prendre aléatoirement des membres des deux personnes pour en faire un nouveaux sample
Concernant l’application de ces augmentations, on peut les tester uniformément (on en applique à chaque fois) ou avec des Policy particulières comme Rand Augment ou Fast Rand Augment [4]. Dans notre cas, nous n’avons fait que les appliquer individuellement sur les samples initiaux. Il est aussi à noter que nous prenons les données augmentées ET les données initiales, une autre éventualité aurait été de transformer les données initiales par ces transformations. 

Pour pouvoir les utiliser, il suffit d'ajouter en argument à main.py --augment et de mettre "--prop 0.05,0.05,0.05,0.05". le premier est la proportion de data augmentée pour backward, le second flip, le troisième cutmix et le dernier Randmix. Il est à noter que notre implémentation <b> ajoute</b> les données augmentées au dataset et non les transforme.


##Todo: 
Pour améliorer la rapidité des modèles sur NTU_RGB, on pourrait faire tous le preprocessing des données en créeant une nouvelle colonne du dataframe et récupérer ensuite l'objet, cela augmenterait grandement la compléxité temporelle au détriment d'une légère complexité spatiale.
Il faut aussi reconcevoir le preprocessing qui est pour l'instant très lourd en implémentation


## Biblio:
[1] Augmented Skeleton Based Contrastive Action Learning with Momentum LSTM for Unsupervised Action Recognition
[2] Topology-aware Convolutional Neural Network for Efficient Skeleton-based Action Recognition

[4]https://github.com/ildoonet/pytorch-randaugment/tree/master/RandAugment
