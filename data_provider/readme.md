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
- NTURGB l'implémentation du dataset NTU_RGBD. Il est à noter qu'il existe des extensions de ce dataset sous le nom NTU_leg, NTU_arm, NTU_body qui permettent de récupérer uniquement certains membres des squelettes. Ce dataset utilise des fonctions présente dans utils/NTU_RGB
- Il existe aussi les datasets initiaux: ETT,PMSEG,MSLeg,SWATSegLoader,UEAloader




##Todo: 
Pour améliorer la rapidité des modèles sur NTU_RGB, on pourrait faire tous le preprocessing des données en créeant une nouvelle colonne du dataframe et récupérer ensuite l'objet, cela augmenterait grandement la compléxité temporelle...