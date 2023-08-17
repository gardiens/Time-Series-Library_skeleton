##models    


C'est ici que sont stockés tous les modèles qu'on va tester sur nos datasets. 
Les modèles sont écrits en torch et prennent en entrée des tensor de la forme (nb_batch,nb_frames,nb_channels).
Les briques de ces modèles sont stockés dans le dossier layer.

Attention, la complexité temporelle des différentes architectures est très variable, autant Autoformer et FEDFormer sont rapide, autant certains modèles comme TimesNet peuvent prendre beaucoup de temps.