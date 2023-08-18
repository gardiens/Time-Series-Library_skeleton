
##models    


C'est ici que sont stockés tous les modèles qu'on va tester sur nos datasets. 
Les modèles sont écrits en torch et prennent en entrée des tensor de la forme (nb_batch,nb_frames,nb_channels).
Les briques de ces modèles sont stockés dans le dossier layer.


### Remarque sur les modèles :
<b>Complexité temporelle </b>: la complexité temporelle des différentes architectures est très variable, autant AutoFormer et FEDFormer sont rapide ,autant certains modèle ciomme Pyraformer ou TimesNet peuvent prendre beaucoup de temps. je suspecte qu'il y a un problème d'implémentation plus que d'un problème de modèle.

Attention, la complexité temporelle des différentes architectures est très variable, autant Autoformer et FEDFormer sont rapide, autant certains modèles comme TimesNet peuvent prendre beaucoup de temps.

<b> Complexité du modèle</b>: Je n'ai pas d'argument concret pour étayer ma thèse, mais je suspecte qu'une simplification de certains modèles permettraient d'améliorer la qualité du modèle.J'ai testé de benchmarker mes modèles à la fin de mon stage et j'ai enlevé une simplification de l'auto-corrélation d'Autoformer. J'ai peut-être fait d'aute modifcation entre temps mais j'ai l'impression que le rajout de la simplification a fortement détérioré la qualité du modèle.


Pour les autres remarques, se référer à la conclusion