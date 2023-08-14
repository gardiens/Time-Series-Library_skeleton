## Models   

Ce dossier contient tous les modèles implémentés par la librairie initiale.
Il est à noté la présence temproaire d'un modèle supplémentaire dit "MetaFormer" qui prédit indépendament différentes parties de membres.

## Metaformer: aspect technique
Avant de lancer le train de Metaformer, il faut déjà entrainer les différents membres, pour cela il faut entrainer le  sous-modèle sur les Dataset NTU_arm,NTU_leg,NTU_body et ensuite faire marcher Metaformer sans phase de train 