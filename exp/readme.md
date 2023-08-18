## exp 
Ce dossier contient toute la pipeline de training et de test de ce repo.
Les deux fichiers importants sont exp_long_term_forecasting qui implémente la boucle de train et de test.
Exp_basic permet de récupérer le dataset et le modèle à l'aide des dictionnaires idiones. 

Les sorties d'une boucle de train et de test sont de trois natures:
1. les runs du training sont stockés dans le dossier runs et sont facilement accessible avec Tensorboard
2. des vidéos à prédire sont stockés dans le fichier test_results. Le squelette bleu est le ground truth et le squelette rouge est la prédiction.
3.  un dataframe avec les loss pour chaque sample est disponible dans le dossier results.