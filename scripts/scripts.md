# Script
On garde ici tous les scripts qui permettent de lancer le logiciel sur un cluster. les scripts sont codé en bash et finissent par un .sh. 

# Organisation des dossiers
les dossiers sont stockés par dâtes d'executions. Un fichier utils contient des fichiers qui permettent d'automatiser les dates de récupérations de logs. un template d'envoi de scripts est donné dans le fichier utils.

## Scripts utiles:
J'aimerais résumer ici les différentes informations importantes si jamais on veut lancer un script:


### Lancer un batch sur la machine: 

```console
sbatch --gres=gpu:2 --time=20:00:00 --exclude=n[1-5] --output=logs/DATE/NOMMODEL.stdout --error=logs/DATE/NOMODEL.stderr --job-name=NOMODEL scripts/DATE/NOMODEL/NOMODEL.sh
```
pour l'utilisation des scripts:
-Si vous voulez mettre en place automatiquement Tboard il faut faire:
bash TensorBoard_run.sh

pour add/commit push automatiquement avec git:
```
git cmp " votremessage"
```
Il est très important d'excluer les noeuds 1 à 5 car il n'ont pas la même version de cuda.

### Lancer Tensorboard :
Il existe deux manières:
```console
sh Tensorboard_run.sh
```
cette méthode peut ne pas marcher certaines fois.
```console
ssh -L 16007:127.0.0.1:7970 USERNAME@slurm-ext
tensorboard --logdir=runs --host=localhost --port=7970
```
ensuite ouvrez [http://localhost:16007](http://localhost:16007)

### récupérer les logs 
pour récupérer les logs:
```console
scp -r bournez@slurm-ext:/mnt/beegfs/home/bournez/babygarches_rambaud/ Time-Series-Library_babygarches/logs/./logs
```


###  pour récupérer les videos
```console
sh scripts/recup_videos NOM-TRAINING NOM RUNS
```
( avec NOM training le nom du dossier qui va être sauvegardé et NOM-run un identifiant qui permet de récupérer tous les fichiers sur la remote machine )

### plus technique: envoyer un nouveau summary_csv
Cela peut être intéressant si on veut changer le début de lancement des frames ou le stockage du dataframe qui résume les .skeletons.
pour envoyer summary_csv:
```console
scp  ./dataset/NTU_RGB+D/summary_NTU/summary_NTU.csv slurm-ext:./dataset/NTU_RGB+D/summary_NTU/
```
```console
scp  ./dataset/NTU_RGB+D/summary_NTU/liste_NTU_skeleton_4.csv slurm-ext:./dataset/NTU_RGB+D/summary_NTU/
```


## Explication des différents paramètres dans un script. sh
Partie technique, nous allons expliquer ici  l'importance de chauqe paramètre dans le lancement de FEDFormer et NTU_RGB+D
| nom du paramètre 	| valeur par défaut 	| explication du paramètre         	| remarque                                                                                                                                 	|   	|
|------------------	|-------------------	|----------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------	|---	|
| pred_len         	| 32                	| longuer à prédire du modèle      	| dans mon implémentation  pred_len= séquence de base+longueur séquence à prédire                                                          	|   	|
| label_len        	| 32                	| Longueur des labels.             	| label_len= séquence de base+ longueur séquence à prédire                                                                                 	|   	|
| seq_len          	| 16                	| longueur de la séquence d'entrée 	|                                                                                                                                          	|   	|
| model            	| FEDformer         	| nom du modèle                    	|                                                                                                                                          	|   	|
| enc_in/dec_in    	| 75                	| nombre de channel d'entrée       	| Il semble relativement obscure d'être capable d'avoir un enc_in différent de c_out. Il y a des bugs dans le modèle                       	|   	|
| c_out            	| 75                	| nombre de channel en sortie      	|                                                                                                                                          	|   	|
| lradj            	| sem_constant      	| nom de l'agorithme d'ajustement  	|                                                                                                                                          	|   	|
| preprocess       	| 1                 	|                                  	| un peu technique mais 0 c'est pour un setting du 1er aout, 1 pour des settings normaux et 2 pour des implémentations de temps en temps.  	|   	|
| split_train_test 	| action            	| nom de la séparation des dataset 	|                                                                                                                                          	|   	|

