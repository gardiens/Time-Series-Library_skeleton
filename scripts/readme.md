J'aimerais résumer ici les différentes informations importantes si jamais on veut lancer un script:



Pour les inputs à mettre:
- Il faut que -pred_len et label_len soit la même chose ( pred_len sert à la dimension de l'output et label_len la dimension du label du dataset de sortie)


pour l'utilisation des scripts:
-Si vous voulez mettre en place automatiquement Tboard il faut faire:
bash TensorBoard_run.sh

pour add/commit push automatiquement avec git:
git cmp " votremessage"

normalement le pull est automatique chez slurm...


pour récupérer les logs:
scp -r bournez@slurm-ext:/mnt/beegfs/home/bournez/babygarches_rambaud/Time-Series-Library_babygarches/logs/./logs



pour envoyer summary_csv:

scp  ./dataset/NTU_RGB+D/summary_NTU/summary_NTU.csv slurm-ext:./dataset/NTU_RGB+D/summary_NTU/

scp  ./dataset/NTU_RGB+D/summary_NTU/liste_NTU_skeleton_4.csv slurm-ext:./dataset/NTU_RGB+D/summary_NTU/