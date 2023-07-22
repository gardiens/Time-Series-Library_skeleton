Pour la sortie des modèles:

Je crois que la sortie d'un modèle est pour NTU_RGB ( batch_size,longueur prédite (pred_len),75=(nb_joints*nb_dimensions))


FEDformer ( et les uatres modèles sont de la forme ):
Input: (batch_size, seq_len,nb_channel(75))
Output:(Batch_size,pred_len,nb_channel)