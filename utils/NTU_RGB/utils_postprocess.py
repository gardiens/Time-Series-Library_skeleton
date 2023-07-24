from utils.constantes import Args_technique_CPU,Args_technique_GPU
def get_args_from_file(file,args_technique=Args_technique_CPU):
    """ On va retourner le setting sachant le Filefait à la main
    POUR FED SI CEST NTU IL FAUT RAJOUTER DES TRUCS  """
    parser=file.split("_")
    class Args(args_technique):
        def __init__(self):
            super().__init__()
            parser = file.split("_")
       
            self.task_name = "long_term_forecast"
            self.model_id = str(parser[3])
            self.model = str(parser[4])
            self.data = str(parser[5])
            self.features = str(parser[6][2:])
            self.seq_len = int(parser[7][2:])
            self.label_len = int(parser[8][2:])
            self.pred_len = int(parser[9][2:])
            self.d_model = int(parser[10][2:])
            self.n_heads = int(parser[11][2:])
            self.e_layers = int(parser[12][2:])
            self.d_layers = int(parser[13][2:])
            self.d_ff = int(parser[14][2:])
            self.factor = int(parser[15][2:])
            self.embed = str(parser[16][2:])
            self.distil = parser[17][2:] == "True"
            self.des = str(parser[18])
            self.ii = int(parser[19])
            self.get_cat_value = int(parser[20][2:])
            self.get_time_value = int(parser[21][3:])
