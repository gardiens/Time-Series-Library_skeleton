""" Tous les outils pour update ou non les writers de tensorboard"""


def update_writer(writer):
    pass


def add_hparams(writer,args):#TODO: NE MARCHE PAS MAISOSEF 
    writer.add_hparams({


        "task_name": args.task_name,
        "model_id": args.model_id,
        "model": args.model,
        "data": args.data,
        "features": args.features,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "factor": args.factor,
        "embed": args.embed,
        "distil": args.distil,
        "des": args.des,
        "num_itr": args.num_itr,
        "get_cat_value": args.get_cat_value,
        "get_time_value": args.get_time_value
    }, {"hparam/test accuracy": 0})

