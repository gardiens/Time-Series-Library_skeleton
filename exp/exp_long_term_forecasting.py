from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from setuptools import distutils
from utils.NTU_RGB.plot_skeleton import plot_video_skeletons,plot_skeleton
from utils.constantes import get_settings,get_args_from_filename
from torch.utils.tensorboard import SummaryWriter
from utils.losses import mape_loss, mase_loss, smape_loss

warnings.filterwarnings('ignore')

from utils.NTU_RGB.tensorboard import add_hparams
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        setting = get_settings(args)
                
        self.setting=setting
        self.writer=SummaryWriter(log_dir=f"runs/{setting}") #* tensorboard
        writer=self.writer
        #print("les args",vars(args))
        add_hparams(self.writer,args) #* tensorboard

        # Add in the writer every parameter which are float,int,str or bool
        #writer.add_hparams({k:v for k,v in vars(args).items() if type(v) in [float,int,str,bool]},{})
        writer.flush()
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self,loss_name="MSE"):
        if  loss_name == 'MSE':
            criterion=nn.MSELoss()
            

        elif loss_name == 'SMAPE':
            criterion=smape_loss
        else:
            criterion=nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
        
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """ Les sauvegardes ont lieu dans early stopping ( dans le dossier/ results/settings, en revanche on a pas le nom du checkpoint clair à priori...)"""
        print("-----loading du dataset ---",flush=True)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print("vérification du dataset:",train_data.liste_path[["filename","debut_frame"]].head())
        print("-----fin du loading du dataset ---",flush=True)
        writer=self.writer
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        n_steps_per_epoch = len(train_data)//self.args.batch_size #* A VERIFIEr
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        print("----- début du training-----",flush=True)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp: #*Jamais utilisé en pratique
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0 #! ATTENTION LA ? 
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    #print(" la len",self.args.pred_len)
                    
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    #print("la len",self.args.pred_len)
                    #print("les batchs",outputs.shape,batch_y.shape)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                
                """metrics = {'train_loss': train_loss,'epoch': epoch}
                if i+1<n_steps_per_epoch:
                    wandb.log(metrics)"""
                


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item(),),flush=True)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time),flush=True)
                    iter_count = 0
                    time_now = time.time()
                    break #!!!!!!!!
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            writer.add_scalar("Loss/train",train_loss,epoch)
            print(vali_loss)
            writer.add_scalar("Loss/vali",vali_loss,epoch)
            writer.add_scalar("Loss/test",test_loss,epoch)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            #* Wandb things 
            early_stopping(vali_loss, self.model, path) #C'est ici qu'on sauvegarde le modèle !!!! avec le path. Il est sauvegardé de la frome path/SETTING !!
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            #* Wandb things
           
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        writer.flush()
        return self.model   

    def test(self, setting, test=0):
        train_data,train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            try: #* On load le check 
                if torch.cuda.is_available():
                    self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
                else:
                    self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),map_location=torch.device('cpu')))
            except FileNotFoundError:
                args1=get_args_from_filename(setting,self.args)
                args1.get_cat_value="_"+str(args1.get_cat_value)
                setting1=get_settings(args1)
                if torch.cuda.is_available():
                    self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting1, 'checkpoint.pth')))
                else:
                    self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting1, 'checkpoint.pth'),map_location=torch.device('cpu')))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 400 == 0: # METTRE 200 sinon 
                    """input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0) # On récupère lesignal avec le DERNIER CHANNEL UNIQUEMENT
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0) #ON RECUPERE LE SIGNAL AVEC LE DERNIER CHANNEL UNIQUEMENT
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))"""
                    input = batch_x.detach().cpu().numpy()
                    y_out=pred[0,:,:]
                    y_true=batch_y[0,:,:]
                    data_set=test_data
                    y_out=data_set.inverse_transform_data(y_out)
                    y_true=data_set.inverse_transform_data(y_true)
                    X_pred=y_out #np.concatenate((X,y_out),axis=0) # Quelle axis?
                    X_true=y_true #np.concatenate((X,y_true),axis=0)
                    #* plot_video_skeletons demande :  (nb_joints,3,nb_frames) et pour l'instant on a ( nb_frames,nb_joints,3) 
                    
                    X_pred=X_pred.transpose(1,2,0)
                    X_true=X_true.transpose(1,2,0)
                    #* On va plot les résultats
                    plot_video_skeletons(mat_skeletons=[X_true,X_pred],save_name="label:"+self.args.model_id+str(i),path_folder_save=os.path.join(folder_path))
                    filename=str(test_data.liste_path["filename"].iloc[i]) # ???
                    plot_skeleton(path_skeleton=os.path.join(self.args.root_path,"raw/",filename+".skeleton"),save_name=str(i)+filename,path_folder_save=folder_path)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        df=test_data.liste_path
        df["mae"]=mae
        df["mse"]=mse
        df["rmse"]=rmse
        df["mape"]=mape
        df["mspe"]=mspe
        df.to_csv(os.path.join(folder_path,"results_df_test.csv"))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics_test.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred_test.npy', preds)
        np.save(folder_path + 'true_test.npy', trues)


        #* On plot aussi le train 
        preds = []
        trues = []
        train_data,train_loader = self._get_data(flag='train')
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 400 == 0: # METTRE 200 sinon 
            
                    input = batch_x.detach().cpu().numpy()
                    y_out=pred[0,:,:]
                    y_true=batch_y[0,:,:]
                    data_set=test_data
                    y_out=data_set.inverse_transform_data(y_out)
                    y_true=data_set.inverse_transform_data(y_true)
                    X_pred=y_out #np.concatenate((X,y_out),axis=0) # Quelle axis?
                    X_true=y_true #np.concatenate((X,y_true),axis=0)
                    #* plot_video_skeletons demande :  (nb_joints,3,nb_frames) et pour l'instant on a ( nb_frames,nb_joints,3) 
                    
                    X_pred=X_pred.transpose(1,2,0)
                    X_true=X_true.transpose(1,2,0)
                    #* On va plot les résultats
                    plot_video_skeletons(mat_skeletons=[X_true,X_pred],save_name="train:"+self.args.model_id+str(i),path_folder_save=os.path.join(folder_path))
                    filename=str(test_data.liste_path["filename"].iloc[i]) # ???
                    plot_skeleton(path_skeleton=os.path.join(self.args.root_path,"raw/",filename+".skeleton"),save_name="train_"+str(i)+filename,path_folder_save=folder_path)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        df=test_data.liste_path
        df["mae"]=mae
        df["mse"]=mse
        df["rmse"]=rmse
        df["mape"]=mape
        df["mspe"]=mspe
        df.to_csv(os.path.join(folder_path,"results_df_train.csv"))
        f = open("result_long_term_forecast_train.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics_train.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred_train.npy', preds)
        np.save(folder_path + 'true_train.npy', trues)

        return
