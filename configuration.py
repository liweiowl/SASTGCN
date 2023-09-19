from easydict import EasyDict as edict
import os
import torch

def myconfig():
    cf = edict()

    ##################### dataset #######################
    cf.data_path = "./dataset/"
    cf.dataset_name = "metr-la.h5"
    # cf.dataset_name = "pems-bay.h5"
    #################### preprocess ####################
    cf.seq_len = 30  # number of timestamps used to predict xxx    []
    cf.pred_len = 6  # number of timestamps to be predict
    cf.trainset_rate = 0.7
    cf.validationset_rate = 0.1
    cf.testset_rate = 0.2
    cf.scaler = "standard"  # min-max scaler, standard scaler
    # cf.scaler = "min-max"
    # cf.scaler = None
    cf.normalize_all = True

    #################### model hyperparameters detail #########################
    cf.num_nodes = 207
    ############ autoencoder #############
    cf.return_hidden = True
    cf.a_hidden_dim1 = 600
    cf.a_hidden_dim2 = 60
    cf.a_hidden_repre_dim = cf.a_hidden_dim2

    ############ gcn part #############
    cf.gcn_hidden_dim1 = 207    # to keep the second gcn layer correct ,the dim1 ought to be 207
    cf.gcn_hidden_dim2 = 100

    ############ lstm part ############
    cf.lstm_num_layers = 2
    cf.lstm_hidden_size = 200
    cf.lstm_output_size = 100
    
    ############ whole last part ############



    ################### training parameters ####################
    cf.model_name = "sastgcn"    #"lstm" "gcn"  "2layer_lstm","t-gcn"
    # cf.model_name = "autoencoder_lstm"    #"lstm" "gcn"  "2layer_lstm","t-gcn"
    # cf.model_name = "lstm"    #"lstm" "gcn"  "2layer_lstm","t-gcn"
    # cf.model_name = "autoencoder"    #"lstm" "gcn"  "2layer_lstm","t-gcn"
    cf.max_epoch = 200
    cf.batch_size = 32
    cf.earlystop = 0.1
    cf.learning_rate = 0.01
    cf.weight_decay = 1e-5
    cf.optimizer = "Adam"
    cf.device = "cuda:1" if torch.cuda.is_available() else "cpu"


    ################## result saving parameters ################
    cf.model_save_path = "./results/saved_models/"
    cf.result_save_path = "./results/result/"



    return cf


cf = myconfig()

if __name__ == "__main__":


    print('love world')