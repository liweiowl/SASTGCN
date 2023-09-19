import numpy as np
import copy
import os
import torch
from configuration import cf
from data_preprocess import load_dataset, slide_windows, ReadDataset, MinMaxscaler, Standardscaler, cal_adjacent_matrix
from model.sastgcn import AutoEncoder, MultiVariateLSTM, AutoEncoder_LSTM, SASTGCN
from metrics import evaluate

device = cf.device

########## get data #########
print("begin to load data using configs of cf defined in configuration")
data = load_dataset(cf)
# to simplify the calculation, using 1/10 to do experiments
data_slice = data[:3000]

# ready to do some normalization  # using z-sxore normalization method as other baselines
# data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
# data_max, data_min = data.max(), data.min()
if cf.scaler == "standard":
    scaler = Standardscaler(mean=data.mean(), std=data.std())
    data = scaler.transform(data_slice)
elif cf.scaler == "min-max":
    scaler = MinMaxscaler(min= data.min(), max = data.max())
    data = scaler.transform(data_slice)
else:
    scaler = None


# divide the train, validation and test set
train_split = int(len(data) * cf.trainset_rate)
validation_split = int(len(data) * (cf.trainset_rate + cf.validationset_rate))
train_data = data[:train_split]
validation_data = data[train_split:validation_split]
test_data = data[validation_split:]

print('begin to split the train,test dataset')
x_train, y_train = slide_windows(train_data, train_data, seq_len=cf.seq_len, pred_len=cf.pred_len)
x_validation, y_vavlidation = slide_windows(validation_data, validation_data, seq_len=cf.seq_len, pred_len=cf.pred_len)
x_test, y_test = slide_windows(test_data, test_data, seq_len=cf.seq_len, pred_len=cf.pred_len)

print(" x and y loaded...")

# creat pytorch dataset
train_dataset = ReadDataset(x=x_train, y=y_train)
test_dataset = ReadDataset(x=x_test, y=y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cf.batch_size, shuffle=True,
                                           num_workers=2, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cf.batch_size, shuffle=True,
                                          num_workers=2, drop_last=True)


## begin to train
def train_model(train_loader, cf):
    if cf.model_name == "autoencoder":
        model = AutoEncoder()
    elif cf.model_name == "lstm":
        model = MultiVariateLSTM()
    elif cf.model_name == "autoencoder_lstm":
        model = AutoEncoder_LSTM()
    elif cf.model_name == "sastgcn":
        model = SASTGCN()

    print(f" use a {cf.model_name} model to train")
    model = model.to(device)

    criterion = torch.nn.MSELoss().to(torch.float32).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)
    for epoch in range(cf.max_epoch):
        epoch_loss = 0
        print(f"start to train epoch {epoch}")
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # put the batch samples on device
            # print(f"input shape: {inputs.shape}, label shape:{labels.shape}")
            inputs = inputs.to(torch.float32).cuda(1)
            labels = labels.to(torch.float32).cuda(1)
            adj = cal_adjacent_matrix(inputs.cpu().numpy(), normalized_category="laplacian")
            adj = torch.from_numpy(adj).to(torch.float32).to(device)
            outputs, autooutput = model(inputs,adj)
            # print(f"outputs shape is {outputs.shape}, inputs shape is {inputs.shape}")
            loss1 = criterion(outputs, labels).cuda(1)
            loss2 = criterion(inputs, autooutput).cuda(1)
            # loss = criterion(outputs, inputs).cuda(1)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            # if i % 10 == 0:
            #     print(f"Epoch {epoch}, step {i}, Loss: {loss}")
            # print(f"Epoch {epoch}, step {i}, Loss: {loss/cf.batch_size}")
            epoch_loss += loss.item()
        # epoch_loss /= (i*cf.batch_size)
        print(f"in epoch {epoch}, the training loss is {epoch_loss / len(train_loader)}")
    print("________________________________________________________")
    torch.save(model, os.path.join(cf.model_save_path, f"model_{cf.model_name}.pkl"))
    print("trained")


def test_model(test_loader, cf):
    model = torch.load(os.path.join(cf.model_save_path, f"model_{cf.model_name}.pkl"))
    print(f"{cf.model_name} model loaded successfully" )
    ret_output = []
    y_label = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            print(i)
            inputs = inputs.to(device, dtype=torch.float)
            outputs, _ = model(inputs)
            ret_output += outputs.tolist()
            y_label += labels.tolist()
    print(f"y_label shape is : {np.array(y_label).shape}")
    y_label = np.array(y_label).reshape(len(y_label) * len(y_label[0]) * len(y_label[0][0]))
    ret_output = np.array(ret_output).reshape(len(ret_output) * len(ret_output[0]) * len(ret_output[0][0]))
    if scaler != None:
        y_label, ret_output = scaler.reverse_transform(y_label), scaler.reverse_transform(ret_output)
    rmse_res, mae_res, mape_res, r2_res, var_res, pcc_res = evaluate(y_label, ret_output)
    print(f"using normalization {cf.scaler}: RMSE:{rmse_res}; MAE:{mae_res};  MAPE:{mape_res}; R2 score: {r2_res}; VAR:{var_res};  PCC:{pcc_res}")

    print("test is over")


if __name__ == "__main__":
    
    train_model(train_loader, cf)
    test_model(test_loader, cf)

print("love world")
