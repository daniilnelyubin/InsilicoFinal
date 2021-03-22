import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from joblib import dump, load
import os
import pickle
import AutoEncoder


class CNN(nn.Module):
    def __init__(self, in_chanel):
        super(CNN, self).__init__()
        self.in_chanel = in_chanel
        self.batch_norm_1 = nn.BatchNorm2d(1)
        self.conv_1 = nn.Conv2d(1, 8, 3)
        self.pooling_1 = nn.MaxPool2d(2)

        self.batch_norm_2 = nn.BatchNorm2d(8)
        self.conv_2 = nn.Conv2d(8, 16, 3)
        self.pooling_2 = nn.MaxPool2d(2)

        self.batch_norm_3 = nn.BatchNorm2d(16)
        self.conv_3 = nn.Conv2d(16, 32, 3)
        self.pooling_3 = nn.MaxPool2d(2)

        self.batch_norm_4 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 32, 32))

        x = self.batch_norm_1(x)
        x = self.conv_1(x)
        x = F.relu(self.pooling_1(x))

        x = self.batch_norm_2(x)
        x = self.conv_2(x)
        x = F.relu(self.pooling_2(x))

        x = self.batch_norm_3(x)
        x = self.conv_3(x)
        x = F.relu(self.pooling_3(x))
        x = self.batch_norm_4(x)

        x = torch.squeeze(x)
        # print(x.shape)
        x = self.dropout(self.flatten(x))
        x = self.sigmoid(self.linear(x))

        return x


class DNN(nn.Module):

    def __init__(self, in_chanel, hidden_layer_size):
        super(DNN, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(in_chanel)
        self.dropout1 = nn.Dropout(0.15)
        self.dense1 = nn.utils.weight_norm(nn.Linear(in_chanel, hidden_layer_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_layer_size)
        self.dropout3 = nn.Dropout(0.1)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_layer_size, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.sigmoid(self.dense3(x))

        return x


class Data_Provider(Dataset):
    def __init__(self, X, y):
        super(Data_Provider, self).__init__()
        self.X_ = X
        self.y_ = y.to_numpy()

    def __getitem__(self, index):
        return self.X_[index, :], self.y_[index].flatten()

    def __len__(self):
        return len(self.X_)


def adjust_learning_rate(optimizer, lr):
    lr = lr / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_imputer(percent_of_nan, n_features, path="notebooks/Imputer/"):
    for file in os.listdir(path):
        if str(percent_of_nan) in file and str(n_features) in file:
            impute_object = load(path + file)
            return impute_object
    return IterativeImputer(random_state=42, max_iter=10, n_nearest_features=n_features)


def droped_values_to_categorical(df, dropped_columns, drop_1095):
    df[dropped_columns].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df[dropped_columns] = np.where(df[dropped_columns] > 0, 1, 0)
    df = df.drop(drop_1095, axis=1)
    # sort_df_test_set = df.isna().sum().sort_values(ascending=False)
    # drop_1095 = sort_df_test_set[sort_df_test_set==df.shape[0]].index
    # df = df.drop(drop_1095,axis=1)
    return df


def drop_constant_columns(df, constant_columns):
    df.drop(constant_columns, axis=1, inplace=True)
    return df


def save_imputer(imputer, n_features, nans):
    dump(imputer, f'notebooks/Imputer/imputer_{n_features}_{nans}.joblib')


def save_normalizer(normalizer, file_name='nn_dumped_normalizer.pkl'):
    with open(file_name, 'wb') as fid:
        pickle.dump(normalizer, fid)


def load_normalizer(file_name='nn_dumped_normalizer.pkl'):
    normalizer = None
    for file in os.listdir():
        if file_name in file:
            with open(file_name, 'rb') as fid:
                normalizer = pickle.load(fid)
    return normalizer


def get_data_for_nn(train_path="./data/train.csv", test_path="./data/test.csv", percent_of_nan=0.9, n_features=100,
                    drop_train_nan=True):
    """Replace all NaN's and Return TrainDataFrame, TrainLabels, KaggleTestData, KaggleTestID"""

    df = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    y = df.y

    df.drop(['sample_id'], inplace=True, axis=1)
    df.drop(['y'], inplace=True, axis=1)

    test_idx = df_test['sample_id']
    df_test.drop(['sample_id'], inplace=True, axis=1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    sort_df = df.isna().sum().sort_values(ascending=False)

    columns_with_nan = sort_df[sort_df > df.shape[0] * percent_of_nan].index
    sort_df = df.isna().sum().sort_values(ascending=False)
    dropped_columns = sort_df[sort_df == df.shape[0]].index

    df_train_dropped = droped_values_to_categorical(df, columns_with_nan, dropped_columns)

    df_test_dropped = droped_values_to_categorical(df_test, columns_with_nan, dropped_columns)

    # constant_columns = df_train_dropped.columns[df_train_dropped.nunique()==1]
    # df_train_dropped = drop_constant_columns(df_train_dropped,constant_columns)
    # df_test_dropped = drop_constant_columns(df_test_dropped,constant_columns)

    impute_object = get_imputer(percent_of_nan, n_features)

    try:
        result_dataset = impute_object.transform(df_train_dropped)

        result_dataset_test = impute_object.transform(df_test_dropped)

    except:

        print("Fit the Imputer")
        result_dataset = impute_object.fit_transform(df_train_dropped)
        result_dataset_test = impute_object.transform(df_test_dropped)
        save_imputer(impute_object, n_features, percent_of_nan)

    print(f"DataSet has {result_dataset.shape[1]} features")

    normalizer = load_normalizer()
    if normalizer:
        result_dataset = normalizer.transform(result_dataset)
        result_dataset_test = normalizer.transform(result_dataset_test)
    else:
        normalizer = StandardScaler()
        result_dataset = normalizer.fit_transform(result_dataset)
        result_dataset_test = normalizer.transform(result_dataset_test)
        save_normalizer(normalizer)

    return result_dataset, y, result_dataset_test, test_idx


def kaggle_write(result, index, file_name="Answer.csv"):
    pd.DataFrame({"y": result}, index=index).to_csv(file_name, index=True, index_label="sample_id")


def get_train_test_loaders(X, y, batch_size, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dataset_train = Data_Provider(X_train, y_train)
    dataset_test = Data_Provider(X_test, y_test)

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        num_workers=4
    )
    return train_loader, test_loader


def train_nn(train_data=None, test_data=None, lr=0.1e-3, batch_size=500, epochs=300, cnn=True, path_to_model_save=""):
    result_dataset, y, _, _ = get_data_for_nn()

    has_cuda = True
    if cnn:
        print("We load CNN")
        model = CNN(1024).to(DEVICE)
    else:
        print("We load DNN")
        model = DNN(1024, 512).to(DEVICE)
    # model.load_state_dict(torch.load('model_newest_1000.pkl'))
    encoder = AutoEncoder.AutoEncoder(result_dataset.shape[1], 1024).to(DEVICE)
    encoder.load_state_dict(torch.load("autoencoder/model_min_loss.pkl"))
    encoder.eval()
    encoder.double()

    # file_name = "model_min_loss.pkl"
    # if os.path.exists(file_name):
    #     model.load_state_dict(torch.load(file_name))
    #     print("-------------------Get " + file_name + "-------------------")

    model.double()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, threshold=1e-7, mode="min", factor=0.9, patience=15, verbose=True)

    train_loader, test_loader = get_train_test_loaders(result_dataset, y, batch_size)

    loss_file_train = open('loss_log_train.txt', 'w+')
    loss_file_test = open('loss_log_test.txt', 'w+')
    prev_test_loss = 10000

    for epoch in range(epochs):
        loss_train_ = 0.0
        model.train()
        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            X = encoder.encode(X)

            result = model(X)

            loss_train = F.binary_cross_entropy(result, y)
            loss_train_ += loss_train.item()

            loss_train.backward()
        optimizer.step()
        scheduler.step(loss_train_)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            for idx, (X_test, y_test) in enumerate(test_loader):
                if has_cuda:
                    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
                X_test = encoder.encode(X_test)
                result_test = model(X_test)

                test_loss += F.binary_cross_entropy(result_test, y_test.double()).item()

            print(f'Epoch:{epoch + 1}, Train Loss:{loss_train_:.5f}, Test Loss:{test_loss:.5f}')
            loss_file_train.write('{},'.format(loss_train_))
            loss_file_test.write('{},'.format(test_loss))
            if prev_test_loss > test_loss:
                prev_test_loss = test_loss
                torch.save(model.state_dict(), f'{path_to_model_save}model_min_loss.pkl')

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f'model_newest_{epoch}.pkl')
    model.load_state_dict(torch.load(path_to_model_save + f'{path_to_model_save}model_min_loss.pkl'))
    return model


def write_for_kaggle(model, file_name="NN_Answer.csv"):
    print("---------Evaluating AUC ROC---------")

    result_dataset, y, result_dataset_test, test_idx = get_data_for_nn()

    encoder = AutoEncoder.AutoEncoder(result_dataset.shape[1], 1024).to(DEVICE)
    encoder.load_state_dict(torch.load("..autoencoder/model_min_loss.pkl"))
    encoder.eval()

    input_data = result_dataset_test
    input_data = encoder.encode(torch.Tensor(input_data).to(DEVICE))

    kaggle_result = model(input_data)
    kaggle_result = kaggle_result.cpu().detach().numpy().flatten()

    kaggle_write(kaggle_result, test_idx, file_name)


if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("\n--------------We use GPU!--------------\n")
    else:
        DEVICE = torch.device("cpu")

    # model = train_nn(lr=0.1e-4, batch_size=1000, epochs=1000,cnn=False,path_to_model_save='dnn/')
    # write_for_kaggle(model, file_name='DNN_Answer.csv')
    model = train_nn(lr=0.1e-4, batch_size=1000, epochs=1000, cnn=True, path_to_model_save='cnn/')
    write_for_kaggle(model, file_name='CNN_Answer.csv')

