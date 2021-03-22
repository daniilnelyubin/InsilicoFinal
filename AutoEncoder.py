import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from NeuralNetworks import get_data_for_nn, get_train_test_loaders


class AutoEncoder(nn.Module):
    def __init__(self, in_chanel, hidden_layer_size=500):
        super(AutoEncoder, self).__init__()
        self.batch_norm_1 = nn.BatchNorm1d(in_chanel)
        self.dense_1 = nn.Linear(in_chanel, hidden_layer_size)
        self.dense_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dense_3 = nn.Linear(hidden_layer_size, in_chanel)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = self.batch_norm_1(x)
        x = F.relu(self.dense_1(x))
        # x = self.dropout(F.relu(self.dense_2(x)))
        x = F.relu(self.dense_3(x))
        return x

    def encode(self, x):
        # x = self.batch_norm_1(x)
        x = F.relu(self.dense_1(x))
        # x = F.relu(self.dense_2(x))

        return x

def train_autoencoder(train_data=None, test_data=None, lr=0.1e-3, batch_size=500, epochs=300, test=False):
    result_dataset, y, result_dataset_test, test_idx = get_data_for_nn()

    has_cuda = True

    model = AutoEncoder(result_dataset.shape[1], 1024).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # file_name = "model_min_loss.pkl"
    # if os.path.exists(file_name):
    #     model.load_state_dict(torch.load(file_name))
    #     print("-------------------Get " + file_name + "-------------------")

    model.double()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", threshold=1e-7,factor=0.9, patience=50, verbose=True)

    train_loader, test_loader = get_train_test_loaders(result_dataset, y, batch_size)

    prev_test_loss = 10000

    for epoch in range(epochs):
        loss_train_ = 0.0
        model.train()
        for X, _ in train_loader:
            X = X.to(DEVICE)
            result = model(X)
            loss_train = F.mse_loss(result, X)
            loss_train_ += loss_train.item()
            loss_train.backward()
        optimizer.step()
        scheduler.step(loss_train_)
        model.eval()

        with torch.no_grad():
            test_loss = 0
            for idx, (X_test, _) in enumerate(test_loader):
                if has_cuda:
                    X_test = X_test.to(DEVICE)
                result_test = model(X_test)
                test_loss += F.mse_loss(result_test, X_test).item()
            print(f'Epoch:{epoch + 1}, Train Loss:{loss_train_:.6f}, Test Loss:{test_loss:.6f}.')

            if prev_test_loss > test_loss:
                prev_test_loss = test_loss
                torch.save(model.state_dict(), 'autoencoder/model_min_loss.pkl')
        if epoch % 100 == 0:
            torch.save(model.state_dict(), 'autoencoder/model_newest_' + str(epoch) + '.pkl')


if __name__ == '__main__':
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("\n--------------We use GPU!--------------\n")
    else:
        DEVICE = torch.device("cpu")

    train_autoencoder(lr=0.1e-4, batch_size=1095, epochs=6000, test=False)
