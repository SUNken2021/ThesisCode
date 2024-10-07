import torch
from torch.utils.data import TensorDataset, DataLoader


# 读取数据集
def load_data(batch_size, data_type, train_type, predict_length=1):
    if train_type == "long":
        train, test, valid = torch.load(f"./data/{data_type}_data_long.pt")
        train_dataset = TensorDataset(train)
        test_dataset = TensorDataset(test)
        valid_dataset = TensorDataset(valid)
    else:
        train_given, train_predict, test_given, \
            test_predict, valid_given, valid_predict = torch.load(
                    f"./data/{data_type}_data_{predict_length}.pt"
                )
        train_dataset = TensorDataset(train_given, train_predict)
        test_dataset = TensorDataset(test_given, test_predict)
        valid_dataset = TensorDataset(valid_given, valid_predict)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    return train_loader, test_loader, valid_loader
