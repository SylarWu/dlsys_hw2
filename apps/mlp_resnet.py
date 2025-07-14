import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(nn.Sequential(
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, dim),
            norm(dim)
        )),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
        dim,
        hidden_dim=100,
        num_blocks=3,
        num_classes=10,
        norm=nn.BatchNorm1d,
        drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    loss_func = nn.SoftmaxLoss()
    total_loss = 0
    correct_count = 0
    for x_batch, y_batch in dataloader:
        x_batch = ndl.Tensor(x_batch.reshape((x_batch.shape[0], -1)), requires_grad=False)
        y_batch = ndl.Tensor(y_batch, requires_grad=False)
        pred_batch = model(x_batch)
        loss = loss_func(pred_batch, y_batch)
        if opt:
            loss.backward()
            opt.step()
        total_loss += loss.numpy() * x_batch.shape[0]
        correct_count += np.sum(np.argmax(pred_batch.numpy(), axis=1) == y_batch.numpy())
    return 1 - (correct_count / len(dataloader.dataset)), total_loss / len(dataloader.dataset)
    ### END YOUR SOLUTION


def train_mnist(
        batch_size=100,
        epochs=10,
        optimizer=ndl.optim.Adam,
        lr=0.001,
        weight_decay=0.001,
        hidden_dim=100,
        data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 1. init dataset
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
    )
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    )
    # 2. init dataloader
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 3. init model -> loss -> optimizer
    model = MLPResNet(28 * 28, hidden_dim=hidden_dim)
    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. begin train
    model.train()
    train_err_rate, train_avg_loss = 0, 0
    for epoch_index in range(epochs):
        train_err_rate, train_avg_loss = epoch(train_dataloader, model, opt)
    # 5. begin test
    model.eval()
    test_err_rate, test_avg_loss = epoch(test_dataloader, model, None)
    return train_err_rate, train_avg_loss, test_err_rate, test_avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
