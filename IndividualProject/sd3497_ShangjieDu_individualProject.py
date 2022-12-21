import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import scipy.io
import numpy as np
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class mydataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)
        self.transform = transform

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, idx):
        # convert data to a tensor
        x = self.X[idx]
        label = self.Label[idx]
        if self.transform:
            x = self.transform(x)

        return x, label


def prepare_data():
    # load data
    train_data = scipy.io.loadmat("dataset/SVHN/format2/train_32x32.mat")
    X_train = train_data["X"].transpose(3, 2, 0, 1)
    Y_train = train_data["y"].reshape(-1)
    Y_train = np.where(Y_train == 10, 0, Y_train)
    test_data = scipy.io.loadmat("dataset/SVHN/format2/test_32x32.mat")
    X_test = test_data["X"].transpose(3, 2, 0, 1)
    Y_test = test_data["y"].reshape(-1)
    Y_test = np.where(Y_test == 10, 0, Y_test)


    # preprae dataset
    transforms = {
        "train": torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    trainset = mydataset(X_train, Y_train, transforms["train"])
    testset = mydataset(X_test, Y_test, transforms["test"])

    # prepare dataloader
    batch_size = 128
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


class simpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)

        self.apply(_init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        output = self.flatten(x)
        logits = self.fc(output)

        return logits


class mediumCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(mediumCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)

        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(2048, num_classes)

        self.apply(_init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        output = self.flatten(self.avgpool(x))
        logits = self.fc(output)

        return logits


def _init_weights(m):
    torch.manual_seed(42)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_loss_acc(model, dataloader, criterion=nn.CrossEntropyLoss()):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        model.eval()
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            total += len(Y_batch)
            num_batches += 1
            logits = model(X_batch)
            y_pred = torch.argmax(logits, dim=1)
            correct += torch.sum(y_pred == Y_batch).cpu().numpy()
            loss = criterion(logits, Y_batch)
            total_loss += loss.item()
    acc = correct / total
    total_loss = total_loss / num_batches

    return total_loss, acc


def train(model : nn.Module,
          epochs : int,
          trainloader : DataLoader,
          testloader : DataLoader,
          save_path : str = None):
    # Prepare model
    model.to(device)

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print("Trained on {}, {} samples".format(device,
                                             len(trainloader.dataset.Label)))
    best_test_acc = 0
    for epoch in range(epochs):
        model.train()
        model.to(device)
        for X_batch, Y_batch in tqdm(trainloader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # forward
            logits = model(X_batch)
            loss = loss_func(logits, Y_batch)
            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # evaluate
        with torch.no_grad():
            model.eval()
            train_loss, train_acc = get_loss_acc(model, trainloader)
            test_loss, test_acc = get_loss_acc(model, testloader)

        print("Epoch{}/{} train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(
            epoch + 1, epochs,
            train_loss, test_loss,
            train_acc, test_acc
        ))

        # Save model weights if it's the best
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("Saved")


def evaluation(model=None, weight_path=None):
    # load weights
    model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    # evaluate
    trainloader, testloader = prepare_data()
    _, test_acc = get_loss_acc(model, testloader)
    print(f"Test Accuracy: {test_acc}")

    # count parameters
    count = sum(p.numel() for p in model.parameters())
    print(f"{count} parameters")


if __name__ == "__main__":
    model = simpleCNN()
    EPOCHS = 50
    trainloader, testloader = prepare_data()
    save_path = "./simpleCNN.pth"
    train(model, EPOCHS, trainloader, testloader, save_path)

    model = simpleCNN()
    weight_path = "./simpleCNN.pth"
    evaluation(model ,weight_path)
