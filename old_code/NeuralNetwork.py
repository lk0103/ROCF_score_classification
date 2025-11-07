import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from ROCFDataset_for_NN import LoadROCFDataset, ROCFDataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = lambda x: torch.flatten(start_dim=1, end_dim=-1, input=x)   # FIXME
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),  # Add batch normalization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, LoadROCFDataset().num_score_classes),
            nn.LogSoftmax()
        )

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

class TrainNeuralNetwork():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def model_training(self):
        # Load your dataset
        X, y = [], []
        ROCF_dataset = LoadROCFDataset()

        for i in range(len(ROCF_dataset)):
            img_name = ROCF_dataset.get_image_path_control_group(i)
            name, order, score = ROCF_dataset.extract_from_name(img_name)
            X.append(img_name)
            y.append(score)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        train_dataset = ROCFDataset(X_train, y_train)
        test_dataset = ROCFDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize the model, loss function, and optimizer
        input_size = len(train_dataset[0][0].flatten())
        l = train_dataset[0][0]
        model = NeuralNetwork(input_size).to(self.device)
        model.apply(model.weights_init)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            loss = self.train(train_loader, model, loss_fn, optimizer)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            print("accuracy on training set: ")
            self.test(model, train_loader, loss_fn)
            scheduler.step()


        # Evaluation
        self.test(model, test_loader, loss_fn)

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)


            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)  # does not require one-hot encoding of y!

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return loss

    def test(self, model, dataloader, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



TrainNeuralNetwork().model_training()