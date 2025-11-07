import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import plot_CE, plot_RE

from ROCFDataset_for_CNN import LoadROCFDataset, ROCFDataset

class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            # CNN layers based on the MATLAB definition
            nn.Conv2d(input_channels, 2, kernel_size=5, padding=2),  # Conv layer 1 (2 filters)
            nn.BatchNorm2d(2),  # Batch nor malization
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(2, 4, kernel_size=5, padding=2),  # Conv layer 2 (4 filters)
            nn.BatchNorm2d(4),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(4, 8, kernel_size=5, padding=2),  # Conv layer 3 (8 filters)
            nn.BatchNorm2d(8),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(8, 16, kernel_size=5, padding=2),  # Conv layer 4 (16 filters)
            nn.BatchNorm2d(16),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max pooling layer

            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # Conv layer 5 (32 filters)
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max pooling layer

            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # Conv layer 6 (64 filters)
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max pooling layer

            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # Conv layer 7 (128 filters)
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max pooling layer

            nn.Conv2d(128, 256, kernel_size=5, padding=2),  # Conv layer 8 (256 filters)
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max pooling layer

            nn.Conv2d(256, 512, kernel_size=5, padding=2),  # Conv layer 9 (512 filters)
            nn.BatchNorm2d(512),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max pooling layer

            nn.Conv2d(512, 768, kernel_size=5, padding=2),  # Conv layer 10 (768 filters)
            nn.BatchNorm2d(768),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max pooling layer

            nn.Flatten(),  # Flatten for fully connected layers
            nn.Linear(768 * 1 * 1, 256),  # Fully connected layer 1 (assuming output feature map is 1x1)
            nn.ReLU(),  # ReLU activation
            nn.Linear(256, 128),  # Fully connected layer 2
            nn.ReLU(),  # ReLU activation
            nn.Linear(128, 64),  # Fully connected layer 3
            nn.ReLU(),  # ReLU activation
            nn.Linear(64, 4),  # Fully connected layer 4 (4 classes for classification)
            nn.LogSoftmax(dim=1)  # LogSoftmax for classification
        )

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        logits = self.layers(x)
        return logits


class TrainCNN():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def model_training(self):
        # Load your dataset
        X, y = [], []
        ROCF_dataset = LoadROCFDataset()

        for i in range(len(ROCF_dataset)):
            img_name = ROCF_dataset.get_image_path_all_files(i)
            name, order, score = ROCF_dataset.extract_from_name(img_name)
            X.append(img_name)
            y.append(score)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        train_dataset = ROCFDataset(X_train, y_train)
        val_dataset = ROCFDataset(X_val, y_val)
        test_dataset = ROCFDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Initialize the model, loss function, and optimizer
        input_channels = 1
        model = CNN(input_channels).to(self.device)
        model.apply(model.weights_init)
        loss_fn = nn.CrossEntropyLoss()  # Assuming classification task
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0025)  # L2 regularization
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)  # Learning rate scheduler

        # Training loop
        num_epochs = 25
        val_loss_epochs = []
        val_accuracy_epochs = []
        for epoch in range(num_epochs):
            loss = self.train(train_loader, model, loss_fn, optimizer)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
            val_accuracy, val_loss = self.test(model, val_loader, loss_fn)
            val_loss_epochs.append(val_loss)
            val_accuracy_epochs.append(val_accuracy)
            scheduler.step()
        plot_CE(val_accuracy_epochs, name_file='my_preproc_ivanyi_CNN_validation_accuracy_training', show=False)
        plot_RE(val_loss_epochs, name_file='my_preproc_CNN_validation_loss_training', show=False)

        # Evaluation
        print("Accuracy on testing set: ")
        self.test(model, test_loader, loss_fn)

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
        return correct, test_loss


# Example usage:
trainer = TrainCNN()
trainer.model_training()