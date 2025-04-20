import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from utils import plot_CE, plot_RE
from matplotlib import pyplot as plt
import numpy as np

from ROCFDataset_for_CNN import LoadROCFDataset, ROCFDataset


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes, global_pooling=True):
        super(ResNet18Classifier, self).__init__()
        self.global_pooling = global_pooling

        # Load ResNet18 pre-trained model
        self.model = models.resnet18(pretrained=True)

        # Modify the first conv layer to accept 1 input channel (for grayscale images)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # # Replace the fully connected layer to match the number of classes
        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, num_classes)

        # Replace the original average pooling and fully connected layer
        num_features = self.model.fc.in_features
        if self.global_pooling:
            #self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Use AdaptiveAvgPool2d
            self.model.fc = nn.Linear(num_features, num_classes)
        else:
            #self.model.avgpool = nn.Identity()  # No pooling
            #self.model.fc = nn.Linear(num_features * 16 * 16, num_classes)

            self.model.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # Use AvgPool2d
            self.model.fc = nn.Linear(num_features * 8 * 8, num_classes)

    def forward(self, x):
        return self.model(x)


class TrainResNet18():
    def __init__(self, augmentations="none"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.augmentations = augmentations

    def get_transforms(self):
        if self.augmentations == "crop":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.98, 1.0)  # Crop between 90% and 100% of the original size
                ),
                transforms.ToTensor()
            ])
        elif self.augmentations == "translate":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05)),
                transforms.ToTensor()
            ])
        elif self.augmentations == "color":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor()
            ])

    def plot_first_10_images(self, dataloader):
        plt.figure(figsize=(15, 6))
        for i, (images, _) in enumerate(dataloader):
            if i >= 10:
                break
            image = images[0].squeeze(0).numpy()
            image = image
            plt.subplot(2, 5, i + 1)
            plt.imshow(image, cmap="gray", origin="upper")
            plt.axis('off')
        plt.show()

    def model_training(self, f, global_pooling=True, train_name='train'):
        # Load your dataset
        X, y = [], []
        self.img_size = 500
        ROCF_dataset = LoadROCFDataset(img_size=self.img_size)

        for i in range(len(ROCF_dataset)):
            img_name = ROCF_dataset.get_image_path_all_files(i)
            name, order, score = ROCF_dataset.extract_from_name(img_name)
            X.append(img_name)
            y.append(score)

        print(len(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        print(len(X_train), len(X_val), len(X_test))
        transform = self.get_transforms()

        train_dataset = ROCFDataset(X_train, y_train, transform=transform)
        val_dataset = ROCFDataset(X_val, y_val, transform=transform)
        test_dataset = ROCFDataset(X_test, y_test, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Plot the first 10 images in the training loader
        # self.plot_first_10_images(train_loader)
        # return

        lr = 0.0001
        loss_fn, model, optimizer, scheduler = self.initialize_model(global_pooling, lr)

        # Training loop
        num_epochs = 8
        train_loss_epochs = []
        val_loss_epochs = []
        val_accuracy_epochs = []
        for epoch in range(num_epochs):
            loss = self.train(f, train_loader, model, loss_fn, optimizer)
            loss = loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
            f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}\n")

            val_accuracy, val_loss = self.test(f, model, val_loader, loss_fn, prefix='Val')

            val_loss_epochs.append(val_loss)
            val_accuracy_epochs.append(val_accuracy)
            train_loss_epochs.append(loss)

            scheduler.step()
            torch.save(model.state_dict(), f'{train_name}_model.pth')

        # plot_CE(val_accuracy_epochs, name_file='adaptive_thresh_resnet18_not_pretrained_validation_accuracy', show=False)
        # plot_RE(val_loss_epochs, name_file='adaptive_thresh_resnet18_not_pretrained_validation_loss', show=False)

        plt.plot(train_loss_epochs, c='r', label='epoch train losses')
        plt.plot(val_loss_epochs, c='b', label='epoch val losses')
        plt.title(f'{train_name} loss')
        plt.legend()
        plt.savefig(f'{train_name}_loss.png')
        plt.show()

        plt.plot(val_accuracy_epochs, c='g', label='epoch val accuracy')
        plt.title(f'{train_name} val accuracy')
        plt.legend()
        plt.savefig(f'{train_name}_val_accuracy.png')
        plt.show()

        # Evaluation
        print("Accuracy on testing set: ")
        self.test(f, model, test_loader, loss_fn)

        # Save the trained model
        torch.save(model.state_dict(), f'{train_name}_model.pth')

    def initialize_model(self, global_pooling, lr):
        # Initialize the ResNet18 model, loss function, and optimizer
        num_classes = 4  # Assuming you have 4 classes
        model = ResNet18Classifier(num_classes, global_pooling=global_pooling).to(self.device)
        loss_fn = nn.CrossEntropyLoss()  # Assuming classification task
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0025)  # L2 regularization
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)  # Learning rate scheduler
        return loss_fn, model, optimizer, scheduler

    def train(self, f, dataloader, model, loss_fn, optimizer):
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
                f.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")

        return loss

    def test(self, f, model, dataloader, loss_fn, prefix='Test'):
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
        print(f"{prefix} Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        f.write(f"{prefix} Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\n")
        return correct, test_loss


    def visualize_with_heatmap(self):
        models = ['C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet18_4outputs/pretrained/global_pooling_lr_0.001/resnet18_pretrained_globpool_color_model.pth',
                  'C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet18_4outputs/pretrained/not_global_pooling_lr_0.001/resnet18_pretrained_notglobpool_translate_model.pth',
                  'C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet18_4outputs/pretrained/global_pooling_lr_0.0001/resnet18_pretrained_lr0.0001_globpool_none_model.pth',
                  'C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet18_4outputs/pretrained/not_global_pooling_lr_0.0001/resnet18_pretrained_lr0.0001_notglobpool_none_model.pth']
        file_names = ['r18_GB_color_model',
                  'r18_notGB_translate_model',
                  'r18_lr0.0001_GB_none_model',
                  'r18_lr0.0001_notGB_none_model']

        for i in range(len(models)):
            model_name = models[i]
            file_name = file_names[i]

            global_pooling = True if '_globpool_' in model_name else False
            lr = 0.0001 if '0.0001' in model_name else 0.001
            _, model, _, _ = self.initialize_model(global_pooling, lr)
            self.augmentations = 'none'

            model.load_state_dict(torch.load(model_name))

            LoadROCFDataset(transform=self.get_transforms()).visualize_heatmaps(model, model_name=file_name)



# trainer = TrainResNet18('color')
# trainer.model_training(global_pooling=False)

trainer = TrainResNet18('none')
trainer.visualize_with_heatmap()

train = False
if train:
    transformations = ['none', 'color', 'translate', 'crop']
    global_pooling_values = [False, True]

    for global_pooling in global_pooling_values:
        for transformation in transformations:
            trainer = TrainResNet18(transformation)

            train_name = f'resnet18_pretrained_lr0.0001_{"globpool" if global_pooling else "notglobpool"}_{transformation}'
            print(train_name)

            with open(f'{train_name}.txt', 'w') as f:
                # Call the model training with the current global_pooling and transformation settings
                trainer.model_training(f, global_pooling=global_pooling, train_name=train_name)
