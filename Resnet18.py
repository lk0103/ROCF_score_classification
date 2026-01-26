import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models
from utils import plot_CE, plot_RE

from ROCFDataset_for_CNN import LoadROCFDataset, ROCFDataset
from GeneralResNetTraining import GeneralResNetTraining


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Classifier, self).__init__()

        # Load ResNet18 pre-trained model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first conv layer to accept 1 input channel (for grayscale images)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the fully connected layer to match the number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class TrainResNet18():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def model_training(self, f, train_name='train'):
        # Load your dataset
        ROCF_dataset = LoadROCFDataset(img_size=224)

        general_resnet_training = GeneralResNetTraining(f=f, img_size=500, augmentation='none', pos_embedding=False)
        transform = general_resnet_training.get_resnet_transforms(default=True)

        train_loader, val_loader, test_loader = general_resnet_training.initialize_datasets(
            rocf_dataset=ROCF_dataset, transform=transform, val_test_transform=transform
        )

        # Initialize the ResNet18 model, loss function, and optimizer
        num_classes = 4  # Assuming you have 4 classes
        model = ResNet18Classifier(num_classes).to(self.device)
        loss_fn = nn.CrossEntropyLoss()  # Assuming classification task
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0025)  # L2 regularization
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)  # Learning rate scheduler

        # Training loop
        num_epochs = 15
        general_resnet_training.training_loop(
            train_name=train_name, num_epochs=num_epochs,
            train_loader=train_loader, val_loader=val_loader,
            model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler
        )

        # Evaluation
        print("Accuracy on testing set: ")
        general_resnet_training.test(model=model, dataloader=test_loader, loss_fn=loss_fn)

        # Save the trained model
        # torch.save(model.state_dict(), 'adaptive_thresh_resnet18_not_pretrained_model.pth')

# Example usage:
trainer = TrainResNet18()
train_name = 'adaptive_thresh_resnet18_not_pretrained'
with open(f'{train_name}.txt', 'w') as f:
    trainer.model_training(f, train_name)