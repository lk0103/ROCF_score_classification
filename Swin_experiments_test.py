import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

from ROCFDataset_for_CNN import LoadROCFDataset, ROCFDataset


class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerClassifier, self).__init__()

        # Load Swin Transformer Tiny model from torchvision with pretrained weights
        weights = Swin_T_Weights.DEFAULT
        self.model = swin_t(weights=weights)

        # Replace the classification head to match the number of classes
        num_features = self.model.head.in_features
        self.model.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class TrainSwinTransformer():
    def __init__(self, augmentations="none"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.augmentations = augmentations

    def get_transforms(self):
        def to_rgb(image):
            if isinstance(image, torch.Tensor):
                image = F.to_pil_image(image)
            return F.to_tensor(image.convert("RGB"))

        def scale_to_minus_one_to_one(image):
            return (image * 2.0) - 1.0  # Scale to [-1, 1]

        if self.augmentations == "crop":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.98, 1.0)),
                transforms.Lambda(to_rgb),  # Ensure 3 channels (RGB)
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])
        elif self.augmentations == "translate":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05)),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])
        elif self.augmentations == "color":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])

    def model_training(self, f, train_name='train', path=None):
        # Load and preprocess data
        X, y = [], []
        self.img_size = 224  # Swin Transformer expects 224x224 images
        ROCF_dataset = LoadROCFDataset(img_size=self.img_size)

        for i in range(len(ROCF_dataset)):
            img_name = ROCF_dataset.get_image_path_all_files(i)
            name, order, score = ROCF_dataset.extract_from_name(img_name)
            X.append(img_name)
            y.append(score)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        transform = self.get_transforms()

        train_dataset = ROCFDataset(X_train, y_train, transform=transform)
        val_dataset = ROCFDataset(X_val, y_val, transform=transform)
        test_dataset = ROCFDataset(X_test, y_test, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Initialize the Swin Transformer model, loss function, and optimizer
        num_classes = 4  # Assuming you have 4 classes
        model = SwinTransformerClassifier(num_classes).to(self.device)

        if path is not None:
            model.load_state_dict(torch.load(path))

        # Ensure all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0025)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        train_loss_epochs = [1.2897, 0.4853, 0.7873, 0.7525, 0.3835, 0.3317, 0.3699, 0.5517, 0.2549, 0.2872]
        val_loss_epochs = [0.9925, 1.0358, 0.6234, 0.4776, 0.3372, 0.2853, 0.2262, 0.1561, 0.1443, 0.1193]
        val_accuracy_epochs = [53.19, 47.87, 76.60, 87.23, 82.98, 93.62, 94.68, 96.81, 95.74, 95.74]


        # Plot training and validation metrics
        plt.plot(train_loss_epochs, c='r', label='Train Loss')
        plt.plot(val_loss_epochs, c='b', label='Validation Loss')
        plt.title(f'{train_name} Loss')
        plt.legend()
        plt.savefig(f'{train_name}_loss.png')
        plt.show()

        plt.plot(val_accuracy_epochs, c='g', label='Validation Accuracy')
        plt.title(f'{train_name} Validation Accuracy')
        plt.legend()
        plt.savefig(f'{train_name}_val_accuracy.png')
        plt.show()

        print("Accuracy on testing set: ")
        self.test(f, model, test_loader, loss_fn)

        #torch.save(model.state_dict(), f'{train_name}_model.pth')

    def test(self, f, model, loader, loss_fn, prefix='Test'):
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                if random.random() < 0.5:
                    print(predicted.shape, predicted, 'ground truth', labels.argmax(dim=1))
                total_correct += (predicted == labels.argmax(dim=1)).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100 * total_correct / total_samples
        print(f"{prefix} Loss: {avg_loss:.4f}, {prefix} Accuracy: {accuracy:.2f}% \n\n")
        #f.write(f"{prefix} Loss: {avg_loss:.4f}, {prefix} Accuracy: {accuracy:.2f}%\n\n")
        return accuracy, avg_loss


# Main training loop
#transformations = ['none', 'color', 'translate', 'crop']
transformations = ['color', 'translate', 'crop']
#urobit test na modeli none!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# for transformation in transformations:
#     trainer = TrainSwinTransformer(transformation)
#     train_name = f'swin_transformer_{transformation}'
#     print(train_name)
#
#     with open(f'{train_name}.txt', 'w') as f:
#         trainer.model_training(f, train_name=train_name)

trainer = TrainSwinTransformer('none')
train_name = f'swin_transformer_none'
trainer.model_training(None, train_name=train_name, path='swin_transformer_none_model.pth')
