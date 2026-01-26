import torch
import torch.nn as nn
import torch.optim as optim

from GeneralResNetTraining import GeneralResNetTraining

from ROCFDataset_for_CNN import LoadROCFDataset

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
            nn.Linear(768 * 2 * 2, 256),  # Fully connected layer 1 (assuming output feature map is 1x1)
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
        self.img_size = 500
        self.augmentations = 'none'
        self.preprocessing = 'grey'
        self.num_epochs = 25

    def model_training(self):
        train_name = 'CNN_ivanyi'
        f = f'{train_name}.txt'
        # Load your dataset
        ROCF_dataset = LoadROCFDataset(img_size=self.img_size, preprocessing=self.preprocessing)

        general_resnet_training = GeneralResNetTraining(
            f=f, img_size=self.img_size, augmentation=self.augmentations, pos_embedding=False,
            preprocessing=self.preprocessing
        )
        transform = general_resnet_training.get_resnet_transforms()
        val_test_transform = general_resnet_training.get_resnet_transforms(default=True)

        train_loader, val_loader, test_loader = general_resnet_training.initialize_datasets(
            rocf_dataset=ROCF_dataset, transform=transform, val_test_transform=val_test_transform
        )

        # Initialize the model, loss function, and optimizer
        input_channels = 1
        model = CNN(input_channels).to(self.device)
        model.apply(model.weights_init)
        loss_fn = nn.CrossEntropyLoss()  # Assuming classification task
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0025)  # L2 regularization
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)  # Learning rate scheduler

        # Training loop
        general_resnet_training.training_loop(
            train_name=train_name, num_epochs=self.num_epochs,
            train_loader=train_loader, val_loader=val_loader,
            model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler
        )

        # Load last trained model weights
        model_path = f'{train_name}_last_model.pth'
        self.load_model(model, model_path)

        # Evaluation
        print("LAST MODEL Accuracy on testing set: ")
        general_resnet_training.test(model=model, dataloader=test_loader, loss_fn=loss_fn, prefix='LAST MODEL Test')

        print(f"\n\nEvaluating model on test set with test-time augmentation:")
        general_resnet_training.test_with_tta_metrics(
            model=model, rocf_dataset=ROCF_dataset, loss_fn=loss_fn, model_type='resnet18'
        )

        # Load best trained model weights
        model_path = f'{train_name}_model.pth'
        self.load_model(model, model_path)

        # Evaluation of best val model
        print("BEST VAL MODEL Accuracy on testing set: ")
        general_resnet_training.test(model=model, dataloader=test_loader, loss_fn=loss_fn, prefix='BEST VAL MODEL Test')

        print(f"\n\nEvaluating model on test set with test-time augmentation:")
        general_resnet_training.test_with_tta_metrics(
            model=model, rocf_dataset=ROCF_dataset, loss_fn=loss_fn, model_type='resnet18'
        )

    def load_model(self, model, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()



# Example usage:
trainer = TrainCNN()
trainer.model_training()