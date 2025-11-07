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
from GeneralResNetTraining import GeneralResNetTraining


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
        self.img_size = 224


    def model_training(self, f, train_name='train', pos_embedding=False):
        # Load and preprocess data
        ROCF_dataset = LoadROCFDataset(img_size=self.img_size)

        general_resnet_training = GeneralResNetTraining(
            f=f, img_size=self.img_size, augmentation=self.augmentations, pos_embedding=pos_embedding
        )
        transform = general_resnet_training.get_swin_transformer_transforms()
        val_test_transform = general_resnet_training.get_swin_transformer_transforms(default=True)

        train_loader, val_loader, test_loader = general_resnet_training.initialize_datasets(
            rocf_dataset=ROCF_dataset, transform=transform, val_test_transform=val_test_transform
        )

        # Initialize the Swin Transformer model, loss function, and optimizer
        num_classes = 4  # Assuming you have 4 classes
        model = SwinTransformerClassifier(num_classes).to(self.device)


        # Ensure all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0025)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        # Training loop
        num_epochs = 12
        general_resnet_training.training_loop(
            train_name=train_name, num_epochs=num_epochs,
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
            model=model, rocf_dataset=ROCF_dataset, loss_fn=loss_fn, model_type='swin_transformer'
        )

        # Load best trained model weights
        model_path = f'{train_name}_model.pth'
        self.load_model(model, model_path)

        # Evaluation of best val model
        print("BEST VAL MODEL Accuracy on testing set: ")
        general_resnet_training.test(model=model, dataloader=test_loader, loss_fn=loss_fn, prefix='BEST VAL MODEL Test')

        print(f"\n\nEvaluating model on test set with test-time augmentation:")
        general_resnet_training.test_with_tta_metrics(
            model=model, rocf_dataset=ROCF_dataset, loss_fn=loss_fn, model_type='swin_transformer'
        )

        # torch.save(model.state_dict(), f'{train_name}_model.pth')

    def model_testing(self, f, model_path, pos_embedding=True):
        # Load and preprocess data
        ROCF_dataset = LoadROCFDataset(img_size=self.img_size)

        general_resnet_training = GeneralResNetTraining(
            f=f, img_size=self.img_size, augmentation=self.augmentations, pos_embedding=pos_embedding
        )
        val_test_transform = general_resnet_training.get_swin_transformer_transforms(default=True)

        _, _, test_loader = general_resnet_training.initialize_datasets(
            rocf_dataset=ROCF_dataset, transform=val_test_transform, val_test_transform=val_test_transform
        )

        # Initialize the Swin Transformer model, loss function, and optimizer
        num_classes = 4  # Assuming you have 4 classes
        model = SwinTransformerClassifier(num_classes).to(self.device)

        loss_fn = nn.CrossEntropyLoss()

        # Load best trained model weights
        self.load_model(model, model_path)

        print("Accuracy on testing set: ")
        general_resnet_training.test(model=model, dataloader=test_loader, loss_fn=loss_fn)

        print(f"\n\nEvaluating model on test set with test-time augmentation:")
        general_resnet_training.test_with_tta_metrics(
            model=model, rocf_dataset=ROCF_dataset, loss_fn=loss_fn, model_type='swin_transformer'
        )

    def load_model(self, model, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()

    def visualize_with_gradcam(self):
        models = [
            r'C:\Users\lucin\OneDrive\Desktop\diplomovka\thesis_code\transformer_new_results\swin_transformer_embedd_none_model.pth',
        ]
        file_names = [
            'swin_embedd_none_model'
        ]

        for i in range(len(models)):
            model_name = models[i]
            file_name = file_names[i]

            num_classes = 4
            model = SwinTransformerClassifier(num_classes).to(self.device)

            model.load_state_dict(torch.load(model_name, map_location=self.device))
            model.eval()

            LoadROCFDataset(transform=GeneralResNetTraining(
                f=None, img_size=self.img_size, augmentation=self.augmentations, pos_embedding=False
            ).get_swin_transformer_transforms(default=True)
                            ).visualize_gradcam(
                model, model_name=file_name, device=self.device, model_type='swin_transformer'
            )

# Main training loop
transformations = ['none', 'color', 'translate', 'crop', 'rotate', 'combo']

train = True
analyze_results = False
test = False

# VISUALIZE WITH Grad-Cam
# trainer = TrainSwinTransformer('none')
# trainer.visualize_with_gradcam()
# train = False

# ONLY TESTING
# train = False
# test = True

# analyze results
train = False
test = False
analyze_results = True

if train:
    for pos_embedding in [False, True]:
        for transformation in transformations:
            trainer = TrainSwinTransformer(transformation)
            emb = 'embedd' if pos_embedding else 'notEmb'
            train_name = f'swin_transformer_{emb}_{transformation}'
            print("\n----------------------\n", train_name)

            f = f'{train_name}.txt'
            with open(f, 'w') as file:
                file.write(f"{f}\n")
            trainer.model_training(f, train_name=train_name, pos_embedding=pos_embedding)

if test:
    model_dir = "C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/transformer_new_results/"

    for pos_embedding in [False, True]:
        for transformation in transformations:
            trainer = TrainSwinTransformer(transformation)
            emb = 'embedd' if pos_embedding else 'notEmb'
            test_name = f'swin_transformer_{emb}_{transformation}'
            print("\n-------------------------------------\n")
            print(test_name)
            model_path = model_dir + test_name + '_model.pth'
            print(model_path)

            f=f'{test_name}.txt'
            with open(f, 'w') as file:
                file.write(f"{f}\n")
            # Call the model training with the current global_pooling and transformation settings
            trainer.model_testing(f=f, pos_embedding=pos_embedding, model_path=model_path)


if analyze_results:
    model_dir = "C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/transformer_new_results/not_embedd/"
    with open(f'swin_transformer_test_results_analysis.txt', 'w') as f:
        general_resnet_training = GeneralResNetTraining(
            f=f, pos_embedding=False
        )
        general_resnet_training.analyze_model_logs_with_tta(log_dir=model_dir)
