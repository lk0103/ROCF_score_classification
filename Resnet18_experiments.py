import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from matplotlib import pyplot as plt

from ROCFDataset_for_CNN import LoadROCFDataset
from GeneralResNetTraining import GeneralResNetTraining


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes, global_pooling=True):
        super(ResNet18Classifier, self).__init__()
        print(num_classes)
        self.global_pooling = global_pooling

        # Load ResNet18 pre-trained model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first conv layer to accept 1 input channel (for grayscale images)
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # # Replace the fully connected layer to match the number of classes
        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, num_classes)

        # freeze parameters in convolutional layers
        # for param in self.model.parameters():
        #     param.requires_grad = False

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

        # unfreeze fully connected layers
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True


    def forward(self, x):
        return self.model(x)


class TrainResNet18():
    def __init__(self, augmentations="none", preprocessing='grey'):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.augmentations = augmentations
        self.preprocessing = preprocessing

        self.img_size = 500
        self.lr = 0.0001
        self.num_epochs = 25
        self.num_classes = LoadROCFDataset(img_size=self.img_size).num_score_classes
        print(f'number of classes {self.num_classes} inside resnet18')

        self.step_size_lr_scheduler = 4
        self.gamma_lr_scheduler = 0.5

    def plot_first_10_images(self, dataloader):
        # Create figure with black background
        plt.figure(figsize=(15, 6), facecolor='black')

        count = 0
        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images = batch[0]

            for j in range(images.size(0)):
                if count >= 12:
                    break

                # images[j] má tvar [3, H, W] pre RGB
                image = images[j].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]

                ax = plt.subplot(2, 6, count + 1)
                ax.imshow(image)  # RGB obrázok, cmap nepoužívame
                ax.axis('off')
                ax.set_facecolor('black')

                count += 1

            if count >= 12:
                break

        plt.tight_layout()
        plt.show()

    def model_training(self, f, global_pooling=True, train_name='train'):
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

        # Plot the first 10 images in the training loader
        # self.plot_first_10_images(train_loader)
        # return

        loss_fn, model, optimizer, scheduler = self.initialize_model(global_pooling)

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

        # Save the trained model
        # torch.save(model.state_dict(), f'{train_name}_model.pth')

    def load_model(self, model, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()

    def model_testing(self, f, model_path, global_pooling=True):
        # Load dataset
        ROCF_dataset = LoadROCFDataset(img_size=self.img_size)

        general_resnet_training = GeneralResNetTraining(
            f=f, img_size=self.img_size, augmentation=self.augmentations, pos_embedding=False,
            preprocessing=self.preprocessing
        )
        val_test_transform = general_resnet_training.get_resnet_transforms(default=True)

        # Initialize only test dataset (train not needed)
        _, _, test_loader = general_resnet_training.initialize_datasets(
            rocf_dataset=ROCF_dataset, transform=val_test_transform, val_test_transform=val_test_transform
        )

        # Initialize model and loss function
        loss_fn, model, _, _ = self.initialize_model(global_pooling)

        # Load best trained model weights
        self.load_model(model, model_path)

        # Evaluate on test set
        print(f"Evaluating model from {model_path} on test set:")
        general_resnet_training.test(
            model=model, dataloader=test_loader, loss_fn=loss_fn, prefix='Test'
        )

        print(f"\n\nEvaluating model on test set with test-time augmentation:")
        general_resnet_training.test_with_tta_metrics(
            model=model, rocf_dataset=ROCF_dataset, loss_fn=loss_fn, model_type='resnet18'
        )

    def initialize_model(self, global_pooling):
        # Initialize the ResNet18 model, loss function, and optimizer
        model = ResNet18Classifier(self.num_classes, global_pooling=global_pooling).to(self.device)
        loss_fn = nn.CrossEntropyLoss()  # Assuming classification task
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0025)  # L2 regularization
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size_lr_scheduler,
                                              gamma=self.gamma_lr_scheduler)  # Learning rate scheduler
        return loss_fn, model, optimizer, scheduler


    def visualize_with_heatmap(self):
        models = [
            # 'C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet_new_results/resnet18_pretrain_lr0.0001_sts3_15e_is400_GP_combo_model.pth',
            #       'C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet_new_results/resnet18_pretrain_lr0.0001_sts3_15e_is400_notGP_rotate_model.pth'
                  ]
        file_names = ['r18_lr0.0001_GB_combo_model',
                  'r18_lr0.0001_notGB_rotate_model']

        for i in range(len(models)):
            model_name = models[i]
            file_name = file_names[i]

            global_pooling = True if '_GP_' in model_name else False
            lr = 0.0001 if '0.0001' in model_name else 0.001
            _, model, _, _ = self.initialize_model(global_pooling, lr)
            self.augmentations = 'none'

            model.load_state_dict(torch.load(model_name))
            model.eval()

            LoadROCFDataset(transform=GeneralResNetTraining(
                    f=None, img_size=self.img_size, augmentation=self.augmentations, pos_embedding=False,
                    preprocessing=self.preprocessing
                ).get_resnet_transforms(default=True)
            ).visualize_heatmaps(model, model_name=file_name)

    def visualize_with_gradcam(self):
        models = [
            # 'C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet_new_results/resnet18_pretrain_lr0.0001_sts3_15e_is400_GP_combo_model.pth',
            # 'C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet_new_results/resnet18_pretrain_lr0.0001_sts3_15e_is400_notGP_rotate_model.pth'
        ]
        file_names = [
            'r18_lr0.0001_GB_combo_model',
            'r18_lr0.0001_notGB_rotate_model'
        ]

        for i in range(len(models)):
            model_name = models[i]
            file_name = file_names[i]

            global_pooling = True if '_GP_' in model_name else False
            lr = 0.0001 if '0.0001' in model_name else 0.001
            _, model, _, _ = self.initialize_model(global_pooling, lr)
            self.augmentations = 'none'

            model.load_state_dict(torch.load(model_name, map_location=self.device))
            model.eval()

            LoadROCFDataset(transform=GeneralResNetTraining(
                f=None, img_size=self.img_size, augmentation=self.augmentations, pos_embedding=False,
                preprocessing=self.preprocessing
            ).get_resnet_transforms(default=True)
                            ).visualize_gradcam(model, model_name=file_name, device=self.device, model_type='resnet18')


# trainer = TrainResNet18('color')
# trainer.model_training(global_pooling=False)

#########################################################spravit automaticke prepinanie medzi 6 a 4 classes
########################################################automaticke mena filov podla parametrov
train = True
test = False
analyze_results = False

# VISUALIZE WITH HEATMAPS
# trainer = TrainResNet18('none')
# trainer.visualize_with_heatmap()
# train = False


# VISUALIZE WITH Grad-Cam
# trainer = TrainResNet18('none')
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
    # transformations = ['none', 'color', 'translate', 'crop', 'rotate', 'combo', 'combo_crop', 'all']
    transformations = ['combo_crop', 'all']
    # global_pooling_values = [True, False]
    preprocessing_settings = ['grey', 'restoration']
    # transformations = ['combo']
    global_pooling_values = [True, False]
    global_pooling_values = [False]

    for global_pooling in global_pooling_values:
        for transformation in transformations:
            for preprocessing in preprocessing_settings:
                trainer = TrainResNet18(augmentations=transformation, preprocessing=preprocessing)

                train_name = f'./results/resnet18_pretrain_lr{trainer.lr}_sts{trainer.step_size_lr_scheduler}_' \
                             f'{trainer.num_epochs}e_is{trainer.img_size}_{"GP" if global_pooling else "notGP"}_' \
                             f'{transformation}_{preprocessing}_{trainer.num_classes}cls'
                print("\n-------------------------------------\n")
                print(train_name)

                f = f'{train_name}.txt'
                with open(f, 'w') as file:
                    file.write(f"{f}\n")
                # Call the model training with the current global_pooling and transformation settings
                trainer.model_training(f=f, global_pooling=global_pooling, train_name=train_name)



# if test:
#     # transformations = ['none', 'color', 'translate', 'crop', 'rotate', 'combo', 'combo_crop', 'all']
#     # global_pooling_values = [True, False]
#     preprocessing_settings = ['grey', 'restoration']
#     transformations = ['combo']
#     global_pooling_values = [True]
#     model_dir = "C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet_new_results/GP/"
#
#     for global_pooling in global_pooling_values:
#         for transformation in transformations:
#             for preprocessing in preprocessing_settings:
#                 if not global_pooling and transformation == 'crop':
#                     continue
#                 trainer = TrainResNet18(augmentations=transformation, preprocessing=preprocessing)
#
#                 test_name = f'resnet18_pretrain_lr0.0001_sts5_25e_is500_{"GP" if global_pooling else "notGP"}_{transformation}_{preprocessing}'
#                 # test_name = f'resnet18_pretrain_lr0_0001_sts3_15e_is400_{"GP" if global_pooling else "notGP"}_{transformation}'
#                 test_name = f'resnet18_pretrain_lr{trainer.lr}_sts{trainer.step_size_lr_scheduler}_' \
#                             f'{trainer.num_epochs}e_is{trainer.img_size}_{"GP" if global_pooling else "notGP"}_{transformation}_{preprocessing}'
#                 print("\n-------------------------------------\n")
#                 print(test_name)
#                 model_path = model_dir + test_name + '_model.pth'
#                 print(model_path)
#                 print(model_path)
#
#                 f = f'{test_name}.txt'
#                 with open(f, 'w') as file:
#                     file.write(f"{f}\n")
#                 # Call the model training with the current global_pooling and transformation settings
#                 trainer.model_testing(f=f, global_pooling=global_pooling, model_path=model_path)


# if analyze_results:
#     model_dir = "C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/resnet_new_results/GP_25e_6classes/grey"
#
#     f = f'resnet18_GP_25_grey_6classes_test_results_analysis.txt'
#     general_resnet_training = GeneralResNetTraining(
#         f=f, pos_embedding=False
#     )
#     general_resnet_training.analyze_model_logs_with_tta(log_dir=model_dir)

