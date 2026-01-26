import torch
import ast
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
import re
from collections import defaultdict
import pandas as pd
import os
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np

from ROCFDataset_for_CNN import ROCFDataset, LoadROCFDataset


class GeneralResNetTraining():
    def __init__(self, f, img_size=500, augmentation='none', pos_embedding=False, preprocessing='grey'):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.split_dir = "./orezane_1500x1500px/Train_Val_Test_split/"
        self.img_size = img_size
        self.augmentation=augmentation
        self.pos_embedding = pos_embedding
        self.preprocessing = preprocessing
        self.f = f

    def get_swin_transformer_transforms(self, default=False):
        def to_rgb(image):
            if isinstance(image, torch.Tensor):
                image = F.to_pil_image(image)
            if self.pos_embedding:
                return F.to_tensor(image)
            return F.to_tensor(image.convert("RGB"))

        def scale_to_minus_one_to_one(image):
            return (image * 2.0) - 1.0  # Scale to [-1, 1]

        if default:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])

        if self.augmentation == "crop":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=(self.img_size, self.img_size), scale=(0.98, 1.0)),
                transforms.Lambda(to_rgb),  # Ensure 3 channels (RGB)
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])
        elif self.augmentation == "translate":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), fill=255),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])
        elif self.augmentation == "color":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])
        elif self.augmentation == "rotate":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=5, fill=255),  # ±5 degrees
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)
            ])
        elif self.augmentation == "combo":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), fill=255),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)
            ])
        elif self.augmentation == "combo_crop":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.RandomResizedCrop(size=(self.img_size, self.img_size), scale=(0.98, 1.0)),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)
            ])
        elif self.augmentation == "all":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), fill=255),
                transforms.RandomResizedCrop(size=(self.img_size, self.img_size), scale=(0.98, 1.0)),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])

    def get_resnet_transforms(self, default=False):
        to_rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # duplikovanie 1-kanálu do 3-kanálového RGB

        if default:
            return transforms.Compose([
                transforms.ToTensor(),
                to_rgb
            ])

        if self.augmentation == "crop":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.98, 1.0)
                ),
                transforms.ToTensor(),
                to_rgb
            ])
        elif self.augmentation == "translate":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), fill=255),
                transforms.ToTensor(),
                to_rgb
            ])
        elif self.augmentation == "color":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.ToTensor(),
                to_rgb
            ])
        elif self.augmentation == "rotate":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.ToTensor(),
                to_rgb
            ])
        elif self.augmentation == "combo":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), fill=255),
                transforms.ToTensor(),
                to_rgb
            ])
        elif self.augmentation == "combo_crop":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.98, 1.0)
                ),
                transforms.ToTensor(),
                to_rgb
            ])
        elif self.augmentation == "all":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), fill=255),
                transforms.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.98, 1.0)
                ),
                transforms.ToTensor(),
                to_rgb
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                to_rgb
            ])

    def initialize_datasets(self, rocf_dataset, transform, val_test_transform, logging=True):
        X, y, c = [], [], []

        for i in range(len(rocf_dataset)):
            img_name = rocf_dataset.get_image_path_all_files(i)
            name, order, score = rocf_dataset.extract_from_name(img_name)
            X.append(img_name)
            y.append(score)
            c.append(rocf_dataset.class_from_score(score))

        # HERE CHANGE SIZE OF DATASET FOR TESTING
        X = X[:]
        y = y[:]
        c = c[:]

        report = "\nnumber of all images:" + str(len(y))
        if logging:
            self.logging(report=report)

        os.makedirs(self.split_dir, exist_ok=True)
        number_classes = LoadROCFDataset(img_size=self.img_size).num_score_classes
        train_file = os.path.join(self.split_dir, f'train_len_{str(len(X))}_{number_classes}_classes.txt')
        val_file = os.path.join(self.split_dir, f'val_len_{str(len(X))}_{number_classes}_classes.txt')
        test_file = os.path.join(self.split_dir, f'test_len_{str(len(X))}_{number_classes}_classes.txt')


        # Check if split files already exist
        if all(os.path.exists(file) for file in [train_file, val_file, test_file]):
            X_test, X_train, X_val, c_test, c_train, c_val, y_test, y_train, y_val = self.load_existing_dataset_split(
                X=X, y=y, c=c, train_file=train_file, val_file=val_file, test_file=test_file, logging=logging
            )
        else:
            X_test, X_train, X_val, c_test, c_train, c_val, y_test, y_train, y_val = self.create_new_dataset_split(
                X=X, y=y, c=c, train_file=train_file, val_file=val_file, test_file=test_file, logging=logging
            )

        report = f'number of images in trainset: {len(X_train)},  valset: {len(X_val)}, testset: {len(X_test)}'
        if logging:
            self.logging(report=report)
        report = f'\nwhole set class counts: {dict(sorted(dict(Counter(c)).items()))}' + \
                 f'\ntrain class counts: {dict(sorted(dict(Counter(c_train)).items()))}' + \
                 f'\nval class counts: {dict(sorted(dict(Counter(c_val)).items()))}' + \
                 f'\ntest class counts: {dict(sorted(dict(Counter(c_test)).items()))}'
        if logging:
            self.logging(report=report)

        test_loader, train_loader, val_loader = self.split_datasets(
            X_test=X_test, X_train=X_train, X_val=X_val,
            y_test=y_test, y_train=y_train, y_val=y_val,
            transform=transform, val_test_transform=val_test_transform,
            logging=logging
        )

        return train_loader, val_loader, test_loader

    def create_new_dataset_split(self, X, y, c, train_file, val_file, test_file, logging=True):
        # Create new stratified split
        X_temp, X_test, y_temp, y_test, c_temp, c_test = train_test_split(
            X, y, c, test_size=0.1, random_state=42, stratify=c
        )
        X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
            X_temp, y_temp, c_temp, test_size=0.1, random_state=42, stratify=c_temp
        )

        # Save splits to files
        with open(train_file, "w") as f:
            f.write("\n".join(X_train).replace('//', '/'))
        with open(val_file, "w") as f:
            f.write("\n".join(X_val).replace('//', '/'))
        with open(test_file, "w") as f:
            f.write("\n".join(X_test).replace('//', '/'))

        if logging:
            self.logging(report="\nCreated new split files!!!\n")
        return X_test, X_train, X_val, c_test, c_train, c_val, y_test, y_train, y_val

    def load_existing_dataset_split(self, X, y, c, train_file, val_file, test_file, logging=True):
        print(f'load existing dataset split: {train_file}, {val_file}, {test_file}')
        with open(train_file) as f:
            X_train = [line.strip() for line in f]
        with open(val_file) as f:
            X_val = [line.strip() for line in f]
        with open(test_file) as f:
            X_test = [line.strip() for line in f]

        # Map file paths back to scores/classes
        lookup = {img.replace('//', '/'): (score, cls) for img, score, cls in zip(X, y, c)}
        y_train, c_train = zip(*[lookup[img] for img in X_train])
        y_val, c_val = zip(*[lookup[img] for img in X_val])
        y_test, c_test = zip(*[lookup[img] for img in X_test])

        if logging:
            self.logging(report="\nLoaded existing split files!!!\n")

        return X_test, X_train, X_val, list(c_test), \
               list(c_train), list(c_val), \
               list(y_test), list(y_train), list(y_val)

    def split_datasets(self, X_test, X_train, X_val, y_test, y_train, y_val, transform, val_test_transform, logging=True):

        train_dataset = ROCFDataset(
            image_paths=X_train, scores=y_train, transform=transform, pos_embedding=self.pos_embedding,
            preprocessing=self.preprocessing
        )
        val_dataset = ROCFDataset(
            image_paths=X_val, scores=y_val, transform=val_test_transform, pos_embedding=self.pos_embedding,
            preprocessing=self.preprocessing
        )
        test_dataset = ROCFDataset(
            image_paths=X_test, scores=y_test, transform=val_test_transform, pos_embedding=self.pos_embedding,
            preprocessing=self.preprocessing
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        report = f'\nTrain class counts: {self.count_classes_in_dataset(train_dataset)}' + \
                 f'\nVal class counts: {self.count_classes_in_dataset(val_dataset)}' + \
                 f'\nTest class counts: {self.count_classes_in_dataset(test_dataset)} \n'
        if logging:
            self.logging(report=report)

        return test_loader, train_loader, val_loader

    def count_classes_in_dataset(self, dataset):
        """
        Count number of samples per class in a DataLoader
        where labels are one-hot encoded.
        """
        class_counts = Counter()
        for i in range(len(dataset)):
            _, score_one_hot, _ = dataset[i]
            cls = torch.argmax(score_one_hot).tolist()
            class_counts.update([cls])
        return dict(sorted(dict(class_counts).items()))

    def training_loop(self, train_name, num_epochs, train_loader, val_loader, model, loss_fn, optimizer, scheduler):
        # Training loop
        train_loss_epochs = []
        val_loss_epochs = []
        val_accuracy_epochs = []
        val_precision_epochs = []
        val_recall_epochs = []
        val_f1_epochs = []

        best_f1 = -1.0  # track the best F1 score so far

        for epoch in range(num_epochs):
            loss = self.train(train_loader, model, loss_fn, optimizer)
            loss_value = loss

            report = f"Epoch [{epoch + 1}/{num_epochs}], Average epoch loss: {loss_value:.4f}"
            self.logging(report)

            val_accuracy, val_loss, val_precision, val_recall, val_f1 = self.test(
                model, val_loader, loss_fn, prefix='Val'
            )

            train_loss_epochs.append(loss_value)
            val_loss_epochs.append(val_loss)
            val_accuracy_epochs.append(val_accuracy)
            val_precision_epochs.append(val_precision)
            val_recall_epochs.append(val_recall)
            val_f1_epochs.append(val_f1)

            scheduler.step()

            # Save checkpoint only if F1 improves
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), f'{train_name}_model.pth')
                report = f"New best model saved with F1={best_f1:.4f}!!\n"
                self.logging(report)

        self.plot_train_val_stats(
            train_loss_epochs, train_name, val_accuracy_epochs, val_loss_epochs,
            val_precision_epochs, val_recall_epochs, val_f1_epochs
        )
        torch.save(model.state_dict(), f'{train_name}_last_model.pth')

    def logging(self, report, f=None):
        if f == None:
            f = self.f
        with open(f, 'a') as file:
            file.write(f"{report}\n")

    def plot_train_val_stats(self, train_loss_epochs, train_name, val_accuracy_epochs, val_loss_epochs,
                             val_precision_epochs=None, val_recall_epochs=None, val_f1_epochs=None):

        plt.plot(train_loss_epochs, c='r', label='Train loss')
        plt.plot(val_loss_epochs, c='b', label='Validation loss')
        plt.title(f'{train_name} loss')
        plt.legend()
        plt.savefig(f'{train_name}_loss.png')
        plt.show()
        plt.clf()

        plt.plot(val_accuracy_epochs, c='g', label='Validation accuracy')
        plt.title(f'{train_name} val accuracy')
        plt.legend()
        plt.savefig(f'{train_name}_val_accuracy.png')
        plt.show()
        plt.clf()

        if val_precision_epochs is not None:
            plt.plot(val_precision_epochs, c='c', label='Validation precision')
            plt.plot(val_recall_epochs, c='m', label='Validation recall')
            plt.plot(val_f1_epochs, c='g', label='Validation F1')
            plt.title(f'{train_name} val metrics')
            plt.legend()
            plt.savefig(f'{train_name}_val_metrics.png')
            plt.show()
            plt.clf()

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()

        epoch_loss = 0.0  # accumulate total loss
        num_batches = len(dataloader)

        for batch, (X, y, _) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Forward
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss (as float)
            epoch_loss += loss.item()

            # Logging every 10 batches
            if batch % 10 == 0:
                current = batch * len(X)
                report = f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]"
                self.logging(report)

        # Compute average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches

        return avg_epoch_loss  # float

    def test(self, model, dataloader, loss_fn, prefix='Test', test_results={}):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        if model is not None:
            model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        all_img_paths = []

        with torch.no_grad():
            for X, y, img_paths in dataloader:
                if test_results == {}:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()

                    pred_classes = pred.argmax(1).cpu()
                else:
                    # Read cached predictions using image paths
                    # test_results format: {image_path: class_idx}
                    pred_classes = torch.tensor(
                        [test_results[p] for p in img_paths],
                        dtype=torch.long
                    )

                true_classes = y.argmax(1).cpu()
                all_preds.extend(pred_classes.tolist())
                all_labels.extend(true_classes.tolist())
                all_img_paths.extend(img_paths)

        test_loss /= num_batches

        all_preds_tensor = torch.tensor(all_preds)
        all_labels_tensor = torch.tensor(all_labels)
        accuracy = (all_preds_tensor == all_labels_tensor).float().mean().item()
        mae = mean_absolute_error(all_labels, all_preds)

        # Vectorized computation of wrongly classified images
        wrong_mask = all_preds_tensor != all_labels_tensor
        wrong_img_paths = [p for p, wrong in zip(all_img_paths, wrong_mask.tolist()) if wrong]

        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(LoadROCFDataset(img_size=self.img_size).num_score_classes)))
        cm_str = np.array2string(cm, separator=', ')

        # histograms for wrongly classified samples
        class_hist, score_hist = self.wrongly_classified_scores_histogram(wrong_img_paths)

        report = (f"{prefix} Error: \n"
                  f"{(all_preds_tensor == all_labels_tensor).sum()} correct out of {size}\n"
                  f"Avg {prefix} loss: {test_loss:>8f}\n"
                  f"Mean absolute score (MAE): {mae:.4f}\n"
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n" 
                  f"{prefix} Confusion Matrix:\n{cm_str}\n"
                  f"Wrong score histogram (score -> count):\n{score_hist}\n"
                  f"Wrong class histogram (class -> count):\n{class_hist}"
                  f"\nWrongly classified images:\n" + "\n".join(wrong_img_paths) + "\n\n")

        self.logging(report=report)

        return accuracy, test_loss, precision, recall, f1

    def wrongly_classified_scores_histogram(self, wrong_img_paths):
        wrong_scores = []
        wrong_true_classes = []
        rocf_class = LoadROCFDataset(img_size=self.img_size)

        for p in wrong_img_paths:
            _, _, score = rocf_class.extract_from_name(p)
            wrong_scores.append(score)
            wrong_true_classes.append(rocf_class.class_from_score(score))#

        # === Dictionary-based histograms ===
        # Score bins: 0, 0.5, ..., 36
        score_bins = np.arange(0, 36.5, 0.5).tolist()
        class_bins = list(range(rocf_class.num_score_classes))

        score_hist = self.build_histogram_dict(wrong_scores, score_bins)
        class_hist = self.build_histogram_dict(wrong_true_classes, class_bins)
        return class_hist, score_hist

    def build_histogram_dict(self, values, bins):
        hist = {b: 0 for b in bins}
        for v in values:
            if v in hist:
                hist[v] += 1
        return {k: v for k, v in hist.items() if v > 0}

    def test_with_tta_metrics(self, model, rocf_dataset, loss_fn, model_type='resnet18'):
        """
        Test model with Test-Time Augmentation (TTA) using multiple transformations,
        compute majority-vote predictions, per-augmentation and average losses,
        and agreement metrics.
        """
        all_losses, all_preds_aug, augmentations, original_transform, per_aug_losses, test_loader, y_true = self.tta_testing(
            loss_fn, model, model_type, rocf_dataset)

        report = self.compute_metrics_tta_testing(all_losses, all_preds_aug, augmentations, per_aug_losses, test_loader,
                                                  y_true)

        self.logging(report)
        self.augmentation = original_transform

    def tta_testing(self, loss_fn, model, model_type, rocf_dataset):
        augmentations = ['none', 'rotate', 'color', 'translate', 'combo']
        all_preds_aug = []
        all_losses = []
        img_paths_per_aug = {}  # store img_paths per augmentation

        prev_img_paths = None
        y_true = []
        model.eval()
        original_transform = self.augmentation
        per_aug_losses = {}

        for aug in augmentations:
            self.augmentation = aug

            # Choose the right transform
            train_transform = self.get_resnet_transforms() if model_type == 'resnet18' else self.get_swin_transformer_transforms()
            val_test_transform = self.get_resnet_transforms() if model_type == 'resnet18' else self.get_swin_transformer_transforms()

            # Get test_loader with the current augmentation
            _, _, test_loader = self.initialize_datasets(
                rocf_dataset=rocf_dataset,
                transform=train_transform,
                val_test_transform=val_test_transform,
                logging=False
            )

            aug_preds = []
            aug_loss = 0.0
            current_img_paths = []
            with torch.no_grad():
                for X, y, img_paths in test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    aug_loss += loss_fn(pred, y).item()
                    aug_preds.extend(pred.argmax(1).cpu().tolist())
                    current_img_paths.extend(img_paths)

                    if aug == 'none':
                        true_classes = y.argmax(1).cpu()
                        y_true.extend(true_classes.tolist())

            # Check that image order matches previous augmentation
            if prev_img_paths is not None:
                assert prev_img_paths == current_img_paths, f"Image order mismatch for augmentation {aug}!"
                print(f'Image order is correct? : {prev_img_paths == current_img_paths}')
            prev_img_paths = current_img_paths
            img_paths_per_aug[aug] = current_img_paths

            # Store per-augmentation predictions and average loss
            all_preds_aug.append(aug_preds)
            aug_loss_avg = aug_loss / len(test_loader)
            all_losses.append(aug_loss_avg)
            per_aug_losses[aug] = aug_loss_avg
        return all_losses, all_preds_aug, augmentations, original_transform, per_aug_losses, test_loader, y_true

    def compute_metrics_tta_testing(self, all_losses, all_preds_aug, augmentations, per_aug_losses, test_loader, y_true):
        # Convert predictions to tensor (shape: num_augs x num_samples)
        preds_tensor = torch.tensor(all_preds_aug)
        voted_preds = preds_tensor.mode(dim=0).values.tolist()

        # Compute standard metrics based on majority votes
        accuracy = (torch.tensor(voted_preds) == torch.tensor(y_true)).float().mean().item()
        precision = precision_score(y_true, voted_preds, average='weighted', zero_division=0)
        recall = recall_score(y_true, voted_preds, average='weighted', zero_division=0)
        f1 = f1_score(y_true, voted_preds, average='weighted', zero_division=0)
        mae = mean_absolute_error(y_true, voted_preds)
        avg_loss = sum(all_losses) / len(all_losses)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, voted_preds)
        cm_str = np.array2string(cm, separator=', ')

        # Wrongly classified images based on majority vote
        wrong_mask = torch.tensor(voted_preds) != torch.tensor(y_true)
        wrong_img_paths = [p for p, wrong in
                           zip([img_path for _, _, img_path in test_loader.dataset], wrong_mask.tolist()) if
                           wrong]

        class_hist, score_hist = self.wrongly_classified_scores_histogram(wrong_img_paths)

        # Agreement metrics
        none_preds = preds_tensor[0]  # predictions for 'none' augmentation

        # Agreement with 'none' for each augmentation separately
        agree_with_none = {}
        for idx, aug in enumerate(augmentations[1:], start=1):
            aug_preds = preds_tensor[idx]
            agree_percent = (aug_preds == none_preds).float().mean().item() * 100  # percent of images
            agree_with_none[aug] = agree_percent

        # Number of augmentations that agree with majority class per image
        voted_tensor = torch.tensor(voted_preds).unsqueeze(0)  # shape: (1, num_samples)
        agree_count_per_image = (preds_tensor == voted_tensor).sum(dim=0).float()  # counts per image
        avg_agree_with_majority = agree_count_per_image.mean().item()

        # Percentage of images where all augmentations agree with majority
        all_agree_with_majority = (agree_count_per_image == len(augmentations)).float().mean().item() * 100

        # Build report
        report = f"Test TTA metrics:\nAugmentations: {augmentations}\n"
        report += f"Average loss across all augmentations: {avg_loss:.4f}\n"

        for aug in augmentations:
            report += f"Average loss for {aug}: {per_aug_losses[aug]:.4f}\n"
        report += f"Accuracy (majority vote): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n"
        report += f"Mean absolute score (MAE): {mae:.4f}\n"
        report += f"Confusion Matrix:\n{cm_str}"

        # Agreement report per augmentation vs none
        report += "\nAgreement with 'none' augmentation:\n"
        for aug, pct in agree_with_none.items():
            report += f"{aug}: {pct:.2f}% images agree with 'none'\n"

        report += f"\nAvg number of augmentations agreeing with majority per image: {avg_agree_with_majority:.2f}\n"
        report += f"Percentage of images where all augmentations agree with majority: {all_agree_with_majority:.2f}%\n"
        report += f"Wrong score histogram (score -> count):\n{score_hist}\n"
        report += f"Wrong class histogram (class -> count):\n{class_hist}\n"
        report += "Wrongly classified images after majority vote:\n" + "\n".join(wrong_img_paths) + "\n"

        return report

    def analyze_model_logs_with_tta(self, log_dir):
        """
        Analyze training and testing logs in a directory, including TTA results,
        per-augmentation statistics, agreement with 'none', wrongly classified images,
        and ordering of models.

        Args:
            log_dir (str): Path to directory containing .txt log files.

        Returns:
            dict: Dictionary with overall statistics and DataFrames.
        """
        results = []
        always_wrong_none_best = defaultdict(int)
        always_wrong_none_last = defaultdict(int)
        always_wrong_tta_best = defaultdict(int)
        always_wrong_tta_last = defaultdict(int)

        txt_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]

        for txt_file in txt_files:
            self.extract_stats_one_model(always_wrong_none_best, always_wrong_none_last,
                                         always_wrong_tta_best, always_wrong_tta_last, log_dir, results, txt_file)

        # Build DataFrame
        df = pd.DataFrame(results)

        # Build human-readable report
        report = self.create_report_stats(always_wrong_none_best, always_wrong_none_last,
                                          always_wrong_tta_best, always_wrong_tta_last, df, txt_files)

        # Print via self.logging
        self.logging(report=report)

    def extract_stats_one_model(self, always_wrong_none_best, always_wrong_none_last,
                                always_wrong_tta_best, always_wrong_tta_last, log_dir, results, txt_file):
        file_path = os.path.join(log_dir, txt_file)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract LAST MODEL metrics
        # last_acc, last_f1, last_loss, last_prec, last_rec = self.extract_stats_last_model_one_doc(content)
        last_acc, last_f1, last_loss, last_prec, last_rec, wrong_none_imgs_last, \
        last_mae, last_confusion_matrix, last_wrong_score_hist, \
        last_wrong_class_hist = self.extract_stats_model_one_doc(
            always_wrong_none_last, content, prefix='LAST MODEL', ending='\nBEST VAL MODEL Test Error:'
        )

        # Extract BEST VAL MODEL metrics
        best_acc, best_f1, best_loss, best_prec, best_rec, wrong_none_imgs_best, \
        best_mae, best_confusion_matrix, best_wrong_score_hist, \
        best_wrong_class_hist = self.extract_stats_model_one_doc(
            always_wrong_none_best, content, prefix='BEST VAL MODEL'
        )

        # Extract last model TTA results
        last_aug_agree_with_none, last_avg_agree_num, last_full_agree_pct, last_per_aug_losses, \
        last_tta_acc, last_tta_f1, last_tta_loss, last_tta_prec, last_tta_rec, \
        last_wrong_tta_imgs, last_tta_confusion_matrix, last_tta_wrong_score_hist, \
        last_tta_wrong_class_hist, last_tta_mae = self.extract_stats_tta_one_doc(always_wrong_tta_last, content,
                                                                       prefix='LAST MODEL',
                                                                       ending='\nBEST VAL MODEL Test Error:')

        # Extract best model TTA results
        best_aug_agree_with_none, best_avg_agree_num, best_full_agree_pct, best_per_aug_losses, \
        best_tta_acc, best_tta_f1, best_tta_loss, best_tta_prec, best_tta_rec, \
        best_wrong_tta_imgs, best_tta_confusion_matrix, best_tta_wrong_score_hist, \
        best_tta_wrong_class_hist, best_tta_mae = self.extract_stats_tta_one_doc(always_wrong_tta_best, content,
                                                                       prefix='BEST VAL MODEL',
                                                                       ending=None)

        results.append({
            'file': txt_file,
            'last_acc': last_acc,
            'last_prec': last_prec,
            'last_rec': last_rec,
            'last_f1': last_f1,
            'last_loss': last_loss,
            'last_mae': last_mae,
            'last_confusion_matrix': last_confusion_matrix,
            'last_wrong_score_hist': last_wrong_score_hist,
            'last_wrong_class_hist': last_wrong_class_hist,
            'best_acc': best_acc,
            'best_prec': best_prec,
            'best_rec': best_rec,
            'best_f1': best_f1,
            'best_loss': best_loss,
            'best_mae': best_mae,
            'best_confusion_matrix': best_confusion_matrix,
            'best_wrong_score_hist': best_wrong_score_hist,
            'best_wrong_class_hist': best_wrong_class_hist,
            'last_tta_acc': last_tta_acc,
            'last_tta_prec': last_tta_prec,
            'last_tta_rec': last_tta_rec,
            'last_tta_f1': last_tta_f1,
            'last_tta_loss': last_tta_loss,
            'last_tta_confusion_matrix': last_tta_confusion_matrix,
            'last_tta_wrong_score_hist': last_tta_wrong_score_hist,
            'last_tta_wrong_class_hist': last_tta_wrong_class_hist,
            'last_tta_mae': last_tta_mae,
            'last_per_aug_losses': last_per_aug_losses,
            'last_aug_agree_with_none': last_aug_agree_with_none,
            'last_avg_agree_num': last_avg_agree_num,
            'last_full_agree_pct': last_full_agree_pct,
            'best_tta_acc': best_tta_acc,
            'best_tta_prec': best_tta_prec,
            'best_tta_rec': best_tta_rec,
            'best_tta_f1': best_tta_f1,
            'best_tta_loss': best_tta_loss,
            'best_tta_confusion_matrix': best_tta_confusion_matrix,
            'best_tta_wrong_score_hist': best_tta_wrong_score_hist,
            'best_tta_wrong_class_hist': best_tta_wrong_class_hist,
            'best_tta_mae': best_tta_mae,
            'best_per_aug_losses': best_per_aug_losses,
            'best_aug_agree_with_none': best_aug_agree_with_none,
            'best_avg_agree_num': best_avg_agree_num,
            'best_full_agree_pct': best_full_agree_pct,
            'wrong_none_best': wrong_none_imgs_best,
            'wrong_none_last': wrong_none_imgs_last,
            'last_wrong_tta': last_wrong_tta_imgs,
            'best_wrong_tta': best_wrong_tta_imgs,
        })

    def extract_stats_tta_one_doc(self, always_wrong_tta, content, prefix, ending=None):
        # Find the position of ending in content
        pos = content.find(prefix)

        # If ending exists in content, slice up to that position
        if pos != -1:
            content = content[pos:]

        # Find the position of ending in content
        pos = -1 if ending is None else content.find(ending)

        # If ending exists in content, slice up to that position
        if ending is not None and pos != -1:
            content = content[:pos]

        tta_match = re.search(
            r"Test TTA metrics:\s*"
            r"Augmentations:\s*(\[.*?\])\s*"
            r"(?:Average loss across all augmentations:\s*([0-9.]+)\s*)?"  # capture avg TTA loss
            r"((?:Average loss for .*?:\s*[0-9.]+\s*)*)"  # all per-augmentation losses
            r"Accuracy \(majority vote\):\s*([0-9.]+),\s*Precision:\s*([0-9.]+),\s*Recall:\s*([0-9.]+),\s*F1:\s*([0-9.]+)\s*"
            r"(?:Mean absolute score \(MAE\):\s*([0-9.]+)\s*)?"
            r"Confusion Matrix:\s*(\[\[[\s\S]*?\]\])"
            r"\n(?:Agreement with 'none' augmentation:\s*((?:.+?\n)+?)\n)?"
            r"(?:Avg number of augmentations agreeing with majority per image:\s*([0-9.]+)\s*\n)?"
            r"(?:Percentage of images where all augmentations agree with majority:\s*([0-9.]+)%\s*\n)?"
            r"Wrong score histogram \(score -> count\):\s*"
            r"(\{.*?\})\s*"
            r"Wrong class histogram \(class -> count\):\s*"
            r"(\{.*?\})\s*"
            r"Wrongly classified images(?: after majority vote)?:\s*\n"
            r"((?:.+\n)+)",
            content, re.DOTALL
        )
        per_aug_losses = {}
        aug_agree_with_none = {}
        tta_acc = tta_prec = tta_rec = tta_f1 = None
        tta_loss = avg_agree_num = full_agree_pct = None
        wrong_tta_imgs = []
        confusion_matrix = None
        wrong_score_hist = {}
        wrong_class_hist = {}
        mae = None
        if tta_match:
            (aug_str, tta_loss, losses_block,
             tta_acc, tta_prec, tta_rec, tta_f1,
             mae,
             confusion_matrix_str,
             agree_block, avg_agree_num, full_agree_pct,
             wrong_score_hist_str, wrong_class_hist_str,
             wrong_tta) = tta_match.groups()

            augmentations = [a.strip().strip("'\"").replace('\'', '').replace('[', '').replace(']', '')
                             for a in aug_str.split(',')]
            tta_acc, tta_prec, tta_rec, tta_f1 = map(float, [tta_acc, tta_prec, tta_rec, tta_f1])
            wrong_tta_imgs = wrong_tta.strip().splitlines()

            if tta_loss:
                tta_loss = float(tta_loss)
            if avg_agree_num:
                avg_agree_num = float(avg_agree_num)
            if full_agree_pct:
                full_agree_pct = float(full_agree_pct)

            if confusion_matrix_str:
                confusion_matrix = confusion_matrix_str

            if wrong_score_hist_str:
                wrong_score_hist = ast.literal_eval(wrong_score_hist_str)

            if wrong_class_hist_str:
                wrong_class_hist = ast.literal_eval(wrong_class_hist_str)

            for img in wrong_tta_imgs:
                always_wrong_tta[img] += 1

            # Extract per-augmentation average loss
            for aug in augmentations:
                loss_match = re.search(fr'Average loss for {re.escape(aug)}:\s*([0-9.]+)', losses_block)
                if loss_match:
                    per_aug_losses[aug] = float(loss_match.group(1))

            # Extract agreement with none
            if agree_block:
                for aug in augmentations[1:]:  # exclude 'none'
                    agree_match = re.search(fr'{re.escape(aug)}:\s*([0-9.]+)% images agree with \'none\'',
                                            agree_block)
                    if agree_match:
                        aug_agree_with_none[aug] = float(agree_match.group(1))
        return aug_agree_with_none, avg_agree_num, full_agree_pct, per_aug_losses, \
                      tta_acc, tta_f1, tta_loss, tta_prec,tta_rec, \
               wrong_tta_imgs,  confusion_matrix, wrong_score_hist, wrong_class_hist, mae

    def extract_stats_model_one_doc(self, always_wrong_none, content, prefix, ending=None):
        # Find the position of ending in content
        pos = content.find(prefix)

        # If ending exists in content, slice up to that position
        if pos != -1:
            content = content[pos:]

        # Find the position of ending in content
        pos = -1 if ending is None else content.find(ending)

        # If ending exists in content, slice up to that position
        if ending is not None and pos != -1:
            content = content[:pos]

        best_val_match = re.search(
            prefix + r' Test Error:.*?Accuracy: ([0-9.]+), Precision: ([0-9.]+), Recall: ([0-9.]+), F1: ([0-9.]+)',
            content, re.DOTALL)
        if best_val_match:
            best_acc, best_prec, best_rec, best_f1 = map(float, best_val_match.groups())
        else:
            best_acc = best_prec = best_rec = best_f1 = None

        best_loss_match = re.search(
            prefix + r' Test Error:.*?Avg Test loss: ([0-9.]+)', content, re.DOTALL)
        best_loss = float(best_loss_match.group(1)) if best_loss_match else None

        mae_match = re.search(
            r'Mean absolute score \(MAE\):\s*([0-9.]+)', content)
        best_mae = float(mae_match.group(1)) if mae_match else None

        conf_match = re.search(
            prefix + r' Test Confusion Matrix:\s*(\[\[[\s\S]*?\]\])',
            content,
            re.DOTALL
        )
        best_confusion_matrix = conf_match.group(1) if conf_match else None

        wrong_score_hist_match = re.search(
            r'Wrong score histogram \(score -> count\):\s*(\{.*?\})',
            content, re.DOTALL)
        best_wrong_score_hist = ast.literal_eval(wrong_score_hist_match.group(1)) if wrong_score_hist_match else {}

        wrong_class_hist_match = re.search(
            r'Wrong class histogram \(class -> count\):\s*(\{.*?\})',
            content, re.DOTALL)
        best_wrong_class_hist = ast.literal_eval(wrong_class_hist_match.group(1)) if wrong_class_hist_match else {}

        # Extract wrongly classified images (normal testing)
        # stop at the next "Test TTA metrics" header (or EOF)
        wrong_none_match = re.search(
            re.escape(prefix) +
            r"[\s\S]*?"
            r"Wrongly classified images:\s*\n"
            r"((?:(?!\n\nTest TTA metrics)[^\n]*\n)*)",
            content
        )
        wrong_none_imgs = wrong_none_match.group(1).strip().splitlines() if wrong_none_match else []

        for img in wrong_none_imgs:
            always_wrong_none[img] += 1
        return best_acc, best_f1, best_loss, best_prec, best_rec, wrong_none_imgs,\
               best_mae, best_confusion_matrix, best_wrong_score_hist, best_wrong_class_hist

    def extract_stats_last_model_one_doc(self, content):
        last_model_match = re.search(
            r'LAST MODEL Test Error:.*?Accuracy: ([0-9.]+), Precision: ([0-9.]+), Recall: ([0-9.]+), F1: ([0-9.]+)',
            content, re.DOTALL)
        if last_model_match:
            last_acc, last_prec, last_rec, last_f1 = map(float, last_model_match.groups())
        else:
            last_acc = last_prec = last_rec = last_f1 = None
        last_loss_match = re.search(
            r'LAST MODEL Test Error:.*?Avg Test loss: ([0-9.]+)', content, re.DOTALL)
        last_loss = float(last_loss_match.group(1)) if last_loss_match else None
        return last_acc, last_f1, last_loss, last_prec, last_rec

    def create_report_stats(self, always_wrong_none_best, always_wrong_none_last, always_wrong_tta_best, always_wrong_tta_last, df, txt_files):
        report = "=== MODEL LOG ANALYSIS ===\n"
        report += f"Total log files: {len(txt_files)}\n\n"

        for _, row in df.iterrows():
            report += f"File: {row['file']}\n"

            report += f"  LAST MODEL -> Acc: {row['last_acc']}, Prec: {row['last_prec']}, Rec: {row['last_rec']}," \
                      f" F1: {row['last_f1']}, Loss: {row['last_loss']}, MAE: {row['last_mae']}\n"
            report += f"  Confusion matrix: \n{row['last_confusion_matrix']}\n"
            report += f"  Wrong score histogram: \n{row['last_wrong_score_hist']}\n"
            report += f"  Wrong class histogram: \n{row['last_wrong_class_hist']}\n"
            report += f"  BEST VAL MODEL -> Acc: {row['best_acc']}, Prec: {row['best_prec']}, Rec: {row['best_rec']}, " \
                      f"F1: {row['best_f1']}, Loss: {row['best_loss']}, MAE: {row['best_mae']}\n"
            report += f"  Confusion matrix: \n{row['best_confusion_matrix']}\n"
            report += f"  Wrong score histogram: \n{row['best_wrong_score_hist']}\n"
            report += f"  Wrong class histogram: \n{row['best_wrong_class_hist']}\n"

            report = self.tta_stats_str(report, row, 'last')
            report = self.tta_stats_str(report, row, 'best')

            report += f"  Wrongly classified images TTA BEST VAL MODEL (majority class): {len(row[f'best_wrong_tta'])}\n"
            report += f"  Wrongly classified images TTA LAST MODEL (majority class): {len(row[f'last_wrong_tta'])}\n"
            report += f"  Wrongly classified images (normal best val model): {len(row['wrong_none_best'])}\n"
            report += f"  Wrongly classified images (normal last model): {len(row['wrong_none_last'])}\n\n"

        report = self.stats_order_models(df, report)

        # === WRONG IMAGES ===
        report += "=== WRONG IMAGES ===\n"

        # Sort dictionaries by frequency (descending)
        sorted_wrong_none_best = dict(sorted(always_wrong_none_best.items(), key=lambda x: x[1], reverse=True))
        sorted_wrong_none_last = dict(sorted(always_wrong_none_last.items(), key=lambda x: x[1], reverse=True))
        sorted_wrong_tta_best = dict(sorted(always_wrong_tta_best.items(), key=lambda x: x[1], reverse=True))
        sorted_wrong_tta_last = dict(sorted(always_wrong_tta_last.items(), key=lambda x: x[1], reverse=True))

        report += f"Normal best val model testing ({len(sorted_wrong_none_best)}): {sorted_wrong_none_best}\n\n"
        report += f"Normal last model testing ({len(sorted_wrong_none_last)}): {sorted_wrong_none_last}\n\n"
        report += f"TTA testing best val model ({len(sorted_wrong_tta_best)}): {sorted_wrong_tta_best}\n\n"
        report += f"TTA testing last model ({len(sorted_wrong_tta_last)}): {sorted_wrong_tta_last}\n\n"

        # Print files that were always misclassified in all runs
        always_wrong_none_best_all = [img for img, count in sorted_wrong_none_best.items() if count == len(txt_files)]
        always_wrong_none_last_all = [img for img, count in sorted_wrong_none_last.items() if count == len(txt_files)]
        always_wrong_tta_all_best = [img for img, count in sorted_wrong_tta_best.items() if count == len(txt_files)]
        always_wrong_tta_all_last = [img for img, count in sorted_wrong_tta_last.items() if count == len(txt_files)]

        report += f"Images always wrong (normal best val model testing) ({len(always_wrong_none_best_all)}): {always_wrong_none_best_all}\n\n"
        report += f"Images always wrong (normal last model testing) ({len(always_wrong_none_last_all)}): {always_wrong_none_last_all}\n\n"
        report += f"Images always wrong (TTA testing best val model) ({len(always_wrong_tta_all_best)}): {always_wrong_tta_all_best}\n\n"
        report += f"Images always wrong (TTA testing last mdoel) ({len(always_wrong_tta_all_last)}): {always_wrong_tta_all_last}\n\n"

        return report

    def tta_stats_str(self, report, row, prefix):
        if row[f'{prefix}_tta_acc'] is not None:
            report += f"  TTA {prefix.upper()} MAJORITY VOTE -> Acc: {row[f'{prefix}_tta_acc']}, Prec: {row[f'{prefix}_tta_prec']}, Rec: {row[f'{prefix}_tta_rec']}," \
                      f" F1: {row[f'{prefix}_tta_f1']}, Loss: {row[f'{prefix}_tta_loss']}, MAE: {row[f'{prefix}_tta_mae']}\n"
            report += f"  Per-augmentation losses: {row[f'{prefix}_per_aug_losses']}\n"
            report += f"  Agreement with 'none': {row[f'{prefix}_aug_agree_with_none']}\n"
            report += f"  Confusion matrix: \n{row[f'{prefix}_tta_confusion_matrix']}\n"
            report += f"  Wrong score histogram: \n{row[f'{prefix}_tta_wrong_score_hist']}\n"
            report += f"  Wrong class histogram: \n{row[f'{prefix}_tta_wrong_class_hist']}\n"

            if row[f'{prefix}_avg_agree_num'] is not None:
                report += f"  Avg augmentations agreeing with majority per image: {row[f'{prefix}_avg_agree_num']}\n"

            if row[f'{prefix}_full_agree_pct'] is not None:
                report += f"  % of images where all augmentations agree with majority: {row[f'{prefix}_full_agree_pct']}%\n"


        return report

    def stats_order_models(self, df, report):
        # Ordering models (include values)
        order_by_last_f1, order_by_last_loss, order_by_last_prec, order_by_last_recall = \
            self.order_model_metrics(df, prefix='last')

        order_by_best_f1, order_by_best_loss, order_by_best_prec, order_by_best_recall = \
            self.order_model_metrics(df, prefix='best')

        order_by_last_tta_f1, order_by_last_tta_loss, order_by_last_tta_prec, order_by_last_tta_recall = \
            self.order_model_metrics(df, prefix='last_tta')

        order_by_best_tta_f1, order_by_best_tta_loss, order_by_best_tta_prec, order_by_best_tta_recall = \
            self.order_model_metrics(df, prefix='best_tta')

        report += "=== MODEL ORDERINGS ===\n"
        report += f"By LAST MODEL Recall: {order_by_last_recall}\n"
        report += f"By LAST MODEL Precision: {order_by_last_prec}\n"
        report += f"By LAST MODEL F1: {order_by_last_f1}\n"
        report += f"By LAST MODEL Loss: {order_by_last_loss}\n\n"
        report += f"By BEST VAL MODEL Recall: {order_by_best_recall}\n"
        report += f"By BEST VAL MODEL Precision: {order_by_best_prec}\n"
        report += f"By BEST VAL MODEL F1: {order_by_best_f1}\n"
        report += f"By BEST VAL MODEL Loss: {order_by_best_loss}\n\n"
        report += f"By LAST TTA Recall: {order_by_last_tta_recall}\n"
        report += f"By LAST TTA Precision: {order_by_last_tta_prec}\n"
        report += f"By LAST TTA F1: {order_by_last_tta_f1}\n"
        report += f"By LAST TTA Loss: {order_by_last_tta_loss}\n\n"
        report += f"By BEST TTA Recall: {order_by_best_tta_recall}\n"
        report += f"By BEST TTA Precision: {order_by_best_tta_prec}\n"
        report += f"By BEST TTA F1: {order_by_best_tta_f1}\n"
        report += f"By BEST TTA Loss: {order_by_best_tta_loss}\n\n"

        return report

    def order_model_metrics(self, df, prefix):
        order_by_recall = list(zip(df.sort_values(f'{prefix}_rec', ascending=False)['file'],
                                        df.sort_values(f'{prefix}_rec', ascending=False)[f'{prefix}_rec']))
        order_by_prec = list(zip(df.sort_values(f'{prefix}_prec', ascending=False)['file'],
                                      df.sort_values(f'{prefix}_prec', ascending=False)[f'{prefix}_prec']))
        order_by_f1 = list(zip(df.sort_values(f'{prefix}_f1', ascending=False)['file'],
                                    df.sort_values(f'{prefix}_f1', ascending=False)[f'{prefix}_f1']))
        order_by_loss = list(zip(df.sort_values(f'{prefix}_loss')['file'],
                                      df.sort_values(f'{prefix}_loss')[f'{prefix}_loss']))
        return order_by_f1, order_by_loss, order_by_prec, order_by_recall



