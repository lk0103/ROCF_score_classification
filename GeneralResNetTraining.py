import torch
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

from ROCFDataset_for_CNN import ROCFDataset


class GeneralResNetTraining():
    def __init__(self, f, img_size=500, augmentation='none', pos_embedding=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.split_dir = "./orezane_1500x1500px/Train_Val_Test_split/"
        self.img_size = img_size
        self.augmentation=augmentation
        self.pos_embedding = pos_embedding
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
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(to_rgb),
                transforms.Lambda(scale_to_minus_one_to_one)  # Scale to [-1, 1]
            ])

    def get_resnet_transforms(self, default=False):
        if default:
            return transforms.Compose([
                transforms.ToTensor()
            ])

        if self.augmentation == "crop":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.98, 1.0)  # Crop between 90% and 100% of the original size
                ),
                transforms.ToTensor()
            ])
        elif self.augmentation == "translate":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), fill=255),
                transforms.ToTensor()
            ])
        elif self.augmentation == "color":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),  # Adjust brightness and contrast
                transforms.ToTensor()
            ])
        elif self.augmentation == "rotate":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=5, fill=255),  # ±5 degrees
                transforms.ToTensor()
            ])
        elif self.augmentation == "combo":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.15),
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), fill=255),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor()
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
        train_file = os.path.join(self.split_dir, f'train_len_{str(len(X))}.txt')
        val_file = os.path.join(self.split_dir, f'val_len_{str(len(X))}.txt')
        test_file = os.path.join(self.split_dir, f'test_len_{str(len(X))}.txt')


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
            f.write("\n".join(X_train))
        with open(val_file, "w") as f:
            f.write("\n".join(X_val))
        with open(test_file, "w") as f:
            f.write("\n".join(X_test))

        if logging:
            self.logging(report="\nCreated new split files!!!\n")
        return X_test, X_train, X_val, c_test, c_train, c_val, y_test, y_train, y_val

    def load_existing_dataset_split(self, X, y, c, train_file, val_file, test_file, logging=True):
        with open(train_file) as f:
            X_train = [line.strip() for line in f]
        with open(val_file) as f:
            X_val = [line.strip() for line in f]
        with open(test_file) as f:
            X_test = [line.strip() for line in f]

        # Map file paths back to scores/classes
        lookup = {img: (score, cls) for img, score, cls in zip(X, y, c)}
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
            image_paths=X_train, scores=y_train, transform=transform, pos_embedding=self.pos_embedding
        )
        val_dataset = ROCFDataset(
            image_paths=X_val, scores=y_val, transform=val_test_transform, pos_embedding=self.pos_embedding
        )
        test_dataset = ROCFDataset(
            image_paths=X_test, scores=y_test, transform=val_test_transform, pos_embedding=self.pos_embedding
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
        print(report)
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

        plt.plot(val_accuracy_epochs, c='g', label='Validation accuracy')
        plt.title(f'{train_name} val accuracy')
        plt.legend()
        plt.savefig(f'{train_name}_val_accuracy.png')
        plt.show()

        if val_precision_epochs is not None:
            plt.plot(val_precision_epochs, c='c', label='Validation precision')
            plt.plot(val_recall_epochs, c='m', label='Validation recall')
            plt.plot(val_f1_epochs, c='g', label='Validation F1')
            plt.title(f'{train_name} val metrics')
            plt.legend()
            plt.savefig(f'{train_name}_val_metrics.png')
            plt.show()

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

    def test(self, model, dataloader, loss_fn, prefix='Test'):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        all_img_paths = []

        with torch.no_grad():
            for X, y, img_paths in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

                pred_classes = pred.argmax(1).cpu()
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
        cm = confusion_matrix(all_labels, all_preds)
        cm_str = np.array2string(cm, separator=', ')

        report = (f"{prefix} Error: \n"
                  f"{(all_preds_tensor == all_labels_tensor).sum()} correct out of {size}\n"
                  f"Avg {prefix} loss: {test_loss:>8f}\n"
                  f"Mean absolute score (MAE): {mae:.4f}\n"
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n" 
                  f"{prefix} Confusion Matrix:\n{cm_str}"
                  f"\nWrongly classified images:\n" + "\n".join(wrong_img_paths) + "\n\n")

        self.logging(report=report)

        return accuracy, test_loss, precision, recall, f1

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
        always_wrong_none = defaultdict(int)
        always_wrong_tta = defaultdict(int)

        txt_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]

        for txt_file in txt_files:
            self.extract_stats_one_model(always_wrong_none, always_wrong_tta, log_dir, results, txt_file)

        # Build DataFrame
        df = pd.DataFrame(results)

        # Build human-readable report
        report = self.create_report_stats(always_wrong_none, always_wrong_tta, df, txt_files)

        # Print via self.logging
        self.logging(report=report)

    def extract_stats_one_model(self, always_wrong_none, always_wrong_tta, log_dir, results, txt_file):
        file_path = os.path.join(log_dir, txt_file)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract LAST MODEL metrics
        last_acc, last_f1, last_loss, last_prec, last_rec = self.extract_stats_last_model_one_doc(content)

        # Extract BEST VAL MODEL metrics
        best_acc, best_f1, best_loss, best_prec, best_rec, wrong_none_imgs = self.extract_stats_best_val_model_one_doc(
            always_wrong_none, content)

        # Extract TTA results
        aug_agree_with_none, avg_agree_num, full_agree_pct, per_aug_losses, \
        tta_acc, tta_f1, tta_loss, tta_prec, tta_rec, wrong_tta_imgs = self.extract_stats_tta_one_doc(
            always_wrong_tta, content)

        results.append({
            'file': txt_file,
            'last_acc': last_acc,
            'last_prec': last_prec,
            'last_rec': last_rec,
            'last_f1': last_f1,
            'last_loss': last_loss,
            'best_acc': best_acc,
            'best_prec': best_prec,
            'best_rec': best_rec,
            'best_f1': best_f1,
            'best_loss': best_loss,
            'tta_acc': tta_acc,
            'tta_prec': tta_prec,
            'tta_rec': tta_rec,
            'tta_f1': tta_f1,
            'tta_loss': tta_loss,
            'per_aug_losses': per_aug_losses,
            'aug_agree_with_none': aug_agree_with_none,
            'avg_agree_num': avg_agree_num,
            'full_agree_pct': full_agree_pct,
            'wrong_none': wrong_none_imgs,
            'wrong_tta': wrong_tta_imgs
        })

    def extract_stats_tta_one_doc(self, always_wrong_tta, content):
        tta_match = re.search(
            r"Test TTA metrics:\s*"
            r"Augmentations:\s*(\[.*?\])\s*"
            r"(?:Average loss across all augmentations:\s*([0-9.]+)\s*)?"  # capture avg TTA loss
            r"((?:Average loss for .*?:\s*[0-9.]+\s*)*)"  # all per-augmentation losses
            r"Accuracy \(majority vote\):\s*([0-9.]+),\s*Precision:\s*([0-9.]+),\s*Recall:\s*([0-9.]+),\s*F1:\s*([0-9.]+)\s*"
            r"(?:Agreement with 'none' augmentation:\s*((?:.+?\n)+?)\n)?"
            r"(?:Avg number of augmentations agreeing with majority per image:\s*([0-9.]+)\s*\n)?"
            r"(?:Percentage of images where all augmentations agree with majority:\s*([0-9.]+)%\s*\n)?"
            r"Wrongly classified images(?: after majority vote)?:\s*\n"
            r"((?:.+\n)+)",
            content, re.DOTALL
        )
        per_aug_losses = {}
        aug_agree_with_none = {}
        tta_acc = tta_prec = tta_rec = tta_f1 = None
        tta_loss = avg_agree_num = full_agree_pct = None
        wrong_tta_imgs = []
        if tta_match:
            (aug_str, tta_loss, losses_block, tta_acc, tta_prec, tta_rec, tta_f1,
             agree_block, avg_agree_num, full_agree_pct, wrong_tta) = tta_match.groups()

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
        return aug_agree_with_none, avg_agree_num, full_agree_pct, per_aug_losses, tta_acc, tta_f1, tta_loss, tta_prec, tta_rec, wrong_tta_imgs

    def extract_stats_best_val_model_one_doc(self, always_wrong_none, content):
        best_val_match = re.search(
            r'BEST VAL MODEL Test Error:.*?Accuracy: ([0-9.]+), Precision: ([0-9.]+), Recall: ([0-9.]+), F1: ([0-9.]+)',
            content, re.DOTALL)
        if best_val_match:
            best_acc, best_prec, best_rec, best_f1 = map(float, best_val_match.groups())
        else:
            best_acc = best_prec = best_rec = best_f1 = None
        best_loss_match = re.search(
            r'BEST VAL MODEL Test Error:.*?Avg Test loss: ([0-9.]+)', content, re.DOTALL)
        best_loss = float(best_loss_match.group(1)) if best_loss_match else None
        # Extract wrongly classified images (normal testing)
        # stop at the next "Test TTA metrics" header (or EOF)
        wrong_none_match = re.search(
            r"Wrongly classified images:\s*\n"  # header
            r"((?:(?!\nTest TTA metrics).*\n)*)",  # any lines that are NOT followed by the TTA header
            content, re.MULTILINE
        )
        wrong_none_imgs = wrong_none_match.group(1).strip().splitlines() if wrong_none_match else []
        for img in wrong_none_imgs:
            always_wrong_none[img] += 1
        return best_acc, best_f1, best_loss, best_prec, best_rec, wrong_none_imgs

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

    def create_report_stats(self, always_wrong_none, always_wrong_tta, df, txt_files):
        report = "=== MODEL LOG ANALYSIS ===\n"
        report += f"Total log files: {len(txt_files)}\n\n"

        for _, row in df.iterrows():
            report += f"File: {row['file']}\n"

            report += f"  LAST MODEL -> Acc: {row['last_acc']}, Prec: {row['last_prec']}, Rec: {row['last_rec']}, F1: {row['last_f1']}, Loss: {row['last_loss']}\n"
            report += f"  BEST VAL MODEL -> Acc: {row['best_acc']}, Prec: {row['best_prec']}, Rec: {row['best_rec']}, F1: {row['best_f1']}, Loss: {row['best_loss']}\n"

            if row['tta_acc'] is not None:
                report += f"  TTA MAJORITY VOTE -> Acc: {row['tta_acc']}, Prec: {row['tta_prec']}, Rec: {row['tta_rec']}, F1: {row['tta_f1']}, Loss: {row['tta_loss']}\n"
                report += f"  Per-augmentation losses: {row['per_aug_losses']}\n"
                report += f"  Agreement with 'none': {row['aug_agree_with_none']}\n"

                if row['avg_agree_num'] is not None:
                    report += f"  Avg augmentations agreeing with majority per image: {row['avg_agree_num']}\n"

                if row['full_agree_pct'] is not None:
                    report += f"  % of images where all augmentations agree with majority: {row['full_agree_pct']}%\n"

                report += f"  Wrongly classified images TTA (majority class): {len(row['wrong_tta'])}\n"

            report += f"  Wrongly classified images (normal): {len(row['wrong_none'])}\n\n"

        report = self.stats_order_models(df, report)

        # === WRONG IMAGES ===
        report += "=== WRONG IMAGES ===\n"

        # Sort dictionaries by frequency (descending)
        sorted_wrong_none = dict(sorted(always_wrong_none.items(), key=lambda x: x[1], reverse=True))
        sorted_wrong_tta = dict(sorted(always_wrong_tta.items(), key=lambda x: x[1], reverse=True))

        report += f"Normal testing ({len(sorted_wrong_none)}): {sorted_wrong_none}\n"
        report += f"TTA testing ({len(sorted_wrong_tta)}): {sorted_wrong_tta}\n\n"

        # Print files that were always misclassified in all runs
        always_wrong_none_all = [img for img, count in sorted_wrong_none.items() if count == len(txt_files)]
        always_wrong_tta_all = [img for img, count in sorted_wrong_tta.items() if count == len(txt_files)]

        report += f"Images always wrong (normal testing) ({len(always_wrong_none_all)}): {always_wrong_none_all}\n"
        report += f"Images always wrong (TTA testing) ({len(always_wrong_tta_all)}): {always_wrong_tta_all}\n\n"

        return report

    def stats_order_models(self, df, report):
        # Ordering models (include values)
        order_by_last_recall = list(zip(df.sort_values('last_rec', ascending=False)['file'],
                                        df.sort_values('last_rec', ascending=False)['last_rec']))
        order_by_last_prec = list(zip(df.sort_values('last_prec', ascending=False)['file'],
                                      df.sort_values('last_prec', ascending=False)['last_prec']))
        order_by_last_f1 = list(zip(df.sort_values('last_f1', ascending=False)['file'],
                                    df.sort_values('last_f1', ascending=False)['last_f1']))
        order_by_last_loss = list(zip(df.sort_values('last_loss')['file'],
                                      df.sort_values('last_loss')['last_loss']))

        order_by_best_recall = list(zip(df.sort_values('best_rec', ascending=False)['file'],
                                        df.sort_values('best_rec', ascending=False)['best_rec']))
        order_by_best_prec = list(zip(df.sort_values('best_prec', ascending=False)['file'],
                                      df.sort_values('best_prec', ascending=False)['best_prec']))
        order_by_best_f1 = list(zip(df.sort_values('best_f1', ascending=False)['file'],
                                    df.sort_values('best_f1', ascending=False)['best_f1']))
        order_by_best_loss = list(zip(df.sort_values('best_loss')['file'],
                                      df.sort_values('best_loss')['best_loss']))

        order_by_tta_recall = list(zip(df.sort_values('tta_rec', ascending=False)['file'],
                                       df.sort_values('tta_rec', ascending=False)['tta_rec']))
        order_by_tta_prec = list(zip(df.sort_values('tta_prec', ascending=False)['file'],
                                     df.sort_values('tta_prec', ascending=False)['tta_prec']))
        order_by_tta_f1 = list(zip(df.sort_values('tta_f1', ascending=False)['file'],
                                   df.sort_values('tta_f1', ascending=False)['tta_f1']))
        order_by_tta_loss = list(zip(df.sort_values('tta_loss')['file'],
                                     df.sort_values('tta_loss')['tta_loss']))

        report += "=== MODEL ORDERINGS ===\n"
        report += f"By LAST MODEL Recall: {order_by_last_recall}\n"
        report += f"By LAST MODEL Precision: {order_by_last_prec}\n"
        report += f"By LAST MODEL F1: {order_by_last_f1}\n"
        report += f"By LAST MODEL Loss: {order_by_last_loss}\n\n"
        report += f"By BEST VAL MODEL Recall: {order_by_best_recall}\n"
        report += f"By BEST VAL MODEL Precision: {order_by_best_prec}\n"
        report += f"By BEST VAL MODEL F1: {order_by_best_f1}\n"
        report += f"By BEST VAL MODEL Loss: {order_by_best_loss}\n\n"
        report += f"By TTA Recall: {order_by_tta_recall}\n"
        report += f"By TTA Precision: {order_by_tta_prec}\n"
        report += f"By TTA F1: {order_by_tta_f1}\n"
        report += f"By TTA Loss: {order_by_tta_loss}\n\n"

        return report



