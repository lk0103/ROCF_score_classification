# Automated ROCF Assessment System

An AI‑powered tool for automating the scoring of the Rey–Osterrieth Complex Figure (ROCF) neuropsychological test using computer vision and machine learning techniques. 

More information on web page dedicated to this Master's thesis: https://davinci.fmph.uniba.sk/~korbelova19/masters_thesis/masters_thesis_ROCF.html 


## 📋 Project Overview

The goal of this project is to develop an end-to-end automated system for evaluating performance on the Rey–Osterrieth Complex Figure test. Leveraging state-of-the-art computer vision and artificial intelligence methods. The system has the following functionality:
 
- **Training of multiple image classifiers**: models classify each drawing into one of four categories (Class 0: least similar, score 0-14.5; Class 1: moderately similar, score 15-22; Class 2: similar, score 22.5-30; Class 3: very similar, score 30.5-36) or one of 6 categories (Class 0: score 0-6; Class 1: score 6-12; Class 2: score 12-18; Class 3: score 18-24; Class 4: score 24-30; Class 3: score 30-36) through comprehensive hyperparameter testing of both ResNet‑18 and Swin Transformer architectures.
- **Augmentation of training images**: one of the hyperparameters selects the augmentation applied to images in the training dataset. Six augmentation modes are available::
    - No augmentation
    - Random crop
    - Random translation
    - Random rotation
    - Random color brightness and contrast
    - A combination of random translation, rotation, and color brightness/contrast
- **Ability to add image preprocessing in the future**: the system is designed to easily incorporate improved preprocessing methods later.
- **Evaluation of results on validation or test set**: comprehensive evaluation of model performance, including multiple metrics. The system computes loss, accuracy, recall, precision, F1 score, the confusion matrix, MAE, and records all misclassified images on the validation and test sets. On the test set, it also evaluates performance in a test-time augmentation mode: for each image, five versions are sent to the model—one unmodified, one rotated, one translated, one with altered brightness/contrast, and one combining all three augmentations. The final prediction is the majority vote across the five outputs.

This approach aims to provide clinicians and researchers with a faster, more consistent, and scalable way to score ROCF tests, ultimately enhancing the diagnosis and monitoring of neurological and psychiatric memory disorders.

## Description of the scripts
+ ***ROCFDataset_for_CNN.py***:
Loads raw images, handles preprocessing steps (noise reduction, resizing, normalization), and includes common image-related functionality used across the project.

+ ***Resnet18_experiments.py***:
Implements training pipelines for ResNet-18 on the ROCF dataset, supports hyperparameter search, saves model checkpoints and result summaries with graphs, and uses Grad-CAM to generate the most significant image regions that influence the final prediction.

+ ***Swin_experiments.py***: 
Similar to Resnet18_experiments.py, but uses a Swin Transformer backbone. Includes hyperparameter search, checkpoint saving, and performance logging.

+ ***Swin_experiments_test.py***: 
A streamlined script to load a pre‑trained Swin transformer model and evaluate it on the test set, producing performance metrics.

+ ***AnalysisImagesInFourCategories.py***:
Computes dataset and class-level statistics, including a histogram of image counts by score and the average number of drawing pixels in the four classes.

+ ***GeneralResNetTraining.py***:
Contains all shared training and evaluation code used for both the ResNet-18 and Swin Transformer models.

+ ***orezane_1500x1500px/***: 
Contains dataset images, all resized and cropped to 1500×1500 pixels.

+ ***Directories with results***: 
Stores output from experiments, including:

  - Model training logs, checkpoint files, and graphs of validation accuracy, recall, precision, F1 score, and loss for different hyperparameter configurations (/old_experiments, /resnet_new_results, /transformer_new_resutls).

  - Heatmap visualizations highlighting which image regions contribute most to correct classifications (/heatmap_visualization, /heatmap_results).
 
  - Grad-CAM visualizations highlighting the regions most influential for the final classification (/grad-cam_results).

  - Statistical analysis outputs (e.g., proportion of pixels containing drawn elements per class, average grayscale intensity distributions) (/pixel_count_analysis).
 
  - Dataset-level statistics (e.g., histogram of image counts by score and phase of the ROCF test) (/dataset_stats).

+ ***Outline of the Master's thesis***:
Stores LaTeX structure of the Master's thesis (ROCF_master_thesis_latex.zip), its PDF version (ROCF_master_thesis.pdf) and presentation of current progress (ROCF_classification_PS1_Korbelova_final.pptx, ROCF_classification_PS2_Korbelova.pptx)

## Results

This project compares several deep learning models (ResNet-18, Swin Transformer, and a baseline CNN) for ROCF image classification using fixed train/validation/test splits. All models are evaluated with recall, precision, F1 score, MAE, and confusion matrices. Test-time augmentation (five augmented variants per image with majority vote) is used to improve prediction stability.

Both architectures are first trained without augmentation, then with multiple augmentation strategies. ResNet-18 consistently performs best, with the strongest results achieved using a combined augmentation consisting of brightness/contrast changes, translation, and rotation. This optimal configuration will be then used for a six-class classification task, along with further experiments involving preprocessing and an encoder–decoder feature-extraction approach.

### Models trained without augmentation (4 classes)

| TTA Metric | Ivanyi | ResNet-18 | Swin Transformer |
| ---------- | ------ | --------- | ---------------- |
| recall     | 72.5%  | 71.43%    | 68.57%           |
| precision  | —      | 72.14%    | 68.61%           |
| F1 score   | —      | 70.51%    | 68.25%           |
| loss       | —      | 0.9951    | 1.0253           |

### ResNet-18 — Augmentation comparison (4 classes)

| TTA Metric | No     | contrast brightness | translation | rotation | crop   | contrast, brightness translation, rotation |
| ---------- | ------ | ------------------- | ----------- | -------- | ------ | ------------------------------------------ |
| recall     | 71.43% | 72.38%              | 73.33%      | 73.33%   | 50.48% | **77.14%**                                 |
| precision  | 72.14% | 72.69%              | 73.51%      | 73.37%   | 53.43% | **77.61%**                                 |
| F1 score   | 70.51% | 71.58%              | 73.17%      | 72.77%   | 42.88% | **76.94%**                                 |
| loss       | 0.9951 | 0.8071              | 0.9922      | 0.9214   | 1.0312 | **0.6135**                                 |

In the following images, the training curves for the best-performing ResNet-18 model using combaned augmentation are shown. They display the loss as well as F1 score, recall, and precision on the validation set throughout training, illustrating model convergence and comparative performance across epochs.
<img width="320" height="240" alt="resnet18_pretrain_lr0_0001_sts3_15e_is400_GP_combo_val_metrics" src="https://github.com/user-attachments/assets/8a77b52e-02cc-4c45-a563-71cbea8efa7d" />
<img width="320" height="240" alt="resnet18_pretrain_lr0_0001_sts3_15e_is400_GP_combo_loss" src="https://github.com/user-attachments/assets/2be7d54a-6933-42d3-8a7b-c312c1b7f592" />

### Swin Transformer — Augmentation comparison (4 classes)

| TTA Metric | No     | contrast brightness | translation | rotation | crop   | contrast, brightness translation, rotation |
| ---------- | ------ | ------------------- | ----------- | -------- | ------ | ------------------------------------------ |
| recall     | 68.57% | 64.76%              | 66.67%      | 73.33%   | 24.76% | 69.52%                                     |
| precision  | 68.61% | 65.41%              | 66.71%      | 73.29%   | 6.13%  | 69.08%                                     |
| F1 score   | 68.25% | 64.98%              | 66.57%      | 72.23%   | 9.83%  | 68.85%                                     |
| loss       | 1.0253 | 0.987               | 0.7173      | 0.7681   | 2.4093 | 0.7053                                     |

In the following images, the training curves for the best-performing Swin Transformer model using rotation augmentation are shown. They display the loss as well as F1 score, recall, and precision on the validation set throughout training, illustrating model convergence and comparative performance across epochs.
<img width="320" height="240" alt="swin_transformer_notEmb_rotate_loss" src="https://github.com/user-attachments/assets/8749abe7-47a3-4849-9b04-3d76aebb5bae" />
<img width="320" height="240" alt="swin_transformer_notEmb_rotate_val_metrics" src="https://github.com/user-attachments/assets/2e76712b-1cbd-4451-9f71-46d27e6c93de" />


## Evaluation results demonstration
 
  Examples of graphs tracking validation performance metrics during training for one of the trained ResNet-18 models:

<table>
  <tr>
    <td><img width="320" height="240" alt="resnet18_pretrain_lr0 0001_sts4_25e_is500_GP_combo_loss" src="https://github.com/user-attachments/assets/88c839a6-1511-48c2-9d50-b388a6c847b5" /></td>
    <td><img width="320" height="240" alt="resnet18_pretrain_lr0 0001_sts4_25e_is500_GP_combo_val_accuracy" src="https://github.com/user-attachments/assets/b3750c6f-c195-4589-bdfd-73f29afa1a8d" /> </td>
    <td><img width="320" height="240" alt="resnet18_pretrain_lr0 0001_sts4_25e_is500_GP_combo_val_metrics" src="https://github.com/user-attachments/assets/c7cba4bf-1153-47d4-a5b1-8fec4cbb7a4f" /> </td>
  </tr>
</table>

Example of validation results during training for one of the ResNet-18 models: 

```
loss: 0.843152  [    0/  842]
loss: 0.726993  [  160/  842]
loss: 0.644688  [  320/  842]
loss: 0.826014  [  480/  842]
loss: 0.992870  [  640/  842]
loss: 0.969303  [  800/  842]
Epoch [2/25], Average epoch loss: 0.7943
Val Error: 
66 correct out of 94
Avg Val loss: 0.740291
Mean absolute score (MAE): 0.2979
Accuracy: 0.7021, Precision: 0.7793, Recall: 0.7021, F1: 0.7041
Val Confusion Matrix:
[[12, 11,  0,  0],
 [ 0, 18,  5,  0],
 [ 0,  2, 21,  1],
 [ 0,  0,  9, 15]]
Wrongly classified images:
./orezane_1500x1500px/Kontrolna skupina//FF2018MA017_1_34.jpg
./orezane_1500x1500px/Kontrolna skupina//CM2018PB022_2_11,5.jpg
./orezane_1500x1500px/Klinicka skupina//SM2017PB014_3_12,5.jpg
...


New best model saved with F1=0.7041!!

loss: 0.518577  [    0/  842]
loss: 0.579501  [  160/  842]
loss: 0.446504  [  320/  842]
loss: 0.383556  [  480/  842]
loss: 0.500368  [  640/  842]
loss: 0.585142  [  800/  842]
Epoch [3/25], Average epoch loss: 0.6309
Val Error: 
69 correct out of 94
Avg Val loss: 0.627315
Mean absolute score (MAE): 0.2766
Accuracy: 0.7340, Precision: 0.7672, Recall: 0.7340, F1: 0.7393
Val Confusion Matrix:
[[15,  7,  1,  0],
 [ 0, 16,  7,  0],
 [ 0,  2, 18,  4],
 [ 0,  0,  4, 20]]
Wrongly classified images:
./orezane_1500x1500px/Kontrolna skupina//FF2018MA001_1_32.jpg
./orezane_1500x1500px/Kontrolna skupina//FF2017PB003_2_30.jpg
...


New best model saved with F1=0.7393!!

loss: 0.322925  [    0/  842]
loss: 0.604705  [  160/  842]
loss: 0.664534  [  320/  842]
loss: 0.753824  [  480/  842]
loss: 0.608738  [  640/  842]
loss: 0.752622  [  800/  842]
Epoch [4/25], Average epoch loss: 0.5254
Val Error: 
74 correct out of 94
Avg Val loss: 0.558598
Mean absolute score (MAE): 0.2128
Accuracy: 0.7872, Precision: 0.7902, Recall: 0.7872, F1: 0.7880
Val Confusion Matrix:
[[18,  5,  0,  0],
 [ 3, 18,  2,  0],
 [ 0,  2, 18,  4],
 [ 0,  0,  4, 20]]
Wrongly classified images:
./orezane_1500x1500px/Klinicka skupina//SM2017PB015_3_13,5.jpg
./orezane_1500x1500px/Kontrolna skupina//CM2017SK023_1_30.jpg
...
```

Example of the final model (model from epoch with the highest F1 score on the validation set during training) testing results logged for one of the ResNet-18 models:: 

```
BEST VAL MODEL Test Error: 
81 correct out of 105
Avg BEST VAL MODEL Test loss: 0.628914
Mean absolute score (MAE): 0.2381
Accuracy: 0.7714, Precision: 0.7718, Recall: 0.7714, F1: 0.7707
BEST VAL MODEL Test Confusion Matrix:
[[22,  4,  0,  0],
 [ 4, 20,  2,  0],
 [ 0,  3, 18,  6],
 [ 0,  1,  4, 21]]
Wrongly classified images:
./orezane_1500x1500px/Klinicka skupina//MCI2017MA007_1_30.jpg
./orezane_1500x1500px/Klinicka skupina//CM2017SK040_2_22,5.jpg
....


Test TTA metrics:
Augmentations: ['none', 'rotate', 'color', 'translate', 'combo']
Average loss across all augmentations: 0.6245
Average loss for none: 0.6289
Average loss for rotate: 0.6657
Average loss for color: 0.6318
Average loss for translate: 0.6187
Average loss for combo: 0.5775
Accuracy (majority vote): 0.7619, Precision: 0.7612, Recall: 0.7619, F1: 0.7611

Agreement with 'none' augmentation:
rotate: 91.43% images agree with 'none'
color: 95.24% images agree with 'none'
translate: 94.29% images agree with 'none'
combo: 90.48% images agree with 'none'

Avg number of augmentations agreeing with majority per image: 4.77
Percentage of images where all augmentations agree with majority: 82.86%
Wrongly classified images after majority vote:
./orezane_1500x1500px/Klinicka skupina//MCI2017MA007_1_30.jpg
./orezane_1500x1500px/Klinicka skupina//CM2017SK040_2_22,5.jpg
./orezane_1500x1500px/Kontrolna skupina//PE2017PB002_2_10,5.jpg
./orezane_1500x1500px/Kontrolna skupina//FF2017VM09_3_12,5.jpg
....
```
