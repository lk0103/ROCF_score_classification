# Automated ROCF Assessment System

An AI‑powered tool for automating the scoring of the Rey–Osterrieth Complex Figure (ROCF) neuropsychological test using computer vision and machine learning techniques. 

More information on web page dedicated to this Master's thesis: https://davinci.fmph.uniba.sk/~korbelova19/masters_thesis/masters_thesis_ROCF.html 


## 📋 Project Overview

The goal of this project is to develop an end-to-end automated system for evaluating performance on the Rey–Osterrieth Complex Figure test. Leveraging state-of-the-art computer vision and artificial intelligence methods. The system currently:

- **Preprocesses input images** to enhance the drawing and remove noise.  
- **Trains Multiple Image Classifiers** that classify each drawing into one of four categories (Class 0: least similar; Class 1: moderately similar; Class 2: similar; Class 3: very similar) through comprehensive hyperparameter testing of both ResNet‑18 and Swin Transformer architectures.

This approach aims to provide clinicians and researchers with a faster, more consistent, and scalable way to score ROCF tests, ultimately enhancing the diagnosis and monitoring of neurological and psychiatric memory disorders.

## Description of the scripts
+ ***ROCFDataset_for_CNN.py***:
Loads raw images, handles preprocessing steps (noise reduction, resizing, normalization), and includes common image-related functionality used across the project.

+ ***Resnet18_experiments.py***:
Implements training pipelines for ResNet‑18 on the ROCF dataset. Supports hyperparameter search, saves model checkpoints and result summaries, and generates heatmap visualizations that show the impact of occluding each square patch (replacing it with a black square) on the softmax output for the correct class.

+ ***Swin_experiments.py***: 
Similar to Resnet18_experiments.py, but uses a Swin Transformer backbone. Includes hyperparameter search, checkpoint saving, and performance logging.

+ ***Swin_experiments_test.py***: 
A streamlined script to load a pre‑trained Swin transformer model and evaluate it on the test set, producing accuracy metrics.

+ ***orezane_1500x1500px/***: 
Contains dataset images, all resized and cropped to 1500×1500 pixels.

+ ***Results of experiments***: 
Stores output from experiments, including:

  - Model training logs, checkpoint files and graphs of validation accuracy and loss during training for different hyperparameter configurations (/old_experiments, /resnet18_4outputs, /swin_transformer, /swin-trans-not-learning).

  - Heatmap visualizations highlighting which image regions contribute most to correct classifications (/heatmap_visualization, /heatmap_results).

  - Statistical analysis outputs (e.g., proportion of pixels containing drawn elements per class, average grayscale pixel intensity distributions) - /pixel_count_analysis.

+ ***Outline of the Master's thesis***:
Stores LaTeX structure of the Master's thesis (ROCF_master_thesis_latex.zip), its PDF version (ROCF_master_thesis.pdf) and presentation of current progress (ROCF_classification_PS1_Korbelova_final.pptx)

