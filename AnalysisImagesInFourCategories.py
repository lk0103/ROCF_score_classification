import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from utils import plot_CE, plot_RE
from matplotlib import pyplot as plt
import numpy as np

from ROCFDataset_for_CNN import LoadROCFDataset

class AnalysisImagesInFourCategories():
    def __init__(self):
        self.ROCFDataset = LoadROCFDataset()

    def count_pixels_binary_images(self, f, binarization_type='morphology'):
        class_num_pixel = {0: [], 1: [], 2: [], 3: []}
        for i in range(len(self.ROCFDataset)):
            img_name = self.ROCFDataset.get_image_path_all_files(i)

            if binarization_type == 'morphology':
                bin_img = self.ROCFDataset.preprocess_image_morphology(img_name, visualize=False)
            else:
                bin_img = self.ROCFDataset.preprocess_image_adaptive_threshold(img_name)

            name, order, score = self.ROCFDataset.extract_from_name(img_name)

            score_class = self.ROCFDataset.class_from_score(score)

            bin_img_flatten = bin_img.flatten() / 255
            number_pixels = np.sum(bin_img_flatten)
            class_num_pixel[score_class].append(number_pixels)

            info_line = "{}. : {}, score: {}, score_class: {}, number of pixels: {}".format(i, name, score, score_class, number_pixels)
            print(info_line)
            f.write(info_line + '\n')

        self.statistics_for_binary_4_score_classes(class_num_pixel, f)

    def statistics_for_binary_4_score_classes(self, class_num_pixel, f):
        for score_class in range(4):
            pixel_counts = np.array(class_num_pixel[score_class])
            avg = np.mean(pixel_counts)
            median = np.median(pixel_counts)
            min = np.min(pixel_counts)
            max = np.max(pixel_counts)
            std = np.std(pixel_counts)
            quantile10 = np.quantile(pixel_counts, 0.1)
            quantile25 = np.quantile(pixel_counts, 0.25)
            quantile50 = np.quantile(pixel_counts, 0.5)
            quantile75 = np.quantile(pixel_counts, 0.75)
            quantile90 = np.quantile(pixel_counts, 0.9)


            info_line = f'\nscore class {score_class} analysis: \naverage: {avg}\nmedian: {median}\nmin: {min}\nmax: {max}' \
                        f'\nstd: {std}\nquantile10: {quantile10}\nquantile25: {quantile25}\nquantile50: {quantile50}\n' \
                        f'quantile75: {quantile75}\nquantile90: {quantile90}\n'
            print(info_line)
            f.write(info_line + '\n')

    def analyse_pixel_intensities_greyscale(self, f):
        class_mean_intensity = {0: [], 1: [], 2: [], 3: []}

        for i in range(len(self.ROCFDataset)):
            img_name = self.ROCFDataset.get_image_path_all_files(i)

            grey_img = self.ROCFDataset.preprocess_image_greyscale(img_name)

            name, order, score = self.ROCFDataset.extract_from_name(img_name)

            score_class = self.ROCFDataset.class_from_score(score)

            grey_filtered = grey_img[grey_img <= 240]
            average_intensity = np.sum(grey_filtered) / (grey_img.shape[0] * grey_img.shape[1])
            class_mean_intensity[score_class].append(average_intensity)

            info_line = f"{i}. : {name}, score: {score}, score_class: {score_class}, " \
                        f"average intensity: {average_intensity}"
            print(info_line)
            f.write(info_line + '\n')

        info_line = f'\nstatistics of average intensities - filtering out values bigger that 240 and average over all the pixels (500 x 500):'
        print(info_line)
        f.write(info_line + '\n')
        self.statistics_for_binary_4_score_classes(class_mean_intensity, f)

    def analysis_four_categories(self):
        with open(f'analysis_pixels_morphol_bin_4_classes.txt', 'w') as f:
            self.count_pixels_binary_images(f, 'morphology')

        with open(f'analysis_pixels_adaptive_bin_4_classes.txt', 'w') as f:
            self.count_pixels_binary_images(f, 'adaptive')

        with open(f'analysis_pixel_intensities_greyscale_4_classes.txt', 'w') as f:
            self.analyse_pixel_intensities_greyscale(f)

AnalysisImagesInFourCategories().analysis_four_categories()

