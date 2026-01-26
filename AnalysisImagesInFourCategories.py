import os
import shutil
import numpy as np

import matplotlib.pyplot as plt
from collections import Counter

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

    def analyse_score_frequency(self, f):
        score_counts = Counter()
        order_counts = Counter()
        unique_names = set()

        order_names = {1: 'copy', 2: 'immediate recall', 3: 'delayed recall'}

        # Iterate through the dataset
        for i in range(len(self.ROCFDataset)):
            img_name = self.ROCFDataset.get_image_path_all_files(i)
            name, order, score = self.ROCFDataset.extract_from_name(img_name)

            order = order_names[order]
            score_counts[score] += 1
            order_counts[order] += 1
            unique_names.add(name)

        # Compute statistics
        num_unique_names = len(unique_names)
        total_images = len(self.ROCFDataset)

        # Prepare readable statistics
        stats_text = []
        stats_text.append("===== ROCFDataset Analysis =====\n")
        stats_text.append(f"Total images: {total_images}\n")
        stats_text.append(f"Unique names: {num_unique_names}\n")
        stats_text.append("\nScore frequencies:\n")

        # Sorted score frequencies
        stats_text.append("\nScore frequencies (sorted):\n")
        for k in sorted(score_counts.keys()):
            stats_text.append(f"  Score {k}: {score_counts[k]}\n")

        # Compute and log score percentage distributions
        score_list = list(score_counts.elements())

        def percentage_in_range(low, high):
            count = sum(1 for s in score_list if low < s <= high)
            return 100 * count / total_images if total_images > 0 else 0

        coarse_bins = [(0, 10), (10, 20), (20, 30), (30, 36)]
        fine_bins = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36)]

        stats_text.append("\nScore percentage distribution (coarse bins):\n")
        for low, high in coarse_bins:
            pct = percentage_in_range(low, high)
            stats_text.append(f"  {low:>2}-{high:<2}: {pct:.2f}%\n")

        stats_text.append("\nScore percentage distribution (fine bins):\n")
        for low, high in fine_bins:
            pct = percentage_in_range(low, high)
            stats_text.append(f"  {low:>2}-{high:<2}: {pct:.2f}%\n")


        # Sorted order frequencies
        stats_text.append("\nOrder frequencies (sorted):\n")
        for k in sorted(order_counts.keys()):
            stats_text.append(f"  Order {k}: {order_counts[k]}\n")

        stats_text_str = "".join(stats_text)

        # Print statistics to console
        print(stats_text_str)

        # Save statistics to the file in readable form
        f.write(stats_text_str)

        # Plot histogram for score frequencies
        plt.figure()
        plt.bar(score_counts.keys(), score_counts.values(), width=0.4)
        plt.title("Score Frequency")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("score_histogram.png")
        plt.close()

        # Plot histogram for order frequencies
        plt.figure()
        plt.bar(order_counts.keys(), order_counts.values())
        plt.title("Drawing Phase Frequency")
        plt.xlabel("Order")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("order_histogram.png")
        plt.close()

        print("\nHistograms saved as 'score_histogram.png' and 'order_histogram.png'.")
        print(f"Statistics saved in '{f}'.")

    def compute_union_intersection_and_histograms(
        self,
        img_dict: dict,
        prefix: str,
        output_root: str = "./wrong_images_analysis"
    ):
        """
        Computes union and intersection of image paths, copies images,
        extracts scores and classes, and creates histograms.

        img_dict: Dict[str, List[str]]
        prefix: prefix for output folders and histogram names
        output_root: root output directory
        """

        os.makedirs(output_root, exist_ok=True)

        # 1. Compute union & intersection
        lists = list(img_dict.values())

        union_paths = set().union(*lists)
        intersection_paths = (
            set(lists[0]).intersection(*lists[1:])
            if len(lists) > 1 else set(lists[0])
        )

        union_dict = {prefix: sorted(union_paths)}
        intersection_dict = {prefix: sorted(intersection_paths)}

        # 2. Prepare directories & copy images
        union_dir = os.path.join(output_root, f"{prefix}_union")
        intersection_dir = os.path.join(output_root, f"{prefix}_intersection")

        self._copy_images(union_paths, union_dir)
        self._copy_images(intersection_paths, intersection_dir)

        # 3. Extract scores & classes
        union_scores, union_classes = self._extract_scores_and_classes(union_paths)
        intersection_scores, intersection_classes = self._extract_scores_and_classes(intersection_paths)

        # 4. Histograms
        self._plot_and_save_histogram(
            data=union_scores,
            bins=np.arange(0, 38) - 0.5,
            xticks=range(0, 37, 2),
            xlabel="Score",
            title=f"{prefix} – Score (Union)",
            save_dir=union_dir,
            filename=f"{prefix}_union_scores_hist.png"
        )

        self._plot_and_save_histogram(
            data=intersection_scores,
            bins=np.arange(0, 38) - 0.5,
            xticks=range(0, 37, 2),
            xlabel="Score",
            title=f"{prefix} – Score (Intersection)",
            save_dir=intersection_dir,
            filename=f"{prefix}_intersection_scores_hist.png"
        )

        self._plot_and_save_histogram(
            data=union_classes,
            bins=np.arange(-0.5, 4.5),
            xticks=range(0, 4),
            xlabel="Class",
            title=f"{prefix} – Class Histogram (Union)",
            save_dir=union_dir,
            filename=f"{prefix}_union_classes_hist.png"
        )

        self._plot_and_save_histogram(
            data=intersection_classes,
            bins=np.arange(-0.5, 4.5),
            xticks=range(0, 4),
            xlabel="Class",
            title=f"{prefix} – Class Histogram (Intersection)",
            save_dir=intersection_dir,
            filename=f"{prefix}_intersection_classes_hist.png"
        )

        return intersection_dict, union_dict


    def _copy_images(self, image_paths, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for path in image_paths:
            if os.path.exists(path):
                shutil.copy(path, target_dir)

    def _extract_scores_and_classes(self, image_paths):
        scores = []
        classes = []

        for img_name in image_paths:
            _, _, score = self.ROCFDataset.extract_from_name(img_name)
            score_class = self.ROCFDataset.class_from_score(score)

            scores.append(score)
            classes.append(score_class)

        return scores, classes

    def _plot_and_save_histogram( self, data, bins, xticks, xlabel,
        title, save_dir, filename
    ):
        plt.figure(figsize=(7, 5))
        plt.hist(data, bins=bins, edgecolor="black")
        plt.xticks(xticks)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.show()


    def analysis_four_categories(self):
        with open(f'analysis_pixels_morphol_bin_4_classes.txt', 'w') as f:
            self.count_pixels_binary_images(f, 'morphology')

        with open(f'analysis_pixels_adaptive_bin_4_classes.txt', 'w') as f:
            self.count_pixels_binary_images(f, 'adaptive')

        with open(f'analysis_pixel_intensities_greyscale_4_classes.txt', 'w') as f:
            self.analyse_pixel_intensities_greyscale(f)

        with open(f'analysis_frequency_scores.txt', 'w') as f:
            self.analyse_score_frequency(f)

# AnalysisImagesInFourCategories().analysis_four_categories()

AnalysisImagesInFourCategories().compute_union_intersection_and_histograms(
    img_dict={
        # 'always wrong (normal best val model testing': ['./orezane_1500x1500px/Klinicka skupina/CM2017SK040_2_22,5.jpg',', './orezane_1500x1500px/Kontrolna skupina/FF2017PB002_3_31.jpg', './orezane_1500x1500px/Kontrolna skupina/FF2018ZD032_3_23.jpg', './orezane_1500x1500px/Klinicka skupina/MCI2018MA019_2_17,5.jpg', './orezane_1500x1500px/Kontrolna skupina/FF2018MA007_1_28,5.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB067_1_28.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB076_3_16.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB051_2_15.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017SK010_1_30.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB046_2_23.jpg', './orezane_1500x1500px/Kontrolna skupina/PE2017ZM13_1_30.jpg', './orezane_1500x1500px/Klinicka skupina/MCI2018MA019_3_20.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB087_2_15,5.jpg'],
        # 'always wrong (normal last model testing)': ['./orezane_1500x1500px/Klinicka skupina/CM2017SK040_2_22,5.jpg', './orezane_1500x1500px/Kontrolna skupina/FF2018ZD032_3_23.jpg', './orezane_1500x1500px/Klinicka skupina/MCI2018MA019_2_17,5.jpg', './orezane_1500x1500px/Kontrolna skupina/FF2018MA007_1_28,5.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB067_1_28.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB076_3_16.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB051_2_15.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017SK010_1_30.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB046_2_23.jpg', './orezane_1500x1500px/Kontrolna skupina/PE2017ZM13_1_30.jpg', './orezane_1500x1500px/Klinicka skupina/MCI2018MA019_3_20.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB087_2_15,5.jpg'],
        # 'always wrong swin (TTA testing best val model)': ['./orezane_1500x1500px/Klinicka skupina/CM2017SK040_2_22,5.jpg', './orezane_1500x1500px/Kontrolna skupina/FF2017DM009_2_13.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2016PB002_3_15,5.jpg', './orezane_1500x1500px/Kontrolna skupina/FF2017PB002_3_31.jpg', './orezane_1500x1500px/Klinicka skupina/SM2017PB018_3_16.jpg', './orezane_1500x1500px/Kontrolna skupina/FF2018MA007_1_28,5.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB076_3_16.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB051_2_15.jpg', './orezane_1500x1500px/Kontrolna skupina/FF2017ES009_2_30,5.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017SK010_1_30.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB046_2_23.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2017PB087_2_15,5.jpg', './orezane_1500x1500px/Kontrolna skupina/CM2018PB008_1_12.jpg'],
        'GPT 5.2 thinking - wrong images- few shot ': [
'./orezane_1500x1500px/Kontrolna skupina/FF2017AS002_2_22.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM17SG02_3_12,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017ES03_1_33.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017PB041_3_8,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK046_2_5,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/PE2017PB002_2_10,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017VM09_3_12,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017DM009_2_13.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK044_3_19,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017ZP007_1_30,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK001_2_14,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/PE2017ZM13_3_11,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017PB002_3_31.jpg',
'./orezane_1500x1500px/Kontrolna skupina/PE2017PB007_1_36.jpg',
'./orezane_1500x1500px/Klinicka skupina/SM2017PB018_3_16.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF17SG002_3_18.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2018ZD032_3_23.jpg',
'./orezane_1500x1500px/Klinicka skupina/MCI2018MA019_2_17,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017PB001_3_9,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK023_2_12,5.jpg',
'./orezane_1500x1500px/Klinicka skupina/SM2017PB025_3_22.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2018PB018_1_35.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK053_1_31.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2018PB019_3_14.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF17ZD013_3_27.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2016AHS03_3_15,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2018PB005_2_14,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/PE2016PB004_2_17,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK007_1_36.jpg',
'./orezane_1500x1500px/Klinicka skupina/SM2017PB011_1_33.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK057_2_8,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2016SK001_2_22.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SG003_3_10.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK048_2_10,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK014_1_31.jpg',
'./orezane_1500x1500px/Kontrolna skupina/PE2017PB007_3_22.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK010_1_30.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2016MH006_3_34.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017ES005_2_20,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017PB046_2_23.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017SK001_1_31.jpg',
'./orezane_1500x1500px/Klinicka skupina/MCI2018MA019_3_20.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2016PB007_2_22.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF17ZD011_1_36.jpg',
'./orezane_1500x1500px/Klinicka skupina/SM2017PB025_1_32.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017PB076_1_33.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017VM07_3_26,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017AS001_3_23.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017PB086_1_35.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2018PB007_1_34.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017PB067_2_11,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2018PB013_3_14,5.jpg',
'./orezane_1500x1500px/Kontrolna skupina/PE2016SK004_1_36.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2018ZD034_1_34.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2018PB022_1_32.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2017PB055_1_34.jpg',
'./orezane_1500x1500px/Kontrolna skupina/CM2018PB008_1_12.jpg',
'./orezane_1500x1500px/Kontrolna skupina/FF2017ES007_2_12,5.jpg'
]
    }, prefix='LLM_few_shot_class'
)

