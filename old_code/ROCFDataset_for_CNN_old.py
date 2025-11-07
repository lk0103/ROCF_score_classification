import os
import cv2 as cv
from matplotlib import pyplot as plt


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.feature import hog
import torch.nn.functional as F
import pandas as pd


class LoadROCFDataset(Dataset):
    def __init__(self, img_size=500, normalize=False):
        self.directory_name_clinical = "./orezane_1500x1500px/Klinicka skupina/"
        self.file_names_clinical = self.list_files(self.directory_name_clinical)

        self.directory_name_control = "./orezane_1500x1500px/Kontrolna skupina/"
        self.file_names_control = self.list_files(self.directory_name_control)

        self.all_file_names = self.file_names_control + self.file_names_clinical

        self.partial_scores_filename = "./orezane_1500x1500px/Neuropsy_data_Reyovky.xlsx"

        #self.num_score_classes = 73 # if we predict score
        self.num_score_classes = 4 # if we predict classes
        self.img_size = img_size
        self.normalize = normalize


    def __len__(self):
        return len(self.all_file_names)


    def __getitem__(self, idx):
        img_path = self.get_image_path_all_files(idx)
        preprocessed = self.preprocess_image(img_path)
        name, order, score = self.extract_from_name(img_path)
        return torch.tensor(preprocessed, dtype=torch.float64), torch.tensor(score, dtype=torch.float32)


    def list_files(self, directory):
        files = []
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                files.append(directory + '/' + filename)
        return files


    def show_image(self, img):
        cv.imshow("Display window", img)
        k = cv.waitKey(0)


    def plot_image(self, img):
        plt.imshow(img, cmap='gray'), plt.show()
        k = cv.waitKey(0)


    def extract_from_name(self, img_name):
        #CM17SG02_1_32.jpg
        end_directory_path_index = img_name.rfind('/')

        if end_directory_path_index != -1:
            img_name = img_name[end_directory_path_index + 1:]

        img_name = os.path.basename(img_name)
        img_name = img_name.replace('.jpg', '')
        img_name = img_name.replace('.png', '')
        name, order, score = img_name.split('_')
        order = int(order)
        score = float(score.replace(',', '.'))
        return name, order, score

    def get_partial_scores_images(self):
        data = pd.read_excel(self.partial_scores_filename)
        all_filenames = [os.path.splitext(os.path.basename(path))[0] for path in self.all_file_names]
        print(len(all_filenames))

        # Function to extract the desired dictionary structure from each row
        def extract_dict_from_row(row):
            return {
                f'{row["ID"]}_1_{row["SPOLUK"]}'.replace('.', ',').upper(): row[2:20].tolist(),
                f'{row["ID"]}_2_{row["SPOLUR3"]}'.replace('.', ',').upper(): row[20:38].tolist(),
                f'{row["ID"]}_3_{row["SPOLUR30"]}'.replace('.', ',').upper(): row[38:56].tolist()
            }

        # Create a dictionary from the data
        extracted_data = {}
        for _, row in data.iterrows():
            for k, values in extract_dict_from_row(row).items():
                values = [0.5 if value == 5 else value for value in values]

                allowed_numbers = {0, 1, 0.5, 2}
                is_valid = all(v in allowed_numbers for v in values)

                if k not in all_filenames:
                    print(k)
                if is_valid and k in all_filenames:
                    extracted_data[k] = values

        # Display a sample of the extracted data
        print(len(extracted_data))
        print(list(extracted_data.items())[:5])

    def get_image_path_all_files(self, i):
        return  self.all_file_names[i]


    def preprocess_image(self, img_name):

        gray = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        resized = cv.resize(gray, (self.img_size, self.img_size), interpolation=cv.INTER_AREA)
        blurred = cv.GaussianBlur(resized, (3, 3), 0)
        blurred = cv.fastNlMeansDenoising(blurred, None, 1.0, 7, 21) #1.25 - trosku menej sumu v pozadi ale
                                                                    #viac z ciar zachovane

        img = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv.THRESH_BINARY_INV, 11, 2)

        #Use dilation followed by erosion (opening) to enhance strokes
        kernel = np.ones((2, 2), np.uint8)
        img = cv.dilate(img, kernel, iterations=1)
        img = cv.erode(img, kernel, iterations=1)

        self.plot_comparing_original_preprocessed(gray, img)

        return img

    def preprocess_image_morphology(self, img_name):

        gray = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        resized = cv.resize(gray, (self.img_size, self.img_size), interpolation=cv.INTER_AREA)
        blurred = cv.GaussianBlur(resized, (3, 3), 0)
        blurred = cv.fastNlMeansDenoising(blurred, None, 1.0, 7, 21)
        img = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv.THRESH_BINARY_INV, 11, 2)

        reconstructed_image = img.copy()

        kernel_er = np.ones((2, 2), dtype=np.uint8)
        kernel_dil_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        kernel_dil_diag_1 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
        kernel_dil_diag_2 = np.array([[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)
        kernel_dil_vertical = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint8)
        kernel_dil_horizontal = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.uint8)

        for i in range(1):
            # Apply Erosion to remove background noise
            eroded_image = cv.erode(reconstructed_image, kernel_er, iterations=1)

            # Apply Dilation to regrow the lines lost during erosion
            dilated_image = cv.dilate(eroded_image, kernel_dil_cross, iterations=1)

            # Perform Morphological Reconstruction
            # Use the original binarized image as the "marker" and the dilated image as the "mask"
            reconstructed_image = cv.bitwise_and(img, dilated_image)

        for i in range(10):
            dilated_image = cv.dilate(reconstructed_image, kernel_dil_cross, iterations=1)
            dilated_image = cv.dilate(dilated_image, kernel_dil_vertical, iterations=1)
            dilated_image = cv.dilate(dilated_image, kernel_dil_horizontal, iterations=1)
            dilated_image = cv.dilate(dilated_image, kernel_dil_diag_1, iterations=1)
            dilated_image = cv.dilate(dilated_image, kernel_dil_diag_2, iterations=1)

            # Perform Morphological Reconstruction
            reconstructed_image = cv.bitwise_and(img, dilated_image)

        self.plot_comparing_original_preprocessed(gray, reconstructed_image)


        return reconstructed_image

    def preprocess_image_greyscale(self, img_name):

        gray = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        img = cv.resize(gray, (self.img_size, self.img_size), interpolation=cv.INTER_AREA)

        if self.normalize:
            img = img.Normalize(mean=[0.485], std=[0.229])

        return img

    def preprocess_image_adaptive_threshold(self, img_name):

        gray = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        resized = cv.resize(gray, (self.img_size, self.img_size), interpolation=cv.INTER_AREA)
        blurred = cv.GaussianBlur(resized, (3, 3), 0)
        img = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY_INV, 11, 2)
        return img

    def plot_comparing_original_preprocessed(self, gray, img):
        compare_images = [gray, img]
        titles = ['original', 'preprocessed']
        height, width = gray.shape
        # Set up the figure size based on the image dimensions (adjust dpi as necessary)
        dpi = 100
        figsize = (2 * width / dpi, height / dpi)
        # Create the figure with the appropriate size
        plt.figure(figsize=figsize)
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow(compare_images[i], cmap='gray', vmin=0, vmax=255)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
        # input("Press Enter to continue...")


    def extract_features(self, image_path):

        return self.preprocess_image_greyscale(image_path)


    def experiment(self):
        for i in range(215, 230):
            img_name = self.get_image_path_all_files(i)
            img = self.preprocess_image_morphology(img_name)

            name, order, score = self.extract_from_name(img_name)
            print("{}. : {}, order of drawing {}, score: {}".format(i, name, order, score))
        #self.plot_image(img)

    def expiriment_with_images(self):
        for i in range(0, 5):
            img_name = self.get_image_path_all_files(i)
            # image = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
            # image = cv.resize(image, (256, 256))
            # edges = cv.Canny(image, threshold1=100, threshold2=200)
            # hog_features, hog_image = hog(edges, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
            #                               visualize=True)
            #
            # self.plot_comparing_original_preprocessed(image, hog_image)

            image = self.preprocess_image(img_name)
            hog_features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                          visualize=True)
            self.plot_comparing_original_preprocessed(image, hog_image)

            sift = cv.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(image, None)
            img_with_keypoints = cv.drawKeypoints(image, keypoints, None,
                                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.plot_comparing_original_preprocessed(image, img_with_keypoints)

            orb = cv.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(image, None)
            img_with_keypoints = cv.drawKeypoints(image, keypoints, None,
                                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.plot_comparing_original_preprocessed(image, img_with_keypoints)


            name, order, score = self.extract_from_name(img_name)
            print("{}. : {}, order of drawing {}, score: {}".format(i, name, order, score))
        #self.plot_image(img)


#LoadROCFDataset().experiment()

class ROCFDataset(LoadROCFDataset):
    def __init__(self, image_paths, scores, transform=None, pos_embedding=False):
        LoadROCFDataset.__init__(self)
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transform
        self.pos_embedding = pos_embedding


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.get_image_path_all_files(idx)
        name, order, score = self.extract_from_name(img_path)
        #print(img_path)
        score_one_hot = self.one_hot_class(score)

        #PREDULOZ OBRAZKY
        #preprocessed = self.images[idx]
        preprocessed = self.extract_features(img_path)


        if self.transform is None:
            preprocessed = preprocessed[np.newaxis,:,:]
            x = torch.tensor(preprocessed, dtype=torch.float32)
        if self.transform:
            x = self.transform(preprocessed)

            # Add positional embeddings if enabled
            if self.pos_embedding:
                _, h, w = x.shape  # Extract height and width
                # Create integer-valued x and y positional embeddings
                x_coords = np.tile(np.arange(w), (h, 1))  # Shape: (H, W)
                y_coords = np.tile(np.arange(h).reshape(-1, 1), (1, w))  # Shape: (H, W)

                # Convert to tensors and add as additional channels
                x_coords = torch.tensor(x_coords, dtype=torch.float32).unsqueeze(0) / w  # Shape: (1, H, W)
                y_coords = torch.tensor(y_coords, dtype=torch.float32).unsqueeze(0) / h # Shape: (1, H, W)
                # Concatenate positional embeddings to the original features
                x = torch.cat((x, x_coords, y_coords), dim=0)  # Shape: (5, H, W)

        return x, score_one_hot

    def one_hot_class(self, score):
        class_rocf = self.class_from_score(score)
        return F.one_hot(torch.tensor(class_rocf), num_classes=self.num_score_classes).float()

    def one_hot_score(self, score):
        score_class = min(round(score * 2), 36)
        #### na vypisanie obrazkov, ktore maju prilis vysoke skore
        # if score > 36:
        #     print(score, score_class, self.num_score_classes)
        return F.one_hot(torch.tensor(score_class), num_classes=self.num_score_classes).float()

    def class_from_score(self, score):
        if score < 15:
            return 0
        if score < 22.5:
            return 1
        if score < 30.5:
            return 2
        return 3


