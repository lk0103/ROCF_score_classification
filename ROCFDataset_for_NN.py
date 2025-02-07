import os
import cv2 as cv
from matplotlib import pyplot as plt


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.feature import hog
import torch.nn.functional as F


class LoadROCFDataset(Dataset):
    def __init__(self):
        self.directory_name_clinical = "./orezane_1500x1500px/Klinicka skupina/"
        self.file_names_clinical = self.list_files(self.directory_name_clinical)

        self.directory_name_control = "./orezane_1500x1500px/Kontrolna skupina/"
        self.file_names_control = self.list_files(self.directory_name_control)

        #self.num_score_classes = 73 # if we predict score
        self.num_score_classes = 4 # if we predict classes


    def __len__(self):
        return len(self.file_names_control)


    def __getitem__(self, idx):
        img_path = self.get_image_path_control_group(idx)
        preprocessed = self.preprocess_image(img_path)
        name, order, score = self.extract_from_name(img_path)
        return torch.tensor(preprocessed, dtype=torch.float64), torch.tensor(score, dtype=torch.float32)


    def list_files(self, directory):
        files = []
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                files.append(filename)
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

        img_name = img_name.replace('.jpg', '')
        img_name = img_name.replace('.png', '')
        name, order, score = img_name.split('_')
        order = int(order)
        score = float(score.replace(',', '.'))
        return name, order, score


    def get_image_path_control_group(self, i):
        return self.directory_name_control + self.file_names_control[i]


    def preprocess_image(self, img_name):

        gray = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        resized = cv.resize(gray, (256, 256), interpolation=cv.INTER_AREA)
        blurred = cv.GaussianBlur(resized, (3, 3), 0)
        blurred = cv.fastNlMeansDenoising(blurred, None, 1.0, 7, 21) #1.25 - trosku menej sumu v pozadi ale
                                                                    #viac z ciar zachovane

        img = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv.THRESH_BINARY_INV, 11, 2)

        #Use dilation followed by erosion (opening) to enhance strokes
        kernel = np.ones((2, 2), np.uint8)
        img = cv.dilate(img, kernel, iterations=1)
        img = cv.erode(img, kernel, iterations=1)

        #self.plot_comparing_original_preprocessed(gray, img)

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

        return self.binary_sift(image_path)

    def binary_orb(self, image_path):
        image = self.preprocess_image(image_path)
        orb = cv.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        num_descriptors = 50
        if descriptors is None:
            descriptors = np.zeros((num_descriptors, 128), dtype=np.float32)
        elif len(descriptors) < num_descriptors:
            padding = np.zeros((num_descriptors - len(descriptors), 128), dtype=np.float32)
            descriptors = np.vstack((descriptors, padding))
        else:
            # If more than num_descriptors, select the top ones
            descriptors = descriptors[:num_descriptors]
        return descriptors.flatten()

    def binary_sift(self, image_path):
        image = self.preprocess_image(image_path)
        sift = cv.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        num_descriptors = 50
        if descriptors is None:
            descriptors = np.zeros((num_descriptors, 128), dtype=np.float32)
        elif len(descriptors) < num_descriptors:
            padding = np.zeros((num_descriptors - len(descriptors), 128), dtype=np.float32)
            descriptors = np.vstack((descriptors, padding))
        else:
            # If more than num_descriptors, select the top ones
            descriptors = descriptors[:num_descriptors]
        return descriptors.flatten()

    def binary_hog(self, image_path):
        image = self.preprocess_image(image_path)
        hog_features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        return hog_features

    def canny_hog(self, image_path):
        # POVODNY PREPROCESSING LEN S CANNY - ALE TO NEBOLO DOBRE - HOG FEATTURES BOLI VELMI NEVYRAZNE
        # A VELA DETAILOV SA STRATILO
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (256, 256))
        edges = cv.Canny(image, threshold1=100, threshold2=200)
        hog_features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        return np.hstack([image.flatten(), hog_features])

    def experiment(self):
        for i in range(0, 20):
            img_name = self.get_image_path_control_group(i)
            img = self.preprocess_image(img_name)

            name, order, score = self.extract_from_name(img_name)
            print("{}. : {}, order of drawing {}, score: {}".format(i, name, order, score))
        #self.plot_image(img)

    def expiriment_with_images(self):
        for i in range(0, 5):
            img_name = self.get_image_path_control_group(i)
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


#LoadROCFDataset().expiriment_with_images()

class ROCFDataset(LoadROCFDataset):
    def __init__(self, image_paths, scores, transform=None):
        LoadROCFDataset.__init__(self)
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transform

        #PREDOLUZENIE OBRAZKOV
        # self.images = []
        # for idx in range(len(self)):
        #     img_path = self.get_image_path_control_group(idx)
        #     preprocessed = self.extract_features(img_path)
        #
        #     x = torch.tensor(preprocessed, dtype=torch.float32)
        #     if self.transform:
        #         x = self.transform(x)
        #
        #     self.images.append(x)


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.get_image_path_control_group(idx)
        name, order, score = self.extract_from_name(img_path)
        #print(img_path)
        score_one_hot = self.one_hot_class(score)

        #PREDULOZ OBRAZKY
        #preprocessed = self.images[idx]
        preprocessed = self.extract_features(img_path)
        x = torch.tensor(preprocessed, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)

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
