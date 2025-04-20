import os
import cv2 as cv
from matplotlib import pyplot as plt
import torch.nn.functional as F


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.feature import hog
import torch.nn.functional as F
import pandas as pd


class LoadROCFDataset(Dataset):
    def __init__(self, img_size=500, normalize=False, transform=None, pos_embedding=False):
        self.directory_name_clinical = "./orezane_1500x1500px/Klinicka skupina/"
        self.file_names_clinical = self.list_files(self.directory_name_clinical)

        self.directory_name_control = "./orezane_1500x1500px/Kontrolna skupina/"
        self.file_names_control = self.list_files(self.directory_name_control)

        self.directory_name_heatmap_img = "./heatmap_visualization/"
        self.file_names_heatmap_imgs = self.list_files(self.directory_name_heatmap_img)

        self.all_file_names = self.file_names_control + self.file_names_clinical

        self.partial_scores_filename = "./orezane_1500x1500px/Neuropsy_data_Reyovky.xlsx"

        #self.num_score_classes = 73 # if we predict score
        self.num_score_classes = 4 # if we predict classes
        self.img_size = img_size
        self.normalize = normalize
        self.transform = transform
        self.pos_embedding = pos_embedding


    def __len__(self):
        return len(self.all_file_names)


    def __getitem__(self, idx):
        img_path = self.get_image_path_all_files(idx)
        preprocessed = self.extract_features(img_path)
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

    def preprocess_image_morphology(self, img_name, visualize=True):

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

        if visualize:
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

    def plot_comparing_original_preprocessed(self, gray, img, save=False, file_name='img', titles=None):
        if titles is None:
            titles = ['original', 'preprocessed']
        compare_images = [gray, img]
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
        if save:
            plt.savefig(f'{file_name}.png')
        plt.show()
        # input("Press Enter to continue...")


    def extract_features(self, image_path):

        return self.preprocess_image_greyscale(image_path)

    def preproces_and_transform_img(self, img_path):
        preprocessed = self.extract_features(img_path)
        if self.transform is None:
            preprocessed = preprocessed[np.newaxis, :, :]
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
                y_coords = torch.tensor(y_coords, dtype=torch.float32).unsqueeze(0) / h  # Shape: (1, H, W)
                # Concatenate positional embeddings to the original features
                x = torch.cat((x, x_coords, y_coords), dim=0)  # Shape: (5, H, W)

        if len(x.shape) == 3:
            x = x[np.newaxis, :, :, :]
        return x

    def prepare_for_model(self, img):
        #img = np.transpose(img[:, :, ::-1], (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        preprocessed = img[np.newaxis, :, :]
        x = torch.tensor(preprocessed, dtype=torch.float32)
        if len(x.shape) == 3:
            x = x[np.newaxis, :, :, :]
        return x

    def generate_heatmap(self, img_torch, img_np, model, rectangle_num, rectangle_size, color='black', f=None):
        img_np = img_np / 255

        pred = model(img_torch)
        class_id = torch.argmax(pred[0])
        print('original class:', class_id)
        f.write(f"original class: {class_id}\n\n")

        heatmap = np.zeros([rectangle_num, rectangle_num])
        xs = np.linspace(0, self.img_size, rectangle_num)
        ys = np.linspace(0, self.img_size, rectangle_num)
        for idx_y, y in enumerate(ys):
            in_array = torch.zeros([rectangle_num, 1, self.img_size, self.img_size])
            for idx_x, x in enumerate(xs):
                c_img = img_np.copy()
                c_img = cv.rectangle(c_img, (int(x), int(y)),
                                      (int(x + rectangle_size), int(y + rectangle_size)),
                                     (0, 0, 0) if color == 'black' else (1, 1, 1), -1)
                # plt.imshow(c_img, cmap='gray')
                # plt.show()
                c_img = self.prepare_for_model(c_img)
                in_array[idx_x] = c_img[:, :]
            pred = model(in_array)
            pred = F.softmax(pred, dim=1)
            print('new classes:', torch.argmax(pred, dim=1))
            f.write(f"new classes: {torch.argmax(pred, dim=1)}\n")
            heatmap[idx_y, :] = 1 - pred[:, class_id].cpu().detach().numpy()
            print('heatmap row: ', (1 - pred[:, class_id]))

        return (heatmap * 255).astype(np.uint8)

    def visualize_heatmaps(self, model, model_name=''):
        model.eval()
        for color in ['black', 'white']:
            for test_img in self.file_names_heatmap_imgs:
                img_torch = self.preproces_and_transform_img(test_img)
                img_np = self.extract_features(test_img)
                print('\nimg_torch: ', img_torch.shape, 'img_np: ', img_np.shape)

                img_name = os.path.basename(test_img).replace('.jpg', '').replace('.png', '')
                with open(f'{model_name}_{img_name}_{color}.txt', 'w') as f:
                    f.write(f"{model_name}_{img_name}\n")
                    hmap = self.generate_heatmap(img_torch, img_np, model, 17, 25, color=color, f=f)

                    f.write('\nheatmap:\n' + '\n'.join([str(list(line)) for line in hmap]))

                self.plot_comparing_original_preprocessed(img_np, hmap, save=True, file_name=f'{model_name}_{img_name}_{color}',
                                                          titles=[f'{img_name}', f'heatmap_{color}'])


    def experiment(self):
        for i in range(400, 430):
            img_name = self.get_image_path_all_files(i)
            img = self.preprocess_image_morphology(img_name)

            name, order, score = self.extract_from_name(img_name)
            print("{}. : {}, order of drawing {}, score: {}".format(i, name, order, score))
        #self.plot_image(img)




#LoadROCFDataset().experiment()

class ROCFDataset(LoadROCFDataset):
    def __init__(self, image_paths, scores, transform=None, pos_embedding=False):
        LoadROCFDataset.__init__(self, transform=transform, pos_embedding=pos_embedding)
        self.image_paths = image_paths
        self.scores = scores

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.get_image_path_all_files(idx)
        name, order, score = self.extract_from_name(img_path)
        #print(img_path)
        score_one_hot = self.one_hot_class(score)

        x = self.preproces_and_transform_img(img_path)

        return x, score_one_hot


# rocf_dataset = LoadROCFDataset()
# rocf_dataset.experiment()