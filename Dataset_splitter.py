import os
import cv2

class DatasetSplitter:
    def __init__(self, split_dir, output_dir):
        """
        :param split_dir: Directory containing the split txt files.
        :param output_dir: Base directory to create train/val/test folders.
        """
        self.split_dir = split_dir
        self.output_dir = output_dir

    def _copy_images(self, image_paths, target_dir, remove_score=False):
        os.makedirs(target_dir, exist_ok=True)

        for path in image_paths:
            print(path)

            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue

            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image: {path}")
                continue

            resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)

            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)

            if remove_score:
                # Remove last underscore + value
                parts = name.split("_")
                if len(parts) > 1:
                    name = "_".join(parts[:-1])

            new_filename = name + ext
            target_path = os.path.join(target_dir, new_filename)

            cv2.imwrite(target_path, resized)

    def create_splits(self):
        # Define split file names
        split_files = {
            "train_without_score": "train_len_1041_4_classes.txt",
            "val_without_score": "val_len_1041_4_classes.txt",
            "test_without_score": "test_len_1041_4_classes.txt"
        }

        for split_name, file_name in split_files.items():
            print('\n\n\n', split_name, '-------------------')
            file_path = os.path.join(self.split_dir, file_name)

            if not os.path.exists(file_path):
                print(f"Error: Split file not found: {file_path}")
                continue

            # Read image paths from txt
            with open(file_path, 'r') as f:
                image_paths = [line.strip() for line in f.readlines() if line.strip()]

            # Define target directory
            target_dir = os.path.join(self.output_dir, split_name)

            # Copy images using the existing function
            self._copy_images(image_paths, target_dir, True)
            print(f"{split_name.capitalize()} images copied: {len(image_paths)}")


# Example usage
if __name__ == "__main__":
    split_dir = r"C:\Users\lucin\OneDrive\Desktop\diplomovka\thesis_code\orezane_1500x1500px\Train_Val_Test_split"
    output_dir = r"C:\Users\lucin\OneDrive\Desktop\diplomovka\thesis_code\train_split"

    splitter = DatasetSplitter(split_dir, output_dir)
    splitter.create_splits()