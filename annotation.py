import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
from pathlib import Path
from ROCFDataset_for_CNN import LoadROCFDataset, ROCFDataset

def annotate_images(dataset, output_file="./orezane_1500x1500px/annotations.csv"):
    annotated_paths = set()

    # Load existing annotations if file exists
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    annotated_paths.add(row[0])

    # Create file with header if new
    if not Path(output_file).exists():
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["img_path"] + [f"x{i+1}" for i in range(5)] + [f"y{i+1}" for i in range(5)]
            writer.writerow(header)

    total = len(dataset.all_file_names)
    for idx, img_path in enumerate(dataset.all_file_names, 1):
        if img_path in annotated_paths:
            print(f"[{idx}/{total}] Skipping already annotated: {img_path}")
            continue

        confirmed = False
        while not confirmed:
            # Let user click 5 points
            img = mpimg.imread(img_path)
            plt.imshow(img, cmap="gray")
            plt.title(f"[{idx}/{total}] Select 5 points for {Path(img_path).name}")
            points = plt.ginput(5, timeout=0)
            plt.close()

            # Show confirmation
            xs, ys = zip(*points)
            plt.imshow(img, cmap="gray")
            plt.scatter(xs, ys, c="red", marker="o")
            plt.title("Confirm points: press 'y' to accept, 'n' to redo")
            plt.draw()
            key = None
            while key not in ["y", "n"]:
                key = input("Confirm points? (y/n): ").strip().lower()
            plt.close()

            if key == "y":
                confirmed = True
                # Append row to CSV
                with open(output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([img_path] + list(xs) + list(ys))
                print(f"[{idx}/{total}] Annotated: {img_path}")
            else:
                print("Redoing selection...")

    print(f"Annotation complete. Saved to {output_file}")

dataset = LoadROCFDataset()
annotate_images(dataset)