import numpy as np
import os
import cv2
import random
from itertools import combinations
# Ensure reproducibility
np.random.seed(1337)
random.seed(1337)

class SignatureDataGenerator:
    """
    A data generator for multiple signature datasets containing genuine and forged signatures.
    """

    def __init__(self, dataset, img_height=155, img_width=220, batch_sz=4):
        """
        Initialize the generator with dataset parameters.

        Args:
            dataset: Dictionary containing dataset paths and writer splits.
            img_height: Target height for resizing images.
            img_width: Target width for resizing images.
            batch_sz: Batch size for training/testing.
        """
        self.dataset = dataset
        self.img_height = img_height
        self.img_width = img_width
        self.batch_sz = batch_sz

        # Storage for train and test writers
        self.train_writers = []
        self.test_writers = []

        # Validate dataset paths and load writer information
        self._load_writers()

    def preprocess_image(self, img_path):
        """
        Load and preprocess an image.

        Args:
            img_path: Path to the image file.

        Returns:
            Preprocessed image as a NumPy array.
        """
        if not os.path.exists(img_path):
            print(f"⚠ Warning: Missing image file {img_path}")
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = cv2.resize(img, (self.img_width, self.img_height))

        img = img.astype(np.float32) / 255.0  # Normalize
        return img

    def _load_writers(self):
        """
        Load writer directories and validate existence.
        """
        for dataset_name, dataset_info in self.dataset.items():
            dataset_path = dataset_info["path"]
            train_writers = dataset_info["train_writers"]
            test_writers = dataset_info["test_writers"]

            for writer in train_writers + test_writers:
                writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
                if os.path.exists(writer_path):
                    if writer in train_writers:
                        self.train_writers.append((dataset_path, writer))
                    else:
                        self.test_writers.append((dataset_path, writer))
                else:
                    print(f"⚠ Warning: Writer directory not found: {writer_path}")

    def generate_pairs(self, dataset_path, writer):
        """
        Generate positive (genuine-genuine) and negative (genuine-forged) pairs.

        Args:
            dataset_path: Path to dataset.
            writer: Writer ID.

        Yields:
            (image1, image2, label) pairs.
        """
        writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
        genuine_path = os.path.join(writer_path, "genuine")
        forged_path = os.path.join(writer_path, "forged")

        if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
            print(f"⚠ Warning: Missing required directories for writer {writer}")
            return

        # Load file names
        genuine_files = sorted([os.path.join(genuine_path, f) for f in os.listdir(genuine_path) if f.endswith(('.png', '.jpg'))])
        forged_files = sorted([os.path.join(forged_path, f) for f in os.listdir(forged_path) if f.endswith(('.png', '.jpg'))])

        if len(genuine_files) < 2 or len(forged_files) == 0:
            print(f"⚠ Warning: Not enough images for writer {writer} (genuine: {len(genuine_files)}, forged: {len(forged_files)})")
            return

        # Generate positive pairs (genuine-genuine)
        for i, j in combinations(range(len(genuine_files)), 2):
            img1 = self.preprocess_image(genuine_files[i])
            img2 = self.preprocess_image(genuine_files[j])
            yield (img1, img2, 1)

        # Generate negative pairs (genuine-forged)
        for i in range(len(genuine_files)):
            for j in range(len(forged_files)):
                img1 = self.preprocess_image(genuine_files[i])
                img2 = self.preprocess_image(forged_files[j])
                yield (img1, img2, 0)

    def get_data(self, writers_list):
        """
        Generate data pairs from the given list of writers.

        Args:
            writers_list: List of (dataset_path, writer_id) tuples.

        Yields:
            (X1, X2, y) - Processed input pairs and labels.
        """
        pairs = []
        labels = []

        for dataset_path, writer in writers_list:
            for pair in self.generate_pairs(dataset_path, writer):
                pairs.append(pair[:2])
                labels.append(pair[2])

                if len(pairs) == self.batch_sz:
                    X1 = np.array([p[0] for p in pairs], dtype=np.float32)
                    X2 = np.array([p[1] for p in pairs], dtype=np.float32)
                    y = np.array(labels, dtype=np.int8)
                    yield (X1, X2, y)
                    pairs = []
                    labels = []

        if pairs:
            X1 = np.array([p[0] for p in pairs], dtype=np.float32)
            X2 = np.array([p[1] for p in pairs], dtype=np.float32)
            y = np.array(labels, dtype=np.int8)
            yield (X1, X2, y)

    def get_train_data(self):
        """
        Generate training data pairs.
        """
        return self.get_data(self.train_writers)

    def get_test_data(self):
        """
        Generate testing data pairs.
        """
        return self.get_data(self.test_writers)