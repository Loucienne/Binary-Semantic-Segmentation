import os
import torch
import shutil
import numpy as np

from PIL import Image  # Used for image processing
from tqdm import tqdm  # Used for progress bars
from urllib.request import urlretrieve  # Used for downloading files

# Define a dataset class for the Oxford Pet Dataset
class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        """
        Initializes the dataset.

        Args:
            root (str): Root directory where the dataset is stored.
            mode (str): Specifies whether to load "train", "valid", or "test" set.
            transform (callable, optional): Optional transform to be applied to samples.
        """
        assert mode in {"train", "valid", "test"}  # Ensure mode is valid

        self.root = root
        self.mode = mode
        self.transform = transform

        # Define paths for images and masks
        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        # Read dataset split (train/valid/test)
        self.filenames = self._read_split()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding mask.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Contains 'image', 'mask', and 'trimap'.
        """
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        # Load the image (RGB) and trimap (grayscale)
        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))

        # Convert trimap to binary mask
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)

        # Apply transformations if provided
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        """
        Converts the trimap into a binary mask.

        Args:
            mask (numpy.ndarray): Original trimap.

        Returns:
            numpy.ndarray: Processed binary mask.
        """
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0  # Convert class 2 to background (0)
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0  # Convert classes 1 & 3 to foreground (1)
        return mask

    def _read_split(self):
        """
        Reads the dataset split file and returns filenames.

        Returns:
            list: List of filenames.
        """
        # Determine which file to read based on mode
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)

        # Read file and extract filenames
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")

        filenames = [x.split(" ")[0] for x in split_data]

        # to not have a test too big
        if self.mode=="test":
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]

        # Split dataset: 90% train, 10% validation
        if self.mode == "train":
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]

        return filenames

    @staticmethod
    def download(root):
        """
        Downloads and extracts the dataset.

        Args:
            root (str): Root directory to store the dataset.
        """
        # Download images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # Download annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

# A simplified version of OxfordPetDataset with resizing and format conversion
class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        """
        Retrieves and processes an image and its corresponding mask.

        Returns:
            dict: Contains resized 'image', 'mask', and 'trimap' in CHW format.
        """
        sample = super().__getitem__(*args, **kwargs)

        # Resize images and masks to 256x256
        image = np.array(Image.fromarray(sample["image"]).resize((128, 128), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((128, 128), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((128, 128), Image.NEAREST))

        # Convert from HWC format to CHW format
        sample["image"] = np.moveaxis(image, -1, 0)  # Move channels to the first dimension
        sample["mask"] = np.expand_dims(mask, 0)  # Add a channel dimension
        sample["trimap"] = np.expand_dims(trimap, 0)  # Add a channel dimension

        return sample

# Class to show progress during downloads
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update the progress bar.

        Args:
            b (int): Blocks transferred so far.
            bsize (int): Size of each block.
            tsize (int, optional): Total size of the file.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # Update progress

# Function to download a file with progress bar
def download_url(url, filepath):
    """
    Downloads a file from a URL with a progress bar.

    Args:
        url (str): URL of the file.
        filepath (str): Local file path to save the file.
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist

    if os.path.exists(filepath):  # Skip download if file already exists
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n  # Set total size at the end

# Function to extract archive files
def extract_archive(filepath):
    """
    Extracts a compressed archive.

    Args:
        filepath (str): Path to the archive file.
    """
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]

    if not os.path.exists(dst_dir):  # Skip if already extracted
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode):
    assert mode in {"train", "valid", "test"}, "Invalid mode! Use 'train', 'valid', or 'test'."
    hard_dataset = OxfordPetDataset(root=data_path, mode=mode)
    easy_dataset = SimpleOxfordPetDataset(root=data_path, mode=mode)
    return easy_dataset


