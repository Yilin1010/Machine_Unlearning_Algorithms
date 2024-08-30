from typing import Tuple, Union
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from transformers import ViTImageProcessor
from torch.utils.data import Dataset, DataLoader, Subset, dataset
from datasets import load_dataset
from datasets import Dataset as HFDataset
from image_net_utils import map_clsidx_imagenet_collate_fn
from typing import Tuple, Union
from PIL import Image
import utils

device = utils.device_config()
utils.set_seed()

# Define the normalization transform for MNIST dataset
normalize = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def load_mnist(batch_size=64):

    # load MINIST dataset
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=normalize
    )
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=normalize
    )
    return train_set, test_set


def load_tiny_imagenet(
    model_name="google/vit-base-patch16-224",
    data_path="Maysee/tiny-imagenet",
    data_info="data/tiny_infos.json",
):
    print("loading tiny")
    tiny_imagenet_train = load_dataset(data_path, split="train")
    tiny_imagenet_test = load_dataset(data_path, split="valid")

    return tiny_imagenet_train, tiny_imagenet_test


# apply hugging face model processer
# Transform Hugging Face dataset to PyTorch DataLoader-compatible
def vit_dataset_collate_fn(model_name="google/vit-base-patch16-224"):

    image_processor = ViTImageProcessor.from_pretrained(model_name)

    # Set up data transformations
    transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB"))])

    def collate_fn(batch):

        images, labels = batch

        if not images:  # Check if the images list is empty
            return None, None

        # Apply transform to each image individually
        images = [transform(image) for image in images]

        # Process images using the image processor
        # image_processor api expects images in RGB format rather than raw image
        processed_images = image_processor(images, return_tensors="pt")

        return processed_images["pixel_values"], torch.tensor(labels)

    return collate_fn


# Transform Hugging Face dataset to PyTorch DataLoader-compatible
def hf_collate_fn(batch):

    # Set up data transformations

    labels = [item["label"] for item in batch]
    images = [item["image"] for item in batch]

    # Convert labels to a tensor
    return images, labels


def combined_imagenet_collate_fn(model_name="google/vit-base-patch16-224"):
    map_fn = map_clsidx_imagenet_collate_fn()
    hf_fn = hf_collate_fn
    vit_fn = vit_dataset_collate_fn(model_name)

    def collate_fn(batch):
        batch = map_fn(batch)
        # Apply hf_collate_fn first
        images, labels = hf_fn(batch)

        # Apply vit_dataset_collate_fn next
        processed_images, labels = vit_fn((images, labels))

        return processed_images, labels

    return collate_fn



def unlearn_class(
    train_set: Union[Dataset, HFDataset], 
    class_idx: int, batch_size: int = 64, 
    collate_fn=None):
    # -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for forgetting a specific class and retaining the others.

    Args:
        train_set (Union[Dataset, HFDataset]): The dataset from which to create subsets.
    class_idx (int): The class index to forget.
    batch_size (int): The number of samples per batch.
    collate_fn (callable, optional): Function to collate data samples into batch.

    Returns:
    Tuple[DataLoader, DataLoader]: A tuple containing DataLoaders for the forget set of the specific class
                                   and the retain set for all other classes.
    """

    # Gather indices for the specified class to forget and others to retain
    if isinstance(train_set, HFDataset):
        mask = [item["label"] == class_idx for item in train_set]
        # Use the mask to split the dataset
        forget_set = train_set.select([i for i, masked in enumerate(mask) if masked])
        retain_set = train_set.select(
            [i for i, masked in enumerate(mask) if not masked]
        )

        forget_loader = DataLoader(
            forget_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        retain_loader = DataLoader(
            retain_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

    else:

        forget_indices = []
        retain_indices = []

        for idx, (_, target) in enumerate(train_set):
            if target == class_idx:
                forget_indices.append(idx)
            else:
                retain_indices.append(idx)

        forget_set = Subset(train_set, forget_indices)
        retain_set = Subset(train_set, retain_indices)

        # Create DataLoader for the forget set of the specified class
        forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=True)

        # Create DataLoader for the retain set of all other classes
        retain_loader = DataLoader(retain_set, batch_size=batch_size, shuffle=True)

    print(f"forget: {len(forget_loader.dataset)} retain: {len(retain_loader.dataset)}")
    return forget_loader, retain_loader


def unlearn_samples_balanced(train_set, forget_ratio, batch_size=64):
    """Function to unlearn certain samples

    Args:
        train_set (_type_): _description_
        forget_ratio (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 64.

    Returns:
        _type_: _description_
    """

    class_samples = {
        c: [] for c in range(10)
    }  # Assuming 10 classes, adjust as necessary

    # Separate samples by class
    for idx, (_, target) in enumerate(train_set):
        class_samples[target].append(idx)

    # Shuffle indices and select forget_ratio for each class
    forget_indices = []
    retain_indices = []
    for indices in class_samples.values():
        np.random.shuffle(indices)
        split_point = int(len(indices) * forget_ratio)
        forget_indices.extend(indices[:split_point])
        retain_indices.extend(indices[split_point:])

    # Create DataLoader for the forget and retain sets
    forget_loader = DataLoader(
        Subset(train_set, forget_indices), batch_size=batch_size, shuffle=True
    )
    retain_loader = DataLoader(
        Subset(train_set, retain_indices), batch_size=batch_size, shuffle=True
    )

    return forget_loader, retain_loader


# Function to unlearn certain samples
def unlearn_samples_random(train_set, forget_ratio, batch_size=64):
    # Total number of samples in the train set
    total_samples = len(train_set)

    # Calculate the number of samples to forget
    forget_count = int(total_samples * forget_ratio)

    # Generate all indices and shuffle them
    indices = torch.randperm(total_samples)

    # Split indices into forget and retain sets
    forget_indices = indices[:forget_count]
    retain_indices = indices[forget_count:]

    # Create DataLoader for the forget and retain sets
    forget_loader = DataLoader(
        Subset(train_set, forget_indices), batch_size=batch_size, shuffle=True
    )
    retain_loader = DataLoader(
        Subset(train_set, retain_indices), batch_size=batch_size, shuffle=True
    )

    return forget_loader, retain_loader


def sample_loader(loader, sample_num):
    # Retrieve arguments from the original DataLoader
    batch_size = loader.batch_size
    collate_fn = loader.collate_fn

    indices = np.random.choice(len(loader.dataset), sample_num, replace=False)
    sampled_loader = DataLoader(
        Subset(loader.dataset, indices), batch_size=batch_size, collate_fn=collate_fn
    )

    return sampled_loader
