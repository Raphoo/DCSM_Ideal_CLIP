import torch
import random
import numpy as np
from functools import partial
import torch.nn.functional as nnf
from torchvision import transforms as T
import spacy


class TextShuffler:
    """
    A class for shuffling text tokens while preserving certain tokens.
    """

    def __init__(self, preserve_tokens=None):
        """
        Initialize the TextShuffler.

        Args:
            preserve_tokens (list): List of tokens to preserve during shuffling
        """
        self.preserve_tokens = preserve_tokens or []

    def __call__(self, tokens):
        """
        Shuffle tokens while preserving specified tokens.

        Args:
            tokens (list): List of tokens to shuffle

        Returns:
            list: Shuffled tokens with preserved tokens in their original positions
        """
        if not tokens:
            return tokens

        # Create a copy of tokens
        shuffled_tokens = tokens.copy()

        # Find positions of preserved tokens
        preserved_positions = []
        for i, token in enumerate(tokens):
            if token in self.preserve_tokens:
                preserved_positions.append(i)

        # Get tokens that can be shuffled (not preserved)
        shuffleable_tokens = [
            token for i, token in enumerate(tokens) if i not in preserved_positions
        ]

        # Shuffle the shuffleable tokens
        random.shuffle(shuffleable_tokens)

        # Reconstruct the sequence with preserved tokens in their original positions
        shuffleable_idx = 0
        for i in range(len(tokens)):
            if i in preserved_positions:
                continue
            else:
                shuffled_tokens[i] = shuffleable_tokens[shuffleable_idx]
                shuffleable_idx += 1

        return shuffled_tokens


class ImagePerturbation:
    """
    Base class for image perturbations.
    """

    def __init__(self):
        pass

    def __call__(self, image):
        """
        Apply perturbation to image.

        Args:
            image: PIL Image or tensor

        Returns:
            Perturbed image
        """
        raise NotImplementedError


class RandomCrop(ImagePerturbation):
    """
    Random crop perturbation.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.transform = T.RandomCrop(crop_size)

    def __call__(self, image):
        return self.transform(image)


class RandomRotation(ImagePerturbation):
    """
    Random rotation perturbation.
    """

    def __init__(self, degrees):
        self.degrees = degrees
        self.transform = T.RandomRotation(degrees)

    def __call__(self, image):
        return self.transform(image)


class ColorJitter(ImagePerturbation):
    """
    Color jitter perturbation.
    """

    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image):
        return self.transform(image)


def get_text_perturbation(perturbation_type, **kwargs):
    """
    Get text perturbation function.

    Args:
        perturbation_type (str): Type of perturbation
        **kwargs: Additional arguments

    Returns:
        callable: Perturbation function
    """
    if perturbation_type == "shuffle":
        preserve_tokens = kwargs.get("preserve_tokens", [])
        return TextShuffler(preserve_tokens=preserve_tokens)
    else:
        return lambda x: x


def get_image_perturbation(perturbation_type, **kwargs):
    """
    Get image perturbation function.

    Args:
        perturbation_type (str): Type of perturbation
        **kwargs: Additional arguments

    Returns:
        callable: Perturbation function
    """
    if perturbation_type == "crop":
        crop_size = kwargs.get("crop_size", 224)
        return RandomCrop(crop_size)
    elif perturbation_type == "rotation":
        degrees = kwargs.get("degrees", 10)
        return RandomRotation(degrees)
    elif perturbation_type == "color_jitter":
        return ColorJitter()
    else:
        return lambda x: x
