import os
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import random
import torch
import math
import json


class ObjaverseDataset_spatial(Dataset):
    def __init__(
        self,
        image_dir,
        no_special_words,
        settype=0,
        add_spw=False,
        cutmix=False,
        notchance=0,
    ):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            notchance: chance that the opposite caption is turned into a negation of the positive.

        Itereate through spatial dataset folder and returns
        """
        self.image_dir = image_dir

        self.settype = settype
        self.no_special_words = no_special_words

        self.special_replacements = {
            "above": "above",
            "below": "below",
            "rightof": "to the right of",
            "leftof": "to the left of",
        }

        self.opposite_dict = {
            "above": "below",
            "below": "above",
            "rightof": "leftof",
            "leftof": "rightof",
        }

        self.special_replacements_brackets = {
            "above": "above <above>",
            "below": "below <below>",
            "rightof": "to the right <right> of",
            "leftof": "to the left <left> of",
        }
        self.image_filenames = [
            f
            for f in os.listdir(image_dir)
            if "CUTMIX" not in f
            and (f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".jpg"))
            and (sum([pp in f for pp in self.special_replacements]) > 0)
        ]

        self.cutmix = cutmix
        self.notchance = notchance

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get the image file path
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load the image
        image = Image.open(img_path)

        if self.cutmix:
            opp_img_name = os.path.join(self.image_dir, "CUTMIX_" + img_name)
            opposite_image = Image.open(opp_img_name)

        # Extract object A, relation, and object B from the filename
        # Example filename: "A above B.png"
        img_basename = os.path.splitext(img_name)[0]  # Remove the file extension
        parts = img_basename.split("_")  # Split by space

        if len(parts) != 3:
            raise ValueError(
                f"Filename '{img_name}' does not follow the expected format of 'A above B.png'"
            )

        A = parts[0]
        position = parts[1]
        B = parts[2]

        randomnot = random.randint(0, 100)

        if self.settype == 0:
            # if settype is 1, then we flip. if not, we don't.
            if self.no_special_words:
                caption = (
                    f"A picture of a {A} {self.special_replacements[position]} a {B}"
                )
                opp_caption = (
                    f"A picture of a {B} {self.special_replacements[position]} a {A}"
                )

                if randomnot < self.notchance * 50:
                    opp_caption = f"Not a picture of a {A} {self.special_replacements[position]} a {B}"
                elif randomnot < self.notchance * 100:
                    opp_caption = f"A picture of a {A} not {self.special_replacements[position]} a {B}"

            else:
                caption = f"A picture of a {A} {self.special_replacements_brackets[position]} a {B}"
                opp_caption = f"A picture of a {B} {self.special_replacements_brackets[position]} a {A}"

                if randomnot < self.notchance * 50:
                    opp_caption = f"Not a picture of a {A} {self.special_replacements_brackets[position]}a {B}"
                elif randomnot < self.notchance * 100:
                    opp_caption = f"A picture of a {A} not {self.special_replacements_brackets[position]} a {B}"
        else:
            if self.no_special_words:
                caption = f"A picture of a {B} {self.special_replacements[self.opposite_dict[position]]} a {A}"
                opp_caption = f"A picture of a {A} {self.special_replacements[self.opposite_dict[position]]} a {B}"

                if randomnot < self.notchance * 50:
                    opp_caption = f"Not a picture of a {B} {self.special_replacements[self.opposite_dict[position]]} a {A}"
                elif randomnot < self.notchance * 100:
                    opp_caption = f"A picture of a {B} not {self.special_replacements[self.opposite_dict[position]]} a {A}"

            else:
                caption = f"A picture of a {B} {self.special_replacements_brackets[self.opposite_dict[position]]} a {A}"
                opp_caption = f"A picture of a {A} {self.special_replacements_brackets[self.opposite_dict[position]]} a {B}"

                if randomnot < self.notchance * 50:
                    opp_caption = f"Not a picture of a {B} {self.special_replacements_brackets[self.opposite_dict[position]]} a {A}"
                elif randomnot < self.notchance * 100:
                    opp_caption = f"A picture of a {B} not {self.special_replacements_brackets[self.opposite_dict[position]]} a {A}"

        if self.cutmix:
            return {
                "image_options": image,
                "opposite_image_options": opposite_image,
                "caption_options": caption,
                "opposite_caption_options": opp_caption,
            }
        else:
            return {
                "image_options": image,
                "caption_options": caption,
                "opposite_caption_options": opp_caption,
            }


class ObjaverseDataset_color(Dataset):
    def __init__(
        self,
        image_dir,
        no_special_words,
        settype=0,
        add_spw=False,
        cutmix=False,
        notchance=0,
    ):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        Itereate through diff color images folder.
        images are of the format c1_A_c2_B.png, or MIRROR_c1_A_c2_B.png.
        color words do not need special brackets since they are visual.
        two image dirs, one of them being the SMALLBIG one.
        """
        self.image_dir = image_dir

        self.settype = settype
        self.no_special_words = no_special_words

        self.image_filenames = [
            f
            for f in os.listdir(image_dir)
            if "MIRROR" not in f
            and (f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".jpg"))
        ]

        self.image_dir_two = image_dir.replace(
            "_attributes_overlay_withbg", "_smallbig_withbg"
        )

        self.image_filenames += [
            f
            for f in os.listdir(self.image_dir_two)
            if "MIRROR" not in f
            and (f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".jpg"))
        ]

        self.cutmix = cutmix
        self.notchance = notchance

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get the image file path
        img_name = self.image_filenames[idx]
        if "small" in img_name:
            img_path = os.path.join(self.image_dir_two, img_name)
        else:
            img_path = os.path.join(self.image_dir, img_name)

        # Load the image
        image = Image.open(img_path)

        if self.cutmix:
            if "small" in img_name:
                opp_img_name = os.path.join(self.image_dir_two, "MIRROR_" + img_name)
            else:
                opp_img_name = os.path.join(self.image_dir, "MIRROR_" + img_name)
            opposite_image = Image.open(opp_img_name)

        # Extract object A, relation, and object B from the filename
        # Example filename: "A above B.png"
        img_basename = os.path.splitext(img_name)[0]  # Remove the file extension
        parts = img_basename.split("_")  # Split by space

        c1 = parts[0]
        A = parts[1]
        c2 = parts[2]
        B = parts[3]

        randomnot = random.randint(0, 100)

        if self.settype == 0:
            caption = f"A picture of a {c1} {A} and a {c2} {B}"
            opp_caption = f"A picture of a {c2} {A} and a {c1} {B}"

            if randomnot < self.notchance * 100:
                opp_caption = f"Not a picture of a {c1} {A} and a {c2} {B}"

        else:
            caption = f"A picture of a {c2} {B} and a {c1} {A}"
            opp_caption = f"A picture of a {c1} {B} and a {c2} {A}"
            if randomnot < self.notchance * 100:
                opp_caption = f"Not a picture of a {c2} {B} and a {c1} {A}"

        return {
            "image_options": image,
            "opposite_image_options": opposite_image,
            "caption_options": caption,
            "opposite_caption_options": opp_caption,
        }


class COCODataset_color_nooppimg(Dataset):
    def __init__(
        self,
        image_dir,
        no_special_words,
        settype=0,
        add_spw=False,
        cutmix=False,
        notchance=0,
    ):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        Itereate through diff color images folder.
        images are of the format c1_A_c2_B.png, or MIRROR_c1_A_c2_B.png.
        color words do not need special brackets.
        I might include small and big in here too....!!!!
        two image dirs, one of them being the SMALLBIG one.
        """
        self.image_dir = image_dir

        self.settype = settype
        self.no_special_words = no_special_words

        self.image_filenames = [
            f
            for f in os.listdir(self.image_dir)
            if "MIRROR" not in f
            and (f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".jpg"))
        ]

        self.cutmix = cutmix
        self.notchance = notchance

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get the image file path
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load the image
        image = Image.open(img_path)

        # Extract object A, relation, and object B from the filename
        # Example filename: "A above B.png"
        img_basename = os.path.splitext(img_name)[0]  # Remove the file extension
        parts = img_basename.split("_")  # Split by space

        c1 = parts[0]
        A = parts[1]
        c2 = parts[2]
        B = parts[3]

        randomnot = random.randint(0, 100)
        # ability to make a negative caption using negatory words (e.g. Not)

        if self.settype == 0:
            caption = f"A picture of a {c1} {A} and a {c2} {B}"
            opp_caption = f"A picture of a {c2} {A} and a {c1} {B}"

            if randomnot < self.notchance * 100:

                opp_caption = f"Not a picture of a {c1} {A} and a {c2} {B}"

        else:
            caption = f"A picture of a {c2} {B} and a {c1} {A}"
            opp_caption = f"A picture of a {c1} {B} and a {c2} {A}"
            if randomnot < self.notchance * 100:
                opp_caption = f"Not a picture of a {c2} {B} and a {c1} {A}"

        return {
            "image_options": image,
            "opposite_image_options": None,
            "caption_options": caption,
            "opposite_caption_options": opp_caption,
        }


class Negation_Dataset(Dataset):
    def __init__(self, json_file, data_dir, settype=0, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file containing image pairs.
            version (int): Determines which object name is used in the caption (0 or 1).
            transform (callable, optional): Optional transform to be applied on images.
        """

        self.version = settype
        self.transform = transform
        self.data_dir = data_dir
        self.negating_words = ["and not", "but no", "without"]

        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img1_path = item["img1_path"]
        img2_path = item["img2_path"]
        if "groups" in self.data_dir:
            img1_path = img1_path.replace("\\", "/")
            img2_path = img2_path.replace("\\", "/")

        # Load images
        img1 = Image.open(os.path.join(self.data_dir, img1_path)).convert("RGB")
        img2 = Image.open(os.path.join(self.data_dir, img2_path)).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Select the correct object names based on the version
        img1_obj = item["img1_obj"][self.version]
        img2_obj = item["img2_obj"][self.version]

        # Construct the caption
        neg_choice = random.choice(self.negating_words)
        caption_options = f"An image of a {img1_obj} {neg_choice} a {img2_obj}"
        neg_choice = random.choice(self.negating_words)
        opposite_caption_options = f"An image of a {img2_obj} {neg_choice} a {img1_obj}"

        return {
            "caption_options": caption_options,
            "image_options": img1,
            "opposite_caption_options": opposite_caption_options,
            "opposite_image_options": img2,
        }
