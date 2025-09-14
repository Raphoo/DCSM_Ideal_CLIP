import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor
import os

from PIL import Image
import csv
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from whatsup_vlms.dataset_zoo.aro_datasets import (
    Controlled_Images,
    Controlled_Images_original,
    COCO_QA,
    COCO_QA_simplified,
    VG_QA,
    Flickr30k_Order,
    VG_Attribution,
    COCO_Order,
)

from src.training.toydataset_dense import COCODataset_color_nooppimg


class CrossModalConvNetwork(nn.Module):
    def __init__(self, img_seq, text_seq, hidden_dim=128, dropout_prob=0.5):
        super(CrossModalConvNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=hidden_dim // 32, kernel_size=(3, 3), padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim // 32,
            out_channels=hidden_dim // 32,
            kernel_size=(3, 3),
            padding=1,
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.dropout4 = nn.Dropout(p=dropout_prob)

        self.fc1 = nn.Linear(hidden_dim // 32 * text_seq * img_seq, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size1 = x.size(0)
        batch_size2 = x.size(1)
        x = x.unsqueeze(2)
        x = x.reshape(batch_size1 * batch_size2, 1, x.size(3), x.size(4))

        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.dropout2(F.relu(self.conv2(x)))
        x = x.view(batch_size1 * batch_size2, -1)
        x = self.dropout3(F.relu(self.fc1(x)))
        x = self.dropout4(self.fc2(x))

        return x.view(batch_size1, batch_size2, 1)


class CrossModalMLPNetwork(nn.Module):
    def __init__(self, img_seq, text_seq, hidden_dim=64, dropout_prob=0.5):
        super(CrossModalMLPNetwork, self).__init__()

        # Flattened input size
        input_dim = img_seq * text_seq

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, batch_size, img_seq, text_seq)
        """
        batch_size1 = x.size(0)
        batch_size2 = x.size(1)

        # Flatten the image and text sequence dimensions
        x = x.view(
            batch_size1 * batch_size2, -1
        )  # Shape: (bs * bs, img_seq * text_seq)

        # Apply the MLP layers
        x = self.fc1(x)  # Shape: (bs * bs, hidden_dim)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)  # Shape: (bs * bs, hidden_dim // 2)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)  # Shape: (bs * bs, 1)

        # Reshape back to (bs, bs, 1)
        x = x.view(batch_size1, batch_size2, 1)

        return x


def initialize_clip(device="cuda"):
    """Initialize CLIP model and processors"""

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.eval()
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor, tokenizer, img_processor


class PairedDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        assert len(datasetA) == len(datasetB), "Datasets must be the same size"
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        dataA = self.datasetA[index]
        dataB = self.datasetB[index]
        return dataA, dataB

    def __len__(self):
        return len(self.datasetA)


def initialize_metrics():
    """Initialize training metrics dictionaries"""
    return {
        "train_loss_list": [],
        "valid_loss_list": [],
        "train_acc_list": [],
        "valid_acc_list": [],
        "train_loss_dict": {f"dataset_{i}": [] for i in range(3)},
        "train_acc_dict": {f"dataset_{i}": [] for i in range(3)},
        "valid_loss_dict": {"blank": []},
    }


class Neg_bench(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            annotation_file (str): Path to the JSON or CSV file with annotations.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations based on file extension
        self.annotations = []

        with open(annotation_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row["image_path"].split("/")[-1]
                pos_caption = row["caption_0"]
                neg_captions = [
                    row["caption_1"],
                    row["caption_2"],
                    row["caption_3"],
                ]
                corr_ans_type = row["correct_answer_template"]

                img_path = os.path.join(self.image_dir, file_name)

                # Load image
                if os.path.exists(img_path):

                    image = Image.open(img_path).convert("RGB")
                    self.annotations.append(
                        {
                            "file_name": image,
                            "pos_caption": pos_caption,
                            "neg_captions": neg_captions,
                            "corr_ans_type": corr_ans_type,
                        }
                    )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        image = annotation["file_name"]

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        returndict = {
            "image_options": [image],
            "caption_options": [annotation["pos_caption"]] + annotation["neg_captions"],
            "corr_ans_type": annotation["corr_ans_type"],
        }
        return returndict


class CLEVR_bind(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            annotation_file (str): Path to the JSON or CSV file with annotations.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations based on file extension
        self.annotations = []
        if annotation_file.endswith(".json"):
            with open(annotation_file, "r") as f:
                raw_annotations = json.load(f)
                for entry in raw_annotations:
                    file_name = entry
                    pos_caption = raw_annotations[entry]["pos"]
                    neg_captions = raw_annotations[entry]["neg"]
                    self.annotations.append(
                        {
                            "file_name": file_name,
                            "pos_caption": pos_caption,
                            "neg_captions": neg_captions,
                        }
                    )
        elif annotation_file.endswith(".csv"):
            with open(annotation_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_name = row["file_name"]
                    pos_caption = row["pos"]
                    neg_captions = [
                        row["neg_0"],
                        row["neg_1"],
                        row["neg_2"],
                        row["neg_3"],
                    ]
                    self.annotations.append(
                        {
                            "file_name": file_name,
                            "pos_caption": pos_caption,
                            "neg_captions": neg_captions,
                        }
                    )
        else:
            raise ValueError("Unsupported annotation file format. Use .json or .csv")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        img_path = os.path.join(self.image_dir, annotation["file_name"])

        # Load image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        returndict = {
            "image_options": [image],
            "caption_options": [annotation["pos_caption"]] + annotation["neg_captions"],
        }
        return returndict


def get_test_dataloaders(root_dir):

    coco_data1 = COCO_QA_simplified(
        image_preprocess=None,
        subset="one",
        download=False,
        root_dir=root_dir,
        no_special_words=True,
    )
    coco_data2 = COCO_QA_simplified(
        image_preprocess=None,
        subset="two",
        download=False,
        root_dir=root_dir,
        no_special_words=True,
    )

    vg_data1 = VG_QA(
        image_preprocess=None, subset="one", download=False, root_dir=root_dir
    )
    vg_data2 = VG_QA(
        image_preprocess=None, subset="two", download=False, root_dir=root_dir
    )

    # we rename filenames with 3d queries to 2d queries
    rename_whatsup_files(os.path.join(root_dir, "controlled_images"))
    whatsup_controlled_A = Controlled_Images_original(
        image_preprocess=None,
        root_dir=root_dir,
        subset="A",
        download=False,
    )

    rename_whatsup_files(os.path.join(root_dir, "controlled_clevr"))
    whatsup_controlled_B = Controlled_Images_original(
        image_preprocess=None,
        root_dir=root_dir,
        subset="B",
        download=False,
    )

    # download NCD here: https://drive.google.com/drive/folders/1OohHr8L1YcEYpIEDQKP-HMQsNcXdj71_?usp=sharing
    ncd = os.path.join(root_dir, "NCD_colorful_composite_squares")
    ncd_dataset = COCODataset_color_nooppimg(
        image_dir=ncd,
        no_special_words=True,
        settype=0,
        add_spw=True,
        notchance=0,
    )
    ncd_dataset2 = COCODataset_color_nooppimg(
        image_dir=ncd,
        no_special_words=True,
        settype=1,
        add_spw=True,
        notchance=0,
    )

    def custom_collate_ncd(sample):
        img_list, text_list = [], []
        for d in sample:
            img_list.append(d["image_options"])
            if "opposite_caption_options" in d:
                text_list.append(
                    [d["caption_options"], d["opposite_caption_options"]],
                )
            else:
                text_list.append(d["caption_options"])

        sample_batch = {
            "image_options": img_list,
            "caption_options": text_list,
        }  # _flipped}
        return sample_batch

    def custom_collate_vgattr(sample):
        img_list, text_list = [], []
        for d in sample:
            img_list.append(d["image_options"][0])

            text_list += d[
                "caption_options"
            ]  # .replace("on", "above").replace("under", "below")

        # text_list_flipped = list(map(list, zip(*text_list)))
        # now the outside list is depth 2 and inside is 8

        sample_batch = {
            "image_options": img_list,
            "caption_options": text_list,
        }  # _flipped}
        return sample_batch

    ncd_dataloader = DataLoader(
        ncd_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_ncd
    )
    ncd_dataloader2 = DataLoader(
        ncd_dataset2, batch_size=1, shuffle=False, collate_fn=custom_collate_ncd
    )

    annot_header = os.path.join(root_dir, "clevr_bind")
    bind_data_two_train = CLEVR_bind(
        annotation_file=os.path.join(annot_header, "two_object", "train.csv"),
        image_dir=os.path.join(annot_header, "two_object", "images", "train"),
    )
    bind_data_two = CLEVR_bind(
        annotation_file=os.path.join(annot_header, "two_object", "val.csv"),
        image_dir=os.path.join(annot_header, "two_object", "images", "val"),
    )
    bind_data_two_gen = CLEVR_bind(
        annotation_file=os.path.join(annot_header, "two_object", "gen.csv"),
        image_dir=os.path.join(annot_header, "two_object", "images", "gen"),
    )

    def custom_collate_clevr(sample):
        img_list, text_list = [], []
        for d in sample:
            img_list.append(d["image_options"])

            text_list.append(d["caption_options"])
        sample_batch = {
            "image_options": img_list,
            "caption_options": text_list,
        }  # _flipped}
        return sample_batch

    two_obj_dataloader_train = DataLoader(
        bind_data_two_train,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_clevr,
    )
    two_obj_dataloader_val = DataLoader(
        bind_data_two, batch_size=1, shuffle=False, collate_fn=custom_collate_clevr
    )
    two_obj_dataloader_gen = DataLoader(
        bind_data_two_gen, batch_size=1, shuffle=False, collate_fn=custom_collate_clevr
    )

    vg_attr = VG_Attribution(image_preprocess=None, download=True, root_dir=root_dir)
    vg_attr_dataloader = DataLoader(
        vg_attr, batch_size=1, shuffle=False, collate_fn=custom_collate_vgattr
    )
    annot_header_neg = os.path.join(root_dir, r"NegBench\evaluation data\images")

    # put coco val2017 in data
    neg_data_coco = Neg_bench(
        annotation_file=os.path.join(
            annot_header_neg, "COCO_val_mcq_llama3.1_rephrased.csv"
        ),
        image_dir=os.path.join(root_dir, "val2017"),
    )

    neg_data_voc = Neg_bench(
        annotation_file=os.path.join(
            annot_header_neg, "VOC2007_mcq_llama3.1_rephrased.csv"
        ),
        image_dir=os.path.join(root_dir, "VOC2007", "JPEGImages"),
    )
    neg_coco_dataloader = DataLoader(
        neg_data_coco, batch_size=1, shuffle=False, collate_fn=custom_collate_clevr
    )
    neg_voc_dataloader = DataLoader(
        neg_data_voc, batch_size=1, shuffle=False, collate_fn=custom_collate_clevr
    )
    # Create dataloaders
    test_loaders = {
        "coco_spatial1": coco_data1,
        "coco_spatial2": coco_data2,
        "vg_subset1": vg_data1,
        "vg_subset2": vg_data2,
        "whatsUp_A": whatsup_controlled_A,
        "whatsUp_B": whatsup_controlled_B,
        "ncd_subset1": ncd_dataloader,
        "ncd_subset2": ncd_dataloader2,
        "clevr_bind_2obj_train": two_obj_dataloader_train,
        "clevr_bind_2obj_valid": two_obj_dataloader_val,
        "clevr_bind_2obj_gen": two_obj_dataloader_gen,
        "vg_attr": vg_attr_dataloader,
        "negbench_coco": neg_coco_dataloader,
        "negbench_voc": neg_voc_dataloader,
    }
    return test_loaders


def standardize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std


def rename_whatsup_files(directory):
    # Since we are reducing all 3d queries to 2d, we replace filenames with "on" with "above" and "under" with "below"
    # Iterate through all files in the given directory
    for filename in os.listdir(directory):
        # "under" to "below", or "on" to "above"
        
        if filename.endswith(".jpeg") and ("_on_" in filename  or "_under_" in filename):
            # Construct the full file path
            old_file_path = os.path.join(directory, filename)

            # Replace 'under' with 'below' in the filename
            new_filename = filename.replace("_on_","_above_").replace("_under_","_below_")
            new_file_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed "{filename}" to "{new_filename}"')
