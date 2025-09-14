import pdb
import os
import json
import subprocess
import torch

import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from torchvision.datasets.utils import download_url

from .perturbations import TextShuffler
from .constants import ARO_ROOT, COCO_ROOT, FLICKR_ROOT
from .retrieval import pre_caption

from .whatsup_util import get_clip_embeddings, get_clip_text_tokens


class VG_Relation(Dataset):
    def __init__(
        self,
        image_preprocess,
        text_perturb_fn=None,
        image_perturb_fn=None,
        root_dir=ARO_ROOT,
        download=False,
    ):
        """
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        root_dir: Directory for the VG-R dataset.
        download: Whether to download the dataset if it does not exist.
        """
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_relation.json")
        image_dir = os.path.join(root_dir, "VG_100K")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Relation could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        if not os.path.exists(annotation_file):
            subprocess.call(
                [
                    "gdown",
                    "--id",
                    "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3",
                    "--output",
                    annotation_file,
                ]
            )

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)

        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert("RGB")
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop(
            (
                test_case["bbox_x"],
                test_case["bbox_y"],
                test_case["bbox_x"] + test_case["bbox_w"],
                test_case["bbox_y"] + test_case["bbox_h"],
            )
        )

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict(
            {"image_options": [image], "caption_options": [true_caption, false_caption]}
        )
        return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 1
        metrics["Accuracy"] = np.mean(correct_mask)

        all_relations = np.array(self.all_relations)

        result_records = []
        # Log the accuracy of all relations
        for relation in np.unique(all_relations):
            relation_mask = all_relations == relation
            if relation_mask.sum() == 0:
                continue
            result_records.append(
                {
                    "Relation": relation,
                    "Accuracy": correct_mask[relation_mask].mean(),
                    "Count": relation_mask.sum(),
                    "Dataset": "Visual Genome Relation",
                }
            )
        return result_records


class VG_Attribution(Dataset):
    def __init__(
        self,
        image_preprocess,
        text_perturb_fn=None,
        image_perturb_fn=None,
        root_dir=ARO_ROOT,
        download=False,
    ):
        """
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        root_dir: Directory for the VG-A dataset.
        """
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_attribution.json")
        image_dir = os.path.join(root_dir, "VG_100K")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Attribution could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        if not os.path.exists(annotation_file):
            subprocess.call(
                [
                    "gdown",
                    "--id",
                    "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw",
                    "--output",
                    annotation_file,
                ]
            )

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)

        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])

        # Set of attributes in each test case
        self.all_attributes = [
            f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset
        ]
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert("RGB")
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop(
            (
                test_case["bbox_x"],
                test_case["bbox_y"],
                test_case["bbox_x"] + test_case["bbox_w"],
                test_case["bbox_y"] + test_case["bbox_h"],
            )
        )

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict(
            {
                "image_options": [image],
                "caption_options": [
                    true_caption,
                    false_caption,
                ],
            }
        )
        return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 1
        result_records = []
        all_attributes = np.array(self.all_attributes)
        for attr in np.unique(all_attributes):
            attr_mask = all_attributes == attr
            if attr_mask.sum() < 25:
                continue
            result_records.append(
                {
                    "Attributes": attr,
                    "Accuracy": correct_mask[attr_mask].mean(),
                    "Count": attr_mask.sum(),
                    "Dataset": "Visual Genome Attribution",
                }
            )
        return result_records


class COCO_Order(Dataset):
    def __init__(
        self,
        image_preprocess=None,
        root_dir=COCO_ROOT,
        max_words=30,
        split="test",
        image_perturb_fn=None,
        download=False,
    ):
        """
        COCO Order Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: not used; for compatibility.
        download: Whether to download the dataset if it does not exist.
        """
        shuffler = TextShuffler()
        perturb_functions = [
            shuffler.shuffle_nouns_and_adj,
            shuffler.shuffle_allbut_nouns_and_adj,
            shuffler.shuffle_within_trigrams,
            shuffler.shuffle_trigrams,
        ]

        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            print("Directory for COCO could not be found!")
            if download:
                print("Downloading COCO now.")
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        urls = {
            "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json",
            "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json",
        }
        filenames = {"val": "coco_karpathy_val.json", "test": "coco_karpathy_test.json"}
        download_url(urls[split], root_dir)

        self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), "r"))
        self.image_preprocess = image_preprocess
        self.image_root = root_dir

        self.test_cases = []

        for img_id, ann in tqdm(enumerate(self.annotation)):
            for i, caption in enumerate(ann["caption"]):
                test_case = {}
                test_case["image"] = ann["image"]
                test_case["caption_options"] = [pre_caption(caption, max_words)]

                for perturb_fn in perturb_functions:
                    test_case["caption_options"].append(
                        pre_caption(perturb_fn(caption), max_words)
                    )
                self.test_cases.append(test_case)

    def __len__(self):
        return len(self.test_cases)

    def __getitem__(self, index):
        test_case = self.test_cases[index]
        image_path = os.path.join(self.image_root, test_case["image"])

        image = Image.open(image_path).convert("RGB")
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        item = edict(
            {"image_options": [image], "caption_options": test_case["caption_options"]}
        )
        return item

    def download(self):
        import subprocess

        os.makedirs(self.root_dir, exist_ok=True)
        # subprocess.call(["wget", "http://images.cocodataset.org/zips/train2014.zip"], cwd=self.root_dir)
        # subprocess.call(["unzip", "train2014.zip"], cwd=self.root_dir)

        subprocess.call(
            ["wget", "http://images.cocodataset.org/zips/val2014.zip"],
            cwd=self.root_dir,
        )
        subprocess.call(["unzip", "val2014.zip"], cwd=self.root_dir)

        subprocess.call(
            ["wget", "http://images.cocodataset.org/zips/test2014.zip"],
            cwd=self.root_dir,
        )
        subprocess.call(["unzip", "test2014.zip"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T  # Make it N_ims x N_text

        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 0
        records = [{"Precision@1": np.mean(correct_mask)}]
        return records


class Flickr30k_Order(Dataset):
    def __init__(
        self,
        image_preprocess,
        split,
        root_dir=FLICKR_ROOT,
        max_words=30,
        *args,
        **kwargs,
    ):
        """
        image_preprocess: image preprocessing function
        split: 'val' or 'test'
        root_dir: The directory of the flickr30k images. This should contain the `flickr30k-images` directory that \
            contains all the images. 
        """
        urls = {
            "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json",
            "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json",
        }
        filenames = {"val": "flickr30k_val.json", "test": "flickr30k_test.json"}
        if not os.path.exists(root_dir):
            print("Directory for Flickr30k could not be found!")
            flickr_url = "https://forms.illinois.edu/sec/229675"
            raise RuntimeError(
                f"You need to manually sign up and download the dataset from {flickr_url} and place it in the `root_dir`."
            )

        download_url(urls[split], root_dir)

        self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), "r"))
        self.image_preprocess = image_preprocess
        self.root_dir = root_dir

        self.test_cases = []

        shuffler = TextShuffler()
        perturb_functions = [
            shuffler.shuffle_nouns_and_adj,
            shuffler.shuffle_allbut_nouns_and_adj,
            shuffler.shuffle_within_trigrams,
            shuffler.shuffle_trigrams,
        ]
        for img_id, ann in tqdm(enumerate(self.annotation)):
            for i, caption in enumerate(ann["caption"]):
                test_case = {}
                test_case["image"] = ann["image"]
                test_case["caption_options"] = [pre_caption(caption, max_words)]

                for perturb_fn in perturb_functions:
                    test_case["caption_options"].append(
                        pre_caption(perturb_fn(caption), max_words)
                    )
                self.test_cases.append(test_case)

    def __len__(self):
        return len(self.test_cases)

    def __getitem__(self, index):
        test_case = self.test_cases[index]
        sepp = test_case["image"].split("/")
        sepp[0] = sepp[0].replace("-", "_")
        image_path = os.path.join(self.root_dir, sepp[0], sepp[1])
        image = Image.open(image_path).convert("RGB")

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        item = edict(
            {"image_options": [image], "caption_options": test_case["caption_options"]}
        )
        return item

    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T  # Make it N_ims x N_text
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 0
        result_records = [{"Precision@1": np.mean(correct_mask)}]
        return result_records


class Controlled_Images_original(Dataset):
    def __init__(
        self,
        image_preprocess,
        text_perturb_fn=None,
        image_perturb_fn=None,
        root_dir=ARO_ROOT,
        download=False,
        subset="A",
    ):
        self.root_dir = root_dir
        if subset == "A":
            annotation_file = os.path.join(root_dir, "controlled_images_dataset.json")
            image_dir = os.path.join(root_dir, "controlled_images")

            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images A could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError(
                        "Please either download the dataset by letting `--download` or specify the correct directory."
                    )

            if not os.path.exists(annotation_file):
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm",
                        "--output",
                        annotation_file,
                    ]
                )

        else:
            annotation_file = os.path.join(root_dir, "controlled_clevr_dataset.json")
            image_dir = os.path.join(root_dir, "controlled_clevr")
            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images B could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError(
                        "Please either download the dataset by letting `--download` or specify the correct directory."
                    )

            if not os.path.exists(annotation_file):
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z",
                        "--output",
                        annotation_file,
                    ]
                )

        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == "A":
            for d in self.dataset:
                if "left_of" in d["image_path"]:
                    self.all_prepositions.append("left_of")
                elif "right_of" in d["image_path"]:
                    self.all_prepositions.append("right_of")
                elif "_above_" in d["image_path"]:
                    self.all_prepositions.append("above")
                else:
                    self.all_prepositions.append("below")
            self.eval_dict = {
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                ): {"left": 0, "right": 0, "above": 0, "below": 0}
                for d in self.dataset
            }
            self.pred_dict = {
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                ): {"left": "", "right": "", "above": "", "below": ""}
                for d in self.dataset
            }

        else:
            for d in self.dataset:
                if "left_of" in d["image_path"]:
                    self.all_prepositions.append("left_of")
                elif "right_of" in d["image_path"]:
                    self.all_prepositions.append("right_of")
                elif "_in-front_of_" in d["image_path"]:
                    self.all_prepositions.append("in-front_of")
                else:
                    self.all_prepositions.append("behind")
            self.eval_dict = {
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                ): {"left": 0, "right": 0, "in-front": 0, "behind": 0}
                for d in self.dataset
            }
            self.pred_dict = {
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                ): {"left": "", "right": "", "in-front": "", "behind": ""}
                for d in self.dataset
            }

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(
            os.path.join(
                self.root_dir,
                test_case["image_path"][5:]
                .replace("_on", "_above")
                .replace("_under", "_below"),
            )
        ).convert("RGB")
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        test_case["caption_options"] = [
            test_case["caption_options"][k]
            .replace(" on", " above")
            .replace(" under", " below")
            for k in range(len(test_case["caption_options"]))
        ]

        item = edict(
            {"image_options": [image], "caption_options": test_case["caption_options"]}
        )
        return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "controlled_images.tar.gz")
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["tar", "-xvf", "controlled_images.tar.gz"], cwd=self.root_dir)
        image_zip_file = os.path.join(self.root_dir, "controlled_clevr.tar.gz")
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["tar", "-xvf", "controlled_clevr.tar.gz"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 4, i.e. first caption is right, next three captions are wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 0
        metrics["Accuracy"] = np.mean(correct_mask)
        print("Individual accuracy: {}".format(metrics["Accuracy"] * 100))

        prepositions = ["above", "below", "front", "behind", "left", "right"]
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        for i, d in enumerate(self.dataset):
            prep = list(
                set(prepositions).intersection(
                    set(d["caption_options"][preds[i]].split())
                )
            )
            gold_prep = list(
                set(prepositions).intersection(set(d["caption_options"][0].split()))
            )
            # if len(prep) != 1 or len(gold_prep)!=1:
            #    pdb.set_trace()
            #    print("?")
            prep = prep[0]
            gold_prep = gold_prep[0]
            prep_counts[gold_prep][prep] += 1

            self.pred_dict[
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                )
            ][d["image_path"].split("/")[-1].split("_")[1]] = prep
        # print(prep_counts)
        for d, correct in zip(self.dataset, correct_mask):
            self.eval_dict[
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                )
            ][d["image_path"].split("/")[-1].split("_")[1]] = correct

        pair_correct = 0
        set_correct = 0
        for obj_pair, correct_dict in self.eval_dict.items():
            if correct_dict["left"] and correct_dict["right"]:
                pair_correct += 1
            if self.subset == "A":
                if correct_dict["above"] and correct_dict["below"]:
                    pair_correct += 1
            else:
                if correct_dict["in-front"] and correct_dict["behind"]:
                    pair_correct += 1
            if sum(correct_dict.values()) == 4:
                set_correct += 1
        pair_accuracy = pair_correct * 100 / (len(self.dataset) / 2)
        set_accuracy = set_correct * 100 / (len(self.dataset) / 4)
        print("Pair accuracy: {}".format(pair_accuracy))
        print("Set accuracy: {}".format(set_accuracy))
        all_prepositions = np.array(self.all_prepositions)

        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = all_prepositions == prepositions
            if prepositions_mask.sum() == 0:
                continue
            result_records.append(
                {
                    "Preposition": prepositions,
                    "Accuracy": correct_mask[prepositions_mask].mean(),
                    "Count": prepositions_mask.sum(),
                    "Dataset": "Controlled Images - {}".format(self.subset),
                }
            )
        return result_records


class Controlled_Images(Dataset):
    def __init__(
        self,
        image_preprocess,
        text_perturb_fn=None,
        image_perturb_fn=None,
        root_dir=ARO_ROOT,
        download=False,
        subset="A",
        clip_model=None,
        wrapped_encoders=None,
        special_tokens=None,
        tokenizer=None,
        img_processor=None,
        no_special_words=False,
        settype=0,
        add_spw=False,
    ):
        """
        settype: if 1, then above/left, and 2 then
        addspw: do we add the spw? or remove the old one and replace?
        """
        self.root_dir = root_dir
        self.add_spw = add_spw
        if subset == "A":
            annotation_file = os.path.join(root_dir, "controlled_images_dataset.json")
            image_dir = os.path.join(root_dir, "controlled_images")

            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images A could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError(
                        "Please either download the dataset by letting `--download` or specify the correct directory."
                    )

            if not os.path.exists(annotation_file):
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm",
                        "--output",
                        annotation_file,
                    ]
                )

        else:
            annotation_file = os.path.join(root_dir, "controlled_clevr_dataset.json")
            image_dir = os.path.join(root_dir, "controlled_clevr")
            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images B could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError(
                        "Please either download the dataset by letting `--download` or specify the correct directory."
                    )

            if not os.path.exists(annotation_file):
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z",
                        "--output",
                        annotation_file,
                    ]
                )

        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == "A":
            for d in self.dataset:
                if "left_of" in d["image_path"]:
                    self.all_prepositions.append("left")  # _of")
                elif "right_of" in d["image_path"]:
                    self.all_prepositions.append("right")  # _of")
                elif "_on_" in d["image_path"]:
                    self.all_prepositions.append("on")
                else:
                    self.all_prepositions.append("under")
            self.eval_dict = {
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                ): {"left": 0, "right": 0, "on": 0, "under": 0}
                for d in self.dataset
            }
            self.pred_dict = {
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                ): {"left": "", "right": "", "on": "", "under": ""}
                for d in self.dataset
            }

        else:
            for d in self.dataset:
                if "left_of" in d["image_path"]:
                    self.all_prepositions.append("left")  # _of")
                elif "right_of" in d["image_path"]:
                    self.all_prepositions.append("right")  # _of")
                elif "_in-front_of_" in d["image_path"]:
                    self.all_prepositions.append("in front")  # _of")
                else:
                    self.all_prepositions.append("behind")
            self.eval_dict = {
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                ): {"left": 0, "right": 0, "in-front": 0, "behind": 0}
                for d in self.dataset
            }
            self.pred_dict = {
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                ): {"left": "", "right": "", "in-front": "", "behind": ""}
                ## NOTE: I changed in-front to "front"
                for d in self.dataset
            }

        self.image_preprocess = image_preprocess

        self.wrapped_encoders = wrapped_encoders
        self.clip_model = clip_model
        self.special_tokens = special_tokens

        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.no_special_words = no_special_words
        self.settype = settype

        # TODO test 8/21. Below is item list for set A.
        items_list = [
            "apple",
            "armchair.jpeg",
            "ball-of-yarn",
            "banana",
            "banjo",
            "beer-bottle",
            "blanket",
            "book",
            "bottle",
            "bottle-of-mustard",
            "bottle-of-orange-juice",
            "bowl",
            "box",
            "bread",
            "can",
            "cap",
            "chair",
            "chair.jpeg",
            "chair2",
            "cup",
            "dog",
            "dragonfruit",
            "glass",
            "glasses",
            "glove",
            "hat",
            "headphones",
            "helmet",
            "kettle",
            "knife",
            "lamp",
            "laptop",
            "lemon",
            "microphone",
            "mug",
            "notepad",
            "orange",
            "oven-mitt",
            "painting",
            "pan",
            "phone",
            "photograph",
            "pillow",
            "plant",
            "plate",
            "pot",
            "remote",
            "rose",
            "scarf",
            "scissors",
            "shoes",
            "soda-can",
            "spatula",
            "sunglasses",
            "table.jpeg",
            "tape",
            "toilet-roll",
            "toy-cactus",
            "vase",
            "water-bottle",
            "water-filter",
            "wineglass",
            "yarn",
        ]

        furniture_list = ["armchair", "chair", "table"]

        self.allowed_positions = ["above", "below", "left", "right"]
        if self.settype == 0:
            self.positions_in_captions = ["above", "left_of"]
            self.positions_extra = ["below", "right_of"]
        else:
            self.positions_extra = ["above", "left_of"]
            self.positions_in_captions = ["below", "right_of"]

        self.data = []
        inverse_pairs = []
        for item in items_list:
            for furniture in furniture_list:

                for kk in range(2):
                    # first check that the file names are actually in the directory.
                    fn0 = item + "_" + self.positions_in_captions[kk] + "_" + furniture
                    fn1 = item + "_" + self.positions_extra[kk] + "_" + furniture

                    if os.path.exists(
                        os.path.join(root_dir, "controlled_images", fn0 + ".jpeg")
                    ) and os.path.exists(
                        os.path.join(root_dir, "controlled_images", fn1 + ".jpeg")
                    ):

                        inverse_pairs.append(
                            [
                                fn0,
                                fn1,
                            ]
                        )

                        fn0_caption = "A photo of a " + (
                            fn0.replace("-", " ")
                            .replace("_", " ")
                            .replace("leftof", "to the left of")
                            .replace("rightof", "to the right of")
                        )
                        fn1_caption = "A photo of a " + (
                            fn1.replace("-", " ")
                            .replace("_", " ")
                            .replace("leftof", "to the left of")
                            .replace("rightof", "to the right of")
                        )

                        if not no_special_words:

                            if self.add_spw:
                                for pp in range(4):
                                    if self.allowed_positions[pp] in fn0:
                                        index0 = fn0_caption.find(
                                            self.allowed_positions[pp]
                                        )
                                        fn0_caption = (
                                            fn0_caption[:index0]
                                            + "<"
                                            + self.allowed_positions[pp]
                                            + ">"
                                            + " "
                                            + fn0_caption[index0:]
                                        )

                                    elif self.allowed_positions[pp] in fn1:
                                        index1 = fn1_caption.find(
                                            self.allowed_positions[pp]
                                        )
                                        fn1_caption = (
                                            fn1_caption[:index1]
                                            + "<"
                                            + self.allowed_positions[pp]
                                            + ">"
                                            + " "
                                            + fn1_caption[index1:]
                                        )
                            else:
                                for pp in range(4):
                                    if self.allowed_positions[pp] in fn0:
                                        fn0_caption = fn0_caption.replace(
                                            self.allowed_positions[pp],
                                            "<" + self.allowed_positions[pp] + ">",
                                        )
                                    elif self.allowed_positions[pp] in fn1:
                                        fn1_caption = fn1_caption.replace(
                                            self.allowed_positions[pp],
                                            "<" + self.allowed_positions[pp] + ">",
                                        )

                        self.data.append(((fn0, fn1), (fn0_caption, fn1_caption)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_names, captions = self.data[index]
        images = []
        for img_name in img_names:
            image_path = os.path.join(
                self.root_dir, "controlled_images", img_name + ".jpeg"
            )
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        return {"image_options": images, "caption_options": captions}
        # NOTE: EDIT HERE!!!!
        # test_case = self.dataset[index]
        # # print(test_case)
        # test_case["image_path"] = os.path.join(
        #     self.root_dir[:-4], test_case["image_path"]
        # )
        # image = Image.open(test_case["image_path"]).convert("RGB")
        # # if self.image_preprocess is not None:
        # #     image = self.image_preprocess(image)

        # # ok, the way this works is - the first is the true caption. the rest are false.
        # # for the sake of this test, let's just take the true + random opposite.

        # texts = test_case["caption_options"]

        # sample = {"img": image, "text": test_case["caption_options"]}
        # # oh, ok. So sample text is 2 things.
        # # img_features, text_features, text_tokens = (
        # #     self.wrapped_encoders.get_embeddings_for_sample(sample)
        # # )

        # # text_tokens = get_clip_text_tokens(textlist=test_case["caption_options"])
        # # img_features = get_clip_embeddings(image=image)
        # # text_features = get_clip_embeddings(textlist=test_case["caption_options"])

        # same_replace = {"on": "above", "under": "below"}

        # # I want to replace all instances of "on" --> "above", and so on
        # # PART A: replace that in texts

        # for new_word in same_replace:
        #     for jj in range(4):
        #         if new_word in texts[jj]:
        #             texts[jj].replace(new_word, same_replace[new_word])
        # # newtexts = []
        # # for sentence in texts:
        # #     for new_word in same_replace:
        # #         new_sentence = sentence.replace(new_word, same_replace[new_word])
        # #         if new_sentence != sentence:
        # #             newtexts.append(new_sentence)

        # # texts = newtexts

        # # PART B: replace that in self.all_prepositions

        # for kk in range(len(self.all_prepositions)):
        #     if self.all_prepositions[kk] in same_replace:
        #         self.all_prepositions[kk] = same_replace[self.all_prepositions[kk]]

        # opposite = {
        #     "left": "right",
        #     "right": "left",
        #     "above": "below",
        #     "below": "above",
        #     "top": "bottom",
        #     "bottom": "top",
        #     "behind": "in front",
        #     "in front": "behind",
        #     "on": "under",
        #     "under": "on",
        # }
        # # STEP: add <tokens>
        # sp_word = self.all_prepositions[index]
        # sp_word_opp = opposite[sp_word]
        # # so there are two items in texts. each of them needs this modification.

        # ## TODO 8/21 adding a modification, where I want to load in pairs of images that are hard negatives of each other.
        # ## This means, I want the two images and two text pairings to be part of one batch.

        # if not self.no_special_words:

        #     for ll in range(4):
        #         for sp_word in opposite:
        #             if sp_word in texts[ll]:
        #                 sp_word_ind = texts[ll].index(sp_word)
        #                 texts[ll] = " ".join(
        #                     [
        #                         texts[ll][
        #                             : sp_word_ind - 1
        #                         ],  # minus 1 to get rid of double space
        #                         "<" + sp_word + ">",
        #                         texts[ll][sp_word_ind:],
        #                     ]
        #                 )
        # # sp_word_opp_ind = texts[1].index(sp_word_opp)

        # # texts[1] = " ".join(
        # #     [
        # #         texts[1][:sp_word_opp_ind],
        # #         "<" + sp_word_opp + ">",
        # #         texts[1][sp_word_opp_ind:],
        # #     ]
        # # )

        # item = edict(
        #     {
        #         "image_options": image,
        #         "caption_options": texts,
        #     }
        #     ##TODO: finish ^
        #     # most sus atm: phrase_tokens, will it add properly?
        #     # actually, turns out "caption_embeddings is useless."
        #     # what we want is: index of special token (one of the special words)
        #     # so that upon tokenization, I can just replace the initial embedding for that word with my network's nn.embedding.
        #     # and what that special token is
        # )
        # return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "controlled_images.tar.gz")
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["tar", "-xvf", "controlled_images.tar.gz"], cwd=self.root_dir)
        image_zip_file = os.path.join(self.root_dir, "controlled_clevr.tar.gz")
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["tar", "-xvf", "controlled_clevr.tar.gz"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 4, i.e. first caption is right, next three captions are wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 0
        metrics["Accuracy"] = np.mean(correct_mask)
        print("Individual accuracy: {}".format(metrics["Accuracy"] * 100))

        prepositions = ["on", "under", "front", "behind", "left", "right"]
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        for i, d in enumerate(self.dataset):
            prep = list(
                set(prepositions).intersection(
                    set(d["caption_options"][preds[i]].split())
                )
            )
            gold_prep = list(
                set(prepositions).intersection(set(d["caption_options"][0].split()))
            )
            # if len(prep) != 1 or len(gold_prep)!=1:
            #    pdb.set_trace()
            #    print("?")
            prep = prep[0]
            gold_prep = gold_prep[0]
            prep_counts[gold_prep][prep] += 1

            self.pred_dict[
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                )
            ][d["image_path"].split("/")[-1].split("_")[1]] = prep
        # print(prep_counts)
        for d, correct in zip(self.dataset, correct_mask):
            self.eval_dict[
                (
                    d["image_path"].split("/")[-1].split("_")[0],
                    d["image_path"].split("/")[-1].split("_")[-1][:-5],
                )
            ][d["image_path"].split("/")[-1].split("_")[1]] = correct

        pair_correct = 0
        set_correct = 0
        for obj_pair, correct_dict in self.eval_dict.items():
            if correct_dict["left"] and correct_dict["right"]:
                pair_correct += 1
            if self.subset == "A":
                if correct_dict["on"] and correct_dict["under"]:
                    pair_correct += 1
            else:
                if correct_dict["in-front"] and correct_dict["behind"]:
                    pair_correct += 1
            if sum(correct_dict.values()) == 4:
                set_correct += 1
        pair_accuracy = pair_correct * 100 / (len(self.dataset) / 2)
        set_accuracy = set_correct * 100 / (len(self.dataset) / 4)
        print("Pair accuracy: {}".format(pair_accuracy))
        print("Set accuracy: {}".format(set_accuracy))
        all_prepositions = np.array(self.all_prepositions)

        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = all_prepositions == prepositions
            if prepositions_mask.sum() == 0:
                continue
            result_records.append(
                {
                    "Preposition": prepositions,
                    "Accuracy": correct_mask[prepositions_mask].mean(),
                    "Count": prepositions_mask.sum(),
                    "Dataset": "Controlled Images - {}".format(self.subset),
                }
            )
        return result_records


class COCO_QA(Dataset):
    def __init__(
        self,
        image_preprocess,
        text_perturb_fn=None,
        image_perturb_fn=None,
        root_dir=ARO_ROOT,
        download=False,
        subset="one",
        wrapped_encoders=None,
        special_tokens=None,
        clip_model=None,
        tokenizer=None,
        img_processer=None,
        processor=None,
        pad=50,
    ):
        """
        Args:
            clip_model: altered clip model with updated embeddings to reflect task_network
            tokenizer: appropriate tokenizer
        """
        self.root_dir = root_dir
        if subset == "one":
            annotation_file = os.path.join(root_dir, "coco_qa_one_obj.json")
            image_dir = os.path.join(root_dir, "val2017")
        else:
            annotation_file = os.path.join(root_dir, "coco_qa_two_obj.json")
            image_dir = os.path.join(root_dir, "val2017")
        if not os.path.exists(image_dir):
            print("Image directory for COCO-QA could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        if not os.path.exists(annotation_file):
            if subset == "one":
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq",
                        "--output",
                        annotation_file,
                    ]
                )
            else:
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s",
                        "--output",
                        annotation_file,
                    ]
                )

        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == "one":
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if "left of" in d[1]:
                    self.all_prepositions.append("left")
                elif "right of" in d[1]:
                    self.all_prepositions.append("right")
                elif "above" in d[1]:
                    self.all_prepositions.append("above")
                else:
                    self.all_prepositions.append("below")
        self.image_preprocess = image_preprocess

        self.wrapped_encoders = wrapped_encoders
        self.special_tokens = special_tokens

        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.img_processor = img_processer
        self.pad = pad

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(
            os.path.join(
                self.root_dir, "val2017/{}.jpg".format(str(test_case[0]).zfill(12))
            )
        ).convert("RGB")
        # if self.image_preprocess is not None:
        #     image = self.image_preprocess(image)

        # here, I want to add the options: "image_embeddings" and "caption_embeddings"
        # EDITED BY RAPHI: to add embeddings.

        # wait, why are the text and image pairings garbage?!

        # if self.subset == "two":
        # we know index [-1] and [4] contains the two words.
        # well, sometimes the concepts are two words long... so we have to do something slightly smarter
        # obj1_singleword = True
        # obj2_singleword = True
        # for k in test_case[1:]:
        #     k_words = k.split(" ")
        #     if k_words[5] in ["to", "above", "below"]:
        #         obj1 = k_words[4]
        #     else:
        #         obj1 = " ".join(k_words[4:6])
        #         obj1_singleword = False

        #     if k_words[-2] in ["a", "an"]:
        #         obj2 = k_words[-1]
        #     else:
        #         obj2 = " ".join(k_words[-2:])
        #         obj2_singleword = False

        #     k_words[-1] = obj1
        #     if not obj2_singleword:
        #         try:
        #             k_words.pop(-2)
        #         except:
        #             pass

        #     k_words[4] = obj2
        #     if not obj1_singleword:
        #         try:
        #             k_words.pop(5)
        #         except:
        #             pass

        #     newphrase = " ".join(k_words)
        #     test_case.append(newphrase)

        # well. I don't like using the wrapped encoder, so I will go back to my old ways.

        # oh, ok. So sample text is 2 things.
        # img_features, text_features, text_tokens = (
        #     self.wrapped_encoders.get_embeddings_for_sample(sample)
        # )

        texts = test_case[1:]

        opposite = {
            "left": "right",
            "right": "left",
            "above": "below",
            "below": "above",
            "top": "bottom",
            "bottom": "top",
        }
        # before I do this, let's manually add the "< word >" into the strings.
        sp_word = self.all_prepositions[index]
        sp_word_opp = opposite[sp_word]
        # so there are two items in texts. each of them needs this modification.

        sp_word_ind = texts[0].index(sp_word)
        texts[0] = " ".join(
            [texts[0][: sp_word_ind - 1], "<" + sp_word + ">", texts[0][sp_word_ind:]]
        )
        sp_word_opp_ind = texts[1].index(sp_word_opp)
        texts[1] = " ".join(
            [
                texts[1][: sp_word_opp_ind - 1],
                "<" + sp_word_opp + ">",
                texts[1][sp_word_opp_ind:],
            ]
        )

        # now the new tokenizer should be able to add the task_network embedding.

        ## TODO: 6/27, let's try not getting any embeddings from the dataloader...:P
        text_tokens = self.tokenizer(text=texts, padding=True, return_tensors="pt").to(
            self.clip_model.device
        )
        img_inputs = self.img_processor(images=image, return_tensors="pt").to(
            self.clip_model.device
        )

        # text_outputs = self.text_model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        # pooled_output = text_outputs[1]
        # text_features = self.text_projection(pooled_output)

        # return text_features

        # image_features = self.clip_model.get_image_features(
        #     **img_inputs, output_hidden_states=True
        # )

        image_features = self.clip_model.vision_model(**img_inputs).last_hidden_state
        # text_features = self.clip_model.text_model(**text_tokens).last_hidden_state

        # can I project?
        image_features = self.clip_model.visual_projection(image_features)
        # text_features = self.clip_model.text_projection(text_features)

        # last_hidden_state shape is 2 x 14 x 512.
        # if you grab the last channel, it's 2 x 512!
        # taking the 1th element is what CLIP does to grab EOS token. But I am not doing that.

        # text_features = self.clip_model.get_text_features(
        #     **text_tokens, output_hidden_states=True
        # )

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # further, let's pad text_features -- 1st row is number of tokens, this can be padded to 50 to be same as img.

        # pad = (0, 0, 0, self.pad - text_features.shape[1], 0, 0)

        # Pad the tensor
        # text_features = torch.nn.functional.pad(text_features, pad)

        # img_features = get_clip_embeddings(image=image)
        # text_features = get_clip_embeddings(textlist=test_case[1:])

        # so text_features and text_tokens should be lists of length 2, each element being each phrase.

        phrases_as_words = [p_.split(" ") for p_ in texts]

        # I actually think special_indices doesn't matter?
        # outdated.

        # special_indices = []
        # special_indices.append(self.special_tokens.index(self.all_prepositions[index]))
        # special_indices.append(
        #     self.special_tokens.index(opposite[self.all_prepositions[index]])
        # )

        # special_indices += special_indices
        # # there are special words like A B A B, labels are 1, 0, 0, 1
        # # I think this doubling up is artifactual of when I tried to make new sentences, not sure.

        # token_location_in_phrase = []
        # token_location_in_phrase.append(
        #     phrases_as_words[0].index(self.all_prepositions[index])
        # )
        # token_location_in_phrase.append(
        #     phrases_as_words[1].index(opposite[self.all_prepositions[index]])
        # )

        # well, this takes to be the case that all words are 1 token.
        # which is not actually true!
        # TODO: address this. 6/17
        # not a hard fix, I just need to run the special word through the tokenizer, replace or precede those tokens.

        # token_location_in_phrase += token_location_in_phrase
        # same thing repeated twice because we assume the sentences are samely formatted :P

        # token_location_in_phrase is also a list of length 2, with 2 indices.

        # phrase_tokens is a phraselength-1(for special word) list. Then we add SOS and EOS.

        # print(
        #     "caption shape:",
        #     len(test_case[1:]),
        #     "\ntextfeat shape:",
        #     text_features.shape,
        # )
        # print(
        #     "img opt shape:",
        #     img_inputs["pixel_values"].shape,
        #     "\nimg embed shape:",
        #     image_features.shape,
        # )

        if sp_word in ["left", "right"]:
            flip_horiz = True
        else:
            flip_horiz = False

        item = edict(
            {
                "image_options": img_inputs,
                "caption_options": texts,  # test_case[1:],
                # "image_embeddings": image_features,
                "caption_tokens": text_tokens,
                "flip_dir": flip_horiz,
                # "caption_embeddings": text_features,
                # "special_indices": special_indices,
                # "token_loc_in_phrase": token_location_in_phrase,
                # "phrase_tokens": text_tokens,
            }
            ##TODO: finish ^
            # most sus atm: will the gradients propagate through the dataloader?? we will see:P
            # actually, turns out "caption_embeddings is useless."
            # what we want is: index of special token (one of the special words)
            # so that upon tokenization, I can just replace the initial embedding for that word with my network's nn.embedding.
            # and what that special token is
        )
        # print(
        #     "caption embed shapes:",
        #     text_features.shape,
        #     "phrase_tokens shapes:",
        #     text_tokens["input_ids"].shape,
        # )
        return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "val2017.zip")
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "1zp5vBRRM4_nSik6o9PeVspDvOsHgPT4l",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["unzip", "val2017.zip"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 0
        metrics["Accuracy"] = np.mean(correct_mask)
        print(metrics["Accuracy"] * 100)

        all_prepositions = np.array(self.all_prepositions)

        prepositions = list(set(self.all_prepositions))
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {
            "left": "right",
            "right": "left",
            "above": "below",
            "below": "above",
            "top": "bottom",
            "bottom": "top",
        }

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        # print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = all_prepositions == prepositions
            if prepositions_mask.sum() == 0:
                continue
            result_records.append(
                {
                    "Preposition": prepositions,
                    "Accuracy": correct_mask[prepositions_mask].mean(),
                    "Count": prepositions_mask.sum(),
                    "Dataset": "COCO-QA {}-object".format(self.subset),
                }
            )
        return result_records


class COCO_QA_simplified(Dataset):
    def __init__(
        self,
        root_dir=ARO_ROOT,
        download=False,
        subset="one",
        special_tokens=None,
        image_preprocess=None,
        no_special_words=False,
    ):
        """
        Args:
            clip_model: altered clip model with updated embeddings to reflect task_network
            tokenizer: appropriate tokenizer

            no_special_words: if True, then do not alter text.
        """
        self.root_dir = root_dir
        if subset == "one":
            annotation_file = os.path.join(root_dir, "coco_qa_one_obj.json")
            image_dir = os.path.join(root_dir, "val2017")
        else:
            annotation_file = os.path.join(root_dir, "coco_qa_two_obj.json")
            image_dir = os.path.join(root_dir, "val2017")
        if not os.path.exists(image_dir):
            print("Image directory for COCO-QA could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        if not os.path.exists(annotation_file):
            if subset == "one":
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq",
                        "--output",
                        annotation_file,
                    ]
                )
            else:
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s",
                        "--output",
                        annotation_file,
                    ]
                )

        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == "one":
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if "left of" in d[1]:
                    self.all_prepositions.append("left")
                elif "right of" in d[1]:
                    self.all_prepositions.append("right")
                elif "above" in d[1]:
                    self.all_prepositions.append("above")
                else:
                    self.all_prepositions.append("below")

        self.special_tokens = special_tokens
        self.image_preprocess = image_preprocess
        self.no_special_words = no_special_words

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(
            os.path.join(
                self.root_dir, "val2017/{}.jpg".format(str(test_case[0]).zfill(12))
            )
        ).convert("RGB")
        # image = self.image_preprocess(image)

        texts = test_case[1:]

        opposite = {
            "left": "right",
            "right": "left",
            "above": "below",
            "below": "above",
            "top": "bottom",
            "bottom": "top",
        }
        # before I do this, let's manually add the "< word >" into the strings.
        sp_word = self.all_prepositions[index]
        sp_word_opp = opposite[sp_word]
        # so there are two items in texts. each of them needs this modification.

        if not self.no_special_words:
            sp_word_ind = texts[0].index(sp_word)
            texts[0] = " ".join(
                [
                    texts[0][: sp_word_ind - 1],
                    "<" + sp_word + ">",
                    texts[0][sp_word_ind:],
                ]
            )
            sp_word_opp_ind = texts[1].index(sp_word_opp)
            texts[1] = " ".join(
                [
                    texts[1][: sp_word_opp_ind - 1],
                    "<" + sp_word_opp + ">",
                    texts[1][sp_word_opp_ind:],
                ]
            )

        if sp_word in ["left", "right"]:
            flip_horiz = True
        else:
            flip_horiz = False

        item = edict(
            {"image_options": image, "caption_options": texts, "flip_dir": flip_horiz}
        )

        return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "val2017.zip")
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "1zp5vBRRM4_nSik6o9PeVspDvOsHgPT4l",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["unzip", "val2017.zip"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 0
        metrics["Accuracy"] = np.mean(correct_mask)
        print(metrics["Accuracy"] * 100)

        all_prepositions = np.array(self.all_prepositions)

        prepositions = list(set(self.all_prepositions))
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {
            "left": "right",
            "right": "left",
            "above": "below",
            "below": "above",
            "top": "bottom",
            "bottom": "top",
        }

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        # print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = all_prepositions == prepositions
            if prepositions_mask.sum() == 0:
                continue
            result_records.append(
                {
                    "Preposition": prepositions,
                    "Accuracy": correct_mask[prepositions_mask].mean(),
                    "Count": prepositions_mask.sum(),
                    "Dataset": "COCO-QA {}-object".format(self.subset),
                }
            )
        return result_records


class VG_QA(Dataset):
    def __init__(
        self,
        image_preprocess,
        text_perturb_fn=None,
        image_perturb_fn=None,
        root_dir=ARO_ROOT,
        download=False,
        subset="one",
    ):
        self.root_dir = root_dir
        if subset == "one":
            annotation_file = os.path.join(root_dir, "vg_qa_one_obj.json")
            image_dir = os.path.join(root_dir, "vg_images")
        else:
            annotation_file = os.path.join(root_dir, "vg_qa_two_obj.json")
            image_dir = os.path.join(root_dir, "vg_images")
        if not os.path.exists(image_dir):
            print("Image directory for VG-QA could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        if not os.path.exists(annotation_file):
            if subset == "one":
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1ARMRzRdohs9QTr1gpIfzyUzvW20wYp_p",
                        "--output",
                        annotation_file,
                    ]
                )
            else:
                subprocess.call(
                    [
                        "gdown",
                        "--id",
                        "1sjVG5O3QMY8s118k7kQM8zzDZH12i_95",
                        "--output",
                        annotation_file,
                    ]
                )

        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == "one":
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if "left of" in d[1]:
                    self.all_prepositions.append("left")
                elif "right of" in d[1]:
                    self.all_prepositions.append("right")
                elif "front of" in d[1]:
                    self.all_prepositions.append("front")
                elif "behind" in d[1]:
                    self.all_prepositions.append("behind")
                else:
                    self.all_prepositions.append("top")
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(
            os.path.join(self.root_dir, "vg_images/{}.jpg".format(test_case[0]))
        ).convert("RGB")
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        item = edict(
            {"image_options": [image], "caption_options": [test_case[1], test_case[2]]}
        )
        return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(
            r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\whatsup_vlms\data",
            "vg_images.tar.gz",
        )  # self.root_dir
        subprocess.call(
            [
                "gdown",
                "--no-cookies",
                "1idW7Buoz7fQm4-670n-oERw9U-2JLJvE",
                "--output",
                image_zip_file,
            ]
        )
        subprocess.call(["tar", "-xvf", "vg_images.tar.gz"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = preds == 0
        metrics["Accuracy"] = np.mean(correct_mask)
        print(metrics["Accuracy"] * 100)

        all_prepositions = np.array(self.all_prepositions)

        prepositions = list(set(self.all_prepositions)) + ["below", "bottom", "front"]
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {
            "left": "right",
            "right": "left",
            "behind": "front",
            "front": "behind",
            "above": "below",
            "below": "above",
            "bottom": "top",
            "top": "bottom",
        }

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        # print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = all_prepositions == prepositions
            if prepositions_mask.sum() == 0:
                continue
            result_records.append(
                {
                    "Preposition": prepositions,
                    "Accuracy": correct_mask[prepositions_mask].mean(),
                    "Count": prepositions_mask.sum(),
                    "Dataset": "VG-QA {}-object".format(self.subset),
                }
            )
        return result_records


def get_visual_genome_relation(
    image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False
):
    return VG_Relation(
        image_preprocess=image_preprocess,
        text_perturb_fn=text_perturb_fn,
        image_perturb_fn=image_perturb_fn,
        download=download,
    )


def get_visual_genome_attribution(
    image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False
):
    return VG_Attribution(
        image_preprocess=image_preprocess,
        text_perturb_fn=text_perturb_fn,
        image_perturb_fn=image_perturb_fn,
        download=download,
    )


def get_controlled_images_a(
    image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False
):
    return Controlled_Images(
        image_preprocess=image_preprocess,
        text_perturb_fn=text_perturb_fn,
        image_perturb_fn=image_perturb_fn,
        download=download,
        subset="A",
    )


def get_controlled_images_b(
    image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False
):
    return Controlled_Images(
        image_preprocess=image_preprocess,
        text_perturb_fn=text_perturb_fn,
        image_perturb_fn=image_perturb_fn,
        download=download,
        subset="B",
    )


def get_coco_qa_one_obj(
    image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False
):
    return COCO_QA(
        image_preprocess=image_preprocess,
        text_perturb_fn=text_perturb_fn,
        image_perturb_fn=image_perturb_fn,
        download=download,
        subset="one",
    )


def get_coco_qa_two_obj(
    image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False
):
    return COCO_QA(
        image_preprocess=image_preprocess,
        text_perturb_fn=text_perturb_fn,
        image_perturb_fn=image_perturb_fn,
        download=download,
        subset="two",
    )


def get_vg_qa_one_obj(
    image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False
):
    return VG_QA(
        image_preprocess=image_preprocess,
        text_perturb_fn=text_perturb_fn,
        image_perturb_fn=image_perturb_fn,
        download=download,
        subset="one",
    )


def get_vg_qa_two_obj(
    image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False
):
    return VG_QA(
        image_preprocess=image_preprocess,
        text_perturb_fn=text_perturb_fn,
        image_perturb_fn=image_perturb_fn,
        download=download,
        subset="two",
    )


def get_coco_order(
    image_preprocess,
    image_perturb_fn,
    text_perturb_fn,
    max_words=30,
    download=False,
    root_dir=COCO_ROOT,
    split="test",
):
    return COCO_Order(
        root_dir=root_dir,
        split=split,
        image_preprocess=image_preprocess,
        image_perturb_fn=image_perturb_fn,
        max_words=max_words,
        download=download,
    )


def get_flickr30k_order(
    image_preprocess,
    image_perturb_fn,
    text_perturb_fn,
    max_words=30,
    download=False,
    root_dir=FLICKR_ROOT,
    split="test",
):
    return Flickr30k_Order(
        root_dir=root_dir,
        split=split,
        image_preprocess=image_preprocess,
        image_perturb_fn=image_perturb_fn,
        max_words=max_words,
        download=download,
    )
