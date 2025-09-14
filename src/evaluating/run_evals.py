# Simplified, modular version of the script

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    CLIPImageProcessor,
    AutoProcessor,
    AutoModel,
    FlavaProcessor,
    FlavaModel,
)

import open_clip
from PIL import Image
from lavis.models import load_model_and_preprocess


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.util.models_and_dataloaders import CrossModalConvNetwork, standardize

from whatsup_vlms.dataset_zoo.aro_datasets import (
    Controlled_Images_original,
    COCO_QA_simplified,
    VG_QA,
)

from src.util.models_and_dataloaders import get_test_dataloaders


def evaluate_all_models(model_dict, root_dir, csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_test_dataloaders(root_dir)

    # for testing:
    # dataloaders = {"whatsUp_B": dataloaders["whatsUp_B"]}

    for model_name_text, (model_file, text_seq) in model_dict.items():
        model, tokenizer, img_processor = load_tokenizer_and_processors(
            model_name_text, device
        )

        task_network = CrossModalConvNetwork(img_seq=14 * 14 + 1, text_seq=text_seq).to(
            device
        )

        if "NAIVE" not in model_name_text:

            state = torch.load(model_file, map_location=device)
            model.load_state_dict(
                state.get("clip_state_dict", model.state_dict()), strict=False
            )
            task_network.load_state_dict(
                state.get("model_state_dict", task_network.state_dict()), strict=False
            )
            task_network.eval()

            if "learned_lookup" in state:
                if state["learned_lookup"] is not None:
                    test_special_words = state["learned_lookup"]
                    print(test_special_words)
                else:
                    test_special_words = [
                        "above",
                        "below",
                        "left",
                        "right",
                        "small",
                        "big",
                        "not",
                        "no",
                        "without",
                        "but",
                        "absent",
                    ]
            else:
                test_special_words = [
                    "above",
                    "below",
                    "left",
                    "right",
                    "small",
                    "big",
                    "not",
                    "no",
                    "without",
                    "but",
                    "absent",
                ]
            external_logic_token_embedding = nn.Embedding(
                len(test_special_words), 14 * 14 + 1
            ).to(device)
            external_logic_token_embedding = state.get(
                "external_logic_token_embeddings", external_logic_token_embedding
            )

        if (
            "NAIVE" in model_name_text
            or "scoring" in model_name_text
            or "SigLIP" in model_name_text
            or "CoCa" in model_name_text
            or "OpenCLIP" in model_name_text
            or "BLIP" in model_name_text
            or "FLAVA" in model_name_text
        ):
            results = evaluate_single_model(
                model, tokenizer, img_processor, dataloaders, device, model_name_text
            )
        else:
            results = evaluate_patch_model(
                model,
                task_network,
                test_special_words,
                tokenizer,
                img_processor,
                dataloaders,
                external_logic_token_embedding,
                device,
                model_name_text,
                text_seq,
            )
        save_results_to_csv({model_name_text: results}, model_name_text, csv_path)


def save_results_to_csv(results, model_name_text, csv_path):
    file_exists = os.path.isfile(csv_path)
    headers = ["Model Name"] + sorted({key for res in results.values() for key in res})

    if file_exists:
        with open(csv_path, mode="r", newline="") as file:
            reader = csv.reader(file)
            rows = list(reader)
            existing_data = {row[0]: row[1:] for row in rows[1:] if row}
            existing_headers = rows[0]
            headers = list(set(headers + existing_headers))
    else:
        existing_data = {}

    row = [model_name_text] + [results[model_name_text].get(h, "") for h in headers[1:]]
    existing_data[model_name_text] = row[1:]

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for name, data in existing_data.items():
            writer.writerow([name] + data)


def evaluate_patch_model(
    model,
    task_network,
    test_special_words,
    tokenizer,
    img_processor,
    dataloaders,
    external_logic_token_embedding,
    device,
    model_name_text,
    text_seq,
):
    results = {}
    model.eval()
    for name, loader in dataloaders.items():

        frontadder = ""
        prefix_needed = ["whats", "clevr", "vg_attr"]
        if "clevr" in name:
            frontadder = "An image of a "
        elif sum([pp in name for pp in prefix_needed]) > 0:
            frontadder = "An image of "
        correct = 0
        total = 0

        test_count = 0
        for batch in loader:
            # test_count += 1
            # if test_count > 5:
            #     break
            images = (
                [batch["image_options"]]
                if not isinstance(batch["image_options"], list)
                else (
                    batch["image_options"][0]
                    if isinstance(batch["image_options"][0], list)
                    else batch["image_options"]
                )
            )
            texts = (
                [batch["caption_options"]]
                if not isinstance(batch["caption_options"], list)
                else (
                    batch["caption_options"][0]
                    if isinstance(batch["caption_options"][0], list)
                    else batch["caption_options"]
                )
            )

            texts = [
                frontadder
                # replacing 3d functional words with proximal 2d words
                + ttt.replace(" behind ", " above ")
                .replace(" in front of ", " below ")
                .replace(" bottom", " below")
                .replace(" top", " above")
                for ttt in texts
            ]

            # print(name, texts)

            vision_inputs = img_processor(images=images, return_tensors="pt").to(device)
            vision_outputs = model.vision_model(**vision_inputs)
            vision_feats = model.visual_projection(
                model.vision_model.post_layernorm(vision_outputs.last_hidden_state)
            )
            image_features = vision_feats / vision_feats.norm(dim=-1, keepdim=True)

            text_tokens = tokenizer(
                texts,
                padding="max_length",
                max_length=text_seq,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            text_features = model.text_projection(
                model.text_model(**text_tokens).last_hidden_state
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = image_features.unsqueeze(1)
            text_features = text_features.unsqueeze(0)
            cossim_mat = torch.einsum(
                "bqie,lpte->bpit", image_features, text_features
            ).to(device)

            if "NO FUNCTIONAL ROWS" not in model_name_text:
                new_cossim_mat = []
                for k, txt in enumerate(texts):
                    sent = txt.split(" ")
                    temp_k_cossim = cossim_mat[:, k, :, :]
                    for sp_w in test_special_words:
                        if sp_w in sent:
                            # print(sp_w)
                            idx = sent.index(sp_w)
                            mid = external_logic_token_embedding(
                                torch.tensor([test_special_words.index(sp_w)]).to(
                                    device
                                )
                            )
                            mid = mid.unsqueeze(2).repeat(len(images), 1, 1)
                            temp_k_cossim = torch.cat(
                                [
                                    temp_k_cossim[:, :, :idx],
                                    mid,
                                    temp_k_cossim[:, :, idx + 1 :],
                                ],
                                dim=-1,
                            )

                            temp_k_cossim = temp_k_cossim[:, :, :text_seq]
                    new_cossim_mat.append(temp_k_cossim)
                cossim_mat = torch.stack(new_cossim_mat, dim=1)

            # if "standardize" in model_name_text:
            #automatically standardize the cossim matrix
            cossim_mat = standardize(cossim_mat)

            task_output = task_network(cossim_mat).squeeze()
            predicted = torch.argmax(task_output).item()
            # plt.imshow(images[0])
            # plt.title("\n".join(texts))
            # plt.show()
            # print(task_output, predicted)
            correct += predicted == 0
            total += 1

        results[name] = correct / total if total > 0 else 0.0
        print(
            f"Model: {model_name_text} | Dataset: {name} | Accuracy: {results[name]:.4f}"
        )
    return results


def evaluate_single_model(
    model, tokenizer, img_processor, dataloaders, device, model_name_text
):
    results = {}
    model.eval()
    model = model.to(device)
    for name, loader in dataloaders.items():
        correct = 0
        total = 0
        frontadder = ""
        prefix_needed = ["whats", "clevr", "vg_attr"]
        if "clevr" in name:
            frontadder = "An image of a "
        elif sum([pp in name for pp in prefix_needed]) > 0:
            frontadder = "An image of "

        with torch.no_grad():
            test_count = 0
            for batch in loader:
                # test_count += 1
                # if test_count > 2:
                #     break
                images = (
                    [batch["image_options"]]
                    if not isinstance(batch["image_options"], list)
                    else (
                        batch["image_options"][0]
                        if isinstance(batch["image_options"][0], list)
                        else batch["image_options"]
                    )
                )
                texts = (
                    [batch["caption_options"]]
                    if not isinstance(batch["caption_options"], list)
                    else (
                        batch["caption_options"][0]
                        if isinstance(batch["caption_options"][0], list)
                        else batch["caption_options"]
                    )
                )

                texts = [frontadder + ttt for ttt in texts]
                # print(name, texts)

                if (
                    "OpenCLIP" in model_name_text
                    or "BLIP" in model_name_text
                    or "CoCa" in model_name_text
                    or "NegCLIP" in model_name_text
                ):  # e.g. for OpenCLIP preprocess
                    image_inputs = torch.stack(
                        [
                            (
                                img_processor(im)
                                if isinstance(img_processor(im), torch.Tensor)
                                else img_processor(im)["pixel_values"][0]
                            )
                            for im in images
                        ]
                    ).to(device)
                    if "BLIP" in model_name_text:
                        text_inputs = [tokenizer(sent) for sent in texts]

                        sample = {"image": image_inputs, "text_input": text_inputs}
                        image_features = model.extract_features(
                            sample, mode="image"
                        ).image_embeds[
                            :, 0, :
                        ]  # size (img_opt, 768)
                        text_features = model.extract_features(
                            sample, mode="text"
                        ).text_embeds[
                            :, 0, :
                        ]  # size (text_opt, 768)
                    else:
                        text_inputs = (
                            tokenizer(texts).to(device)
                            if hasattr(tokenizer, "__call__")
                            else tokenizer(texts, return_tensors="pt").to(device)
                        )
                        image_features = model.encode_image(image_inputs)
                        text_features = model.encode_text(text_inputs)
                else:
                    inputs = img_processor(images=images, return_tensors="pt").to(
                        device
                    )
                    text_inputs = tokenizer(
                        texts, padding=True, return_tensors="pt"
                    ).to(device)
                    image_features = model.get_image_features(**inputs)
                    text_features = model.get_text_features(**text_inputs)

                img_emb = F.normalize(image_features, dim=-1)
                txt_emb = F.normalize(text_features, dim=-1)
                sim = img_emb @ txt_emb.T
                pred = sim.argmax(dim=1).item()
                correct += pred == 0
                total += 1
        results[name] = correct / total if total > 0 else 0.0
        print(
            f"Model: {model_name_text} | Dataset: {name} | Accuracy: {results[name]:.4f}"
        )
    return results


def load_tokenizer_and_processors(model_name_text, device):
    if "SigLIP" in model_name_text:
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        return model.eval(), processor.tokenizer, processor.image_processor

    elif "OpenCLIP" in model_name_text or "NegCLIP" in model_name_text:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        if "NegCLIP" in model_name_text:
            # Load custom weights for NegCLIP
            checkpoint_path = os.path.join(
                r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\negclip.pth"
            )
            state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
            model.load_state_dict(state_dict, strict=False)
        return model.eval(), tokenizer, preprocess

    elif "CoCa" in model_name_text:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "coca_ViT-B-32", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        tokenizer = open_clip.get_tokenizer("coca_ViT-B-32")
        return model.eval(), tokenizer, preprocess

    elif "BLIP" in model_name_text:
        model, vis_processors, txt_processors = load_model_and_preprocess(
            "blip_feature_extractor", model_type="base", is_eval=True, device=device
        )
        return model.eval(), txt_processors["eval"], vis_processors["eval"]

    elif "FLAVA" in model_name_text:
        processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
        return model.eval(), processor.tokenizer, processor.image_processor

    else:
        model_name = (
            "openai/clip-vit-base-patch32"
            if "VIT-B32" in model_name_text
            else "openai/clip-vit-base-patch16"
        )
        print("loading model...")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model.eval(), processor.tokenizer, processor.image_processor
