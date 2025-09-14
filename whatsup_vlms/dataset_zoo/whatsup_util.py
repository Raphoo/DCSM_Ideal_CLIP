from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import os
import torch.nn.functional as F
import torch


def get_clip_text_tokens(device="cuda", textlist=None):

    CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # CLIP_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Disable gradient calculations for CLIP model
    for param in CLIP_model.parameters():
        param.requires_grad = False

    inputs = tokenizer(textlist, padding=True, return_tensors="pt")
    # now what we want to do is, pass texts through tokenizer.
    # The text will be something like ["triangle", "square", "circle", "pentagon", "star"]

    # I don't want to concatenate everything for the purposes of this test! I want SOS sentence EOS.
    # textword = " ".join(textlist)
    # inputs = tokenizer([textword], padding=True, return_tensors="pt")[
    #     "input_ids"
    # ].squeeze()
    # token_dict = {"<sos>": inputs[0].item()}
    # for id, word in enumerate(textlist):
    #     token_dict[word] = inputs[1 + id].item()
    # token_dict["<eos>"] = inputs[-1].item()

    # could return a dictionary, or a list where first and last item are sos and eos.
    return inputs


def get_clip_embeddings(device="cuda", image=None, textlist=None):

    texts = textlist

    CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # CLIP_model.to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Disable gradient calculations for CLIP model
    for param in CLIP_model.parameters():
        param.requires_grad = False

    def get_clip_embeddings_img(image):
        # we assume something is a PIL Image.
        inputs = processor(images=image, return_tensors="pt")
        # inputs = inputs.to(device)
        image_features = CLIP_model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def get_clip_embeddings_text(texts):
        inputs = tokenizer(texts, padding=True, return_tensors="pt")  # .to(device)
        text_features = CLIP_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    if image is not None:
        return get_clip_embeddings_img(image)
    if texts is not None:
        return get_clip_embeddings_text(texts)
