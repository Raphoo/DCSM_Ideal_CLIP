import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor
import torch.nn.functional as F
import os
import torch.nn as nn
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.util.models_and_dataloaders import CrossModalConvNetwork, standardize


def get_dcsm_scores(pretrain_model_dir=None, images=None, texts=None):
    """
    Try evaluating your own images and texts with DCSMs.
    """

    # STEP 1:
    # Load the pre-trained CLIP model and processor
    device = "cuda"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model = model.eval()
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

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

    with torch.no_grad():
        external_logic_token_embedding.weight = nn.Parameter(
            (F.normalize(external_logic_token_embedding.weight, p=2, dim=1) + 1) / 3
        )

    # Set up network n optimizer
    image_seq = 14 * 14 + 1
    text_seq = 30
    hidden_dim = 128

    # Initialize the network
    task_network = CrossModalConvNetwork(
        img_seq=image_seq,
        text_seq=text_seq,
        hidden_dim=hidden_dim,
        dropout_prob=0,
    ).to(device)

    loaded = torch.load(pretrain_model_dir, weights_only=False)

    task_network.load_state_dict(loaded["model_state_dict"])
    external_logic_token_embedding = loaded["external_logic_token_embeddings"]

    task_network.eval()
    model.eval()

    # Extract patch and token level embeddings
    vision_inputs = img_processor(images=images, return_tensors="pt").to(device)
    vision_outputs = model.vision_model(**vision_inputs)
    last_hidden_state = (
        vision_outputs.last_hidden_state
    )  # Shape: [batch_size, num_patches+1, hidden_size]
    post_layernorm = model.vision_model.post_layernorm
    lh1 = post_layernorm(last_hidden_state)
    lh2 = model.visual_projection(lh1)
    image_features = lh2 / torch.norm(lh2, p=2, dim=-1, keepdim=True)

    text_tokens = tokenizer(
        text=texts,
        padding="max_length",
        max_length=text_seq,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    text_features = model.text_model(**text_tokens).last_hidden_state
    text_features = model.text_projection(text_features)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_features = image_features.unsqueeze(1)
    # Shape: (batch_size, 1, iseq, embed_dim)
    text_features = text_features.unsqueeze(0)
    # Shape: (1, batch_size, tseq, embed_dim)

    cossim_mat = torch.einsum("bqie,lpte->bpit", image_features, text_features).to(
        device
    )

    new_cossim_mat = []
    for k in range(len(texts)):
        sent = texts[k].split(" ")
        temp_k_cossim = cossim_mat[:, k, :, :]

        for sp_w in test_special_words:
            if sp_w in sent:
                print(sp_w)
                special_location = sent.index(sp_w)
                chosen_w = test_special_words.index(sp_w)
                mid = external_logic_token_embedding(
                    torch.tensor([chosen_w]).to(model.device)
                )
                mid = mid.unsqueeze(2).repeat(len(images), 1, 1)
                temp_k_cossim = torch.concat(
                    [
                        temp_k_cossim[:, :, :special_location],
                        mid,
                        temp_k_cossim[:, :, special_location:-1],
                    ],
                    dim=-1,
                )

        new_cossim_mat.append(temp_k_cossim)

    cossim_mat = torch.stack(new_cossim_mat, dim=1)
    # this is now shape bs_i x bs_t x i_seq x t_seq+1 , theoretically.
    cossim_mat = standardize(cossim_mat)

    task_output = task_network(cossim_mat).to(device).squeeze()  # shape bs x bs x 1.
    return task_output


def get_naive_clip_scores(
    images: list[Image.Image],
    texts: list[str],
    patchsize=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Compute cosine similarity between a list of PIL images and a list of text strings using CLIP.

    Args:
        images (list of PIL.Image): The input images.
        texts (list of str): The input text strings.
        device (str): 'cuda' or 'cpu'.

    Returns:s
        torch.Tensor: A similarity matrix of shape (num_images, num_texts)
    """

    # Load pretrained CLIP model and processor
    model = CLIPModel.from_pretrained(f"openai/clip-vit-base-patch{patchsize}").to(
        device
    )
    model.eval()
    processor = CLIPProcessor.from_pretrained(f"openai/clip-vit-base-patch{patchsize}")

    # Preprocess and encode images
    image_inputs = processor(images=images, return_tensors="pt", padding=True).to(
        device
    )
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = F.normalize(image_features, dim=-1)

    # Preprocess and encode text
    text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)

    similarity_matrix = image_features @ text_features.T

    return similarity_matrix


if __name__ == "__main__":
    torch.manual_seed(15)

    images = [Image.open(r"D:\Ideal_CLIP\data\image_0004.jpg")]
    texts = ["white dog, black cat", "black dog, white cat"]
    pretrain_model_dir = r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\models\patchnet_MANY26_spw_True_batchsize_8_lr_0.001_drop_0.1_lambda_0_singlew_Falsehalfneg_False_fix_objaverse_std_wneg_v5.pt"
    out = get_dcsm_scores(pretrain_model_dir, images, texts)
    print(out)
