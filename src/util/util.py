from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import os
from PIL import Image, ImageDraw
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms

import numpy as np

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_clip_text_tokens(device="cuda", textlist=None):

    CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # CLIP_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Disable gradient calculations for CLIP model
    for param in CLIP_model.parameters():
        param.requires_grad = False
    # now what we want to do is, pass texts through tokenizer.
    # The text will be something like ["triangle", "square", "circle", "pentagon", "star"]

    inputs = tokenizer(textlist, padding=True, return_tensors="pt")

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


def flip_images(image_list, directions):
    """
    for creating 'hard negatives'
    """
    if len(image_list) != len(directions):
        raise ValueError("The length of image_list and directions must be the same.")

    flipped_image_list = []

    for img, flip_horizontal in zip(image_list, directions):
        if flip_horizontal:
            flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            flipped_image = img.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_image_list.append(flipped_image)

    return flipped_image_list


def return_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):

        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        images.append(img)
    return images


class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.01, pepper_prob=0.01):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, img):
        np_img = np.array(img).astype(np.float32)

        # Create a binary mask for "salt" (setting to 255) and "pepper" (setting to 0)
        salt_mask = np.random.rand(*np_img.shape) < self.salt_prob
        pepper_mask = np.random.rand(*np_img.shape) < self.pepper_prob

        np_img[salt_mask] = 255
        np_img[pepper_mask] = 0

        return Image.fromarray(np_img.astype(np.uint8))


def augment_all_img_lists(
    images_regular_and_flipped, allowed_flips=None, boolean=False, grid=False
):
    # the first thing is a list of shape [2, batchsize] of images
    # the second thing is a list of shape [batchsize] containing directions of allowed flips
    # let's iterate through the images and make a new augmented list of images

    new_aug_imgs = []

    for ii, all_fl in enumerate(allowed_flips):

        if not boolean:
            # print("this is what I am!")
            augmentation_transform = transforms.Compose(
                [
                    # all_fl,
                    transforms.RandomPerspective(distortion_scale=0.05, p=0.2),
                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.RandomRotation(degrees=(-4, 4)),
                    # transforms.ColorJitter(
                    #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    # ),
                    AddSaltPepperNoise(
                        salt_prob=0.02, pepper_prob=0.02
                    ),  # Salt and pepper noise
                ]
            )
        elif grid:
            # print("why would it be this?")
            augmentation_transform = transforms.Compose(
                [
                    transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.RandomRotation(degrees=(-10, 10)),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    AddSaltPepperNoise(
                        salt_prob=0.02, pepper_prob=0.02
                    ),  # Salt and pepper noise
                ]
            )
        else:
            # print("why would it be this?")
            augmentation_transform = transforms.Compose(
                [
                    all_fl,
                    transforms.RandomAffine(
                        degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                    ),
                    transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.RandomRotation(degrees=(-20, 20)),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    AddSaltPepperNoise(
                        salt_prob=0.02, pepper_prob=0.02
                    ),  # Salt and pepper noise
                ]
            )

        # if len(images_regular_and_flipped) == 2:
        #     new_aug_imgs[0].append(
        #         augmentation_transform(images_regular_and_flipped[0][ii])
        #     )
        #     new_aug_imgs[1].append(
        #         augmentation_transform(images_regular_and_flipped[1][ii])
        #     )
        # else:
        new_aug_imgs += [augmentation_transform(images_regular_and_flipped[ii])]

    return new_aug_imgs


def save_train_output_proj(
    vis_proj_state=None,
    text_proj_state=None,
    optimizer_state=None,
    epoch=0,
    loss=0,
    train_loss=None,
    valid_loss=None,
    grads=None,
    clip_model=None,
    tokenizer=None,
    savename="model.pt",
    scoremodel=None,
    external_logic_token_embeddings=None,
):

    torch.save(
        {
            "epoch": epoch,
            "vis_proj_state": vis_proj_state,
            "text_proj_state": text_proj_state,
            "optimizer_state_dict": optimizer_state,
            "clip_state_dict": clip_model,
            "clip_text_tokenizer": tokenizer,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "grads": grads,
            "scoremodel": scoremodel,
            "external_logic_token_embeddings": external_logic_token_embeddings,
        },
        savename,
    )


def save_train_output(
    model_state,
    optimizer_state,
    epoch=0,
    loss=0,
    train_loss=None,
    valid_loss=None,
    grads=None,
    clip_model=None,
    tokenizer=None,
    savename="model.pt",
    external_logic_token_embeddings=None,
    learned_lookup=None,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "clip_state_dict": clip_model,
            "clip_text_tokenizer": tokenizer,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "grads": grads,
            "external_logic_token_embeddings": external_logic_token_embeddings,
            "learned_lookup": learned_lookup,
        },
        savename,
    )

    # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model_state:
    #     print(param_tensor, "\t", model_state[param_tensor].size())

    # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer_state:
    #     print(var_name, "\t", optimizer_state[var_name])


def generate_random_tensors(num_tensors, tensor_size, similarity_threshold=0.1):
    tensors = []
    while len(tensors) < num_tensors:
        candidate = torch.randn(tensor_size).cuda()  # Add an extra dimension
        if all(
            F.cosine_similarity(candidate, tensor, dim=0).abs() < similarity_threshold
            for tensor in tensors
        ):
            tensors.append(candidate)  # Remove the extra dimension before storing
    return tensors
