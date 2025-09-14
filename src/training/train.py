from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor

import torch.nn.functional as F
import os
import torch.nn as nn
import gc
import torch

import torchvision.transforms as transforms

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.util import vis
from src.util.util import save_train_output, flip_images, augment_all_img_lists

from src.training.training_util import make_and_return_train_data, standardize
from src.util.models_and_dataloaders import CrossModalConvNetwork


def run_training(
    no_special_words=True,
    batch_size=8,
    lr=1e-4,
    epochs=100,
    losstype="CE",
    general_path=r"figures/",
    general_model_path=r"models/",
    general_data_path=r"whatsup_vlms/data/",
    single_words=False,
    l2_lambda=1e-4,
    dropout_prob=0.001,
    halfneg_bool=False,
    no_replacements=False,
):
    """
    here, we first try not replacing any of the text tokens. So no <special tokens>
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

    # STEP 2: load in train and valid sets

    combined_dataloader_train = make_and_return_train_data(
        general_data_path, halfneg_bool, batch_size
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
        dropout_prob=dropout_prob,
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(task_network.parameters()),
        lr=lr,
    )

    train_loss_list = []
    valid_loss_list = []

    train_acc_list = []
    valid_acc_list = []
    grads_dict = dict()

    train_loss_dict = dict()
    train_loss_dict["dataset_0"] = []
    train_loss_dict["dataset_1"] = []
    train_loss_dict["dataset_2"] = []

    train_acc_dict = dict()
    train_acc_dict["dataset_0"] = []
    train_acc_dict["dataset_1"] = []
    train_acc_dict["dataset_2"] = []

    valid_loss_dict = dict()
    valid_loss_dict["blank"] = []

    # Step 4: TRAIN
    for epoch in range(epochs):

        task_network.train()
        print("-----------------")
        print("Starting... Epoch:", epoch)
        print("----------------\n\n\n")

        running_loss = 0
        running_correct = 0

        running_samplect = 0
        print("train loader shape:", len(combined_dataloader_train))

        straggler = 0

        for i, batch in enumerate(combined_dataloader_train):
            optimizer.zero_grad()
            if len(batch["dataset_0"]) < batch_size:
                straggler = len(batch["dataset_0"])
                continue

            for dataset_key, data in batch.items():

                train_loss_dict[dataset_key].append(0)
                train_acc_dict[dataset_key].append(0)

                # THERE ARE TWO VERSIONS, WHERE A above B (ver1) --> B below A (ver2)
                for version in [0, 1]:
                    images = [
                        data[qq][version]["image_options"] for qq in range(batch_size)
                    ] + [
                        data[qq][version]["opposite_image_options"]
                        for qq in range(batch_size)
                    ]

                    allowed_crops = [
                        transforms.RandomResizedCrop(224, (0.96, 1.0))
                        for iii in range(len(images))
                    ]

                    # allowed_flips twice since the images are going to essentially repeat
                    images = augment_all_img_lists(images, allowed_crops)

                    texts = [
                        data[qq][version]["caption_options"] for qq in range(batch_size)
                    ] + [
                        data[qq][version]["opposite_caption_options"]
                        for qq in range(batch_size)
                    ]

                    # STEP 5:
                    # Extract patch and token level embeddings
                    vision_inputs = img_processor(
                        images=images, return_tensors="pt"
                    ).to(device)
                    vision_outputs = model.vision_model(**vision_inputs)
                    last_hidden_state = (
                        vision_outputs.last_hidden_state
                    )  # Shape: [batch_size, num_patches+1, hidden_size]
                    post_layernorm = model.vision_model.post_layernorm
                    lh1 = post_layernorm(last_hidden_state)
                    lh2 = model.visual_projection(lh1)
                    image_features = lh2 / torch.norm(lh2, p=2, dim=-1, keepdim=True)

                    if not single_words:
                        text_tokens = tokenizer(
                            text=texts,
                            padding="max_length",
                            max_length=text_seq,
                            truncation=True,
                            return_tensors="pt",
                        ).to(device)
                        text_features = model.text_model(
                            **text_tokens
                        ).last_hidden_state
                        text_features = model.text_projection(text_features)
                        text_features = text_features / text_features.norm(
                            dim=-1, keepdim=True
                        )

                    image_features = image_features.unsqueeze(1)
                    # Shape: (batch_size, 1, iseq, embed_dim)
                    text_features = text_features.unsqueeze(0)
                    # Shape: (1, batch_size, tseq, embed_dim)

                    cossim_mat = torch.einsum(
                        "bqie,lpte->bpit", image_features, text_features
                    ).to(device)
                    # shape: bs,bs,img_seq,textseq
                    # so the second batch dim should be the text dim.
                    # but we can double check.

                    # STEP 6:
                    # Add special words! TO THE COSSIM MATRIX.
                    # so

                    if not no_replacements:
                        new_cossim_mat = []
                        for k in range(len(texts)):
                            # 0th dim is # of images. 1st dim is # of texts. so we can index in that dimension here.
                            # we want to replace
                            # add the special embedding to the correct location in the final output.
                            sent = texts[k].split(" ")
                            # find places where kth sentence has special words.
                            temp_k_cossim = cossim_mat[:, k, :, :]

                            # num_valid = len(sent)

                            for sp_w in test_special_words:
                                if sp_w in sent:
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
                                            temp_k_cossim[:, :, special_location + 1 :],
                                        ],
                                        dim=-1,
                                    )

                            new_cossim_mat.append(temp_k_cossim)
                        cossim_mat = torch.stack(new_cossim_mat, dim=1)
                        # this is now shape bs_i x bs_t x i_seq x t_seq+1 , theoretically.

                    # let's imshow to confirm
                    cossim_mat = standardize(cossim_mat)

                    shuffle_bs1 = torch.randperm(cossim_mat.shape[0])
                    shuffle_bs2 = torch.randperm(cossim_mat.shape[1])

                    cossim_mat = cossim_mat[shuffle_bs1]

                    cossim_mat = cossim_mat[:, shuffle_bs2]

                    task_output = (
                        task_network(cossim_mat).to(device).squeeze()
                    )  # shape bs x bs x 1.
                    quicklabels = torch.eye(task_output.shape[0]).to(device)
                    quicklabels = quicklabels[shuffle_bs1]
                    quicklabels = quicklabels[:, shuffle_bs2]

                    loss = criterion(task_output, quicklabels) + criterion(
                        task_output.squeeze().t(), quicklabels.t()
                    )

                    # if l2_lambda > 0:
                    #     l2_reg = sum(param.norm(2) for param in model.parameters())
                    #     loss += l2_lambda * l2_reg

                    running_loss += loss.item()
                    train_loss_dict[dataset_key][-1] = (
                        train_loss_dict[dataset_key][-1] + loss.item()
                    )

                    loss.backward()

                    with torch.no_grad():

                        num_correct = torch.sum(
                            torch.argmax(task_output, dim=0)
                            == torch.argmax(quicklabels, dim=0)
                        ) + torch.sum(
                            torch.argmax(task_output, dim=1)
                            == torch.argmax(quicklabels, dim=1),
                        )

                        num_correct = num_correct.item() / 2

                        running_correct += num_correct
                        print("Training Accuracy:", num_correct / task_output.shape[0])
                        print("----------")
                        train_acc_dict[dataset_key][-1] = (
                            train_acc_dict[dataset_key][-1]
                            + num_correct / task_output.shape[0]
                        )
                        running_samplect += task_output.shape[0]

            for param in task_network.parameters():
                if param.grad is not None:
                    param.grad.data /= 3
            optimizer.step()

            print("----------")
            print("Training Loss:", loss.item())
            print("----------")

            del cossim_mat, task_output, quicklabels, loss
            gc.collect()
            torch.cuda.empty_cache()

        total_epochs = len(combined_dataloader_train) - int(straggler == 0)
        # number of total epochs is not the maximum if last batch was too small

        train_loss_list.append(running_loss / (2 * total_epochs))
        train_acc_list.append(running_correct / running_samplect)

        for dataset_key in train_loss_dict:
            train_loss_dict[dataset_key][-1] / (2 / 3 * total_epochs)
            train_acc_dict[dataset_key][-1] / (running_samplect / 3)

        if (epoch + 1) % 3 == 0 or epoch in [1, 2]:
            vis.plot_training_justlist(
                train_loss_list,
                valid_loss_list,
                train_acc_list,
                valid_acc_list,
                total_epochs=epoch,
                losstype=losstype,
                savename=os.path.join(
                    general_path,
                    "patchnet_MANY"
                    + str(epoch)
                    + "_batchsize_"
                    + str(batch_size)
                    + "_lr_"
                    + str(lr)
                    + "_drop_"
                    + str(dropout_prob)
                    + "_lambda_"
                    + str(l2_lambda)
                    + "_singlew_"
                    + str(single_words)
                    + "_fix_objaverse_std_NOSPECIAL_v6.png",
                ),
            )
            vis.plot_training_lossparts(
                train_loss_dict,
                train_acc_dict,
                total_epochs=epoch,
                losstype=losstype,
                savename=os.path.join(
                    general_path,
                    "patchnet_MANY"
                    + str(epoch)
                    + "_batchsize_"
                    + str(batch_size)
                    + "_lr_"
                    + str(lr)
                    + "_drop_"
                    + str(dropout_prob)
                    + "_lambda_"
                    + str(l2_lambda)
                    + "_singlew_"
                    + str(single_words)
                    + "_fix_objaverse_std_lossparts_NOSPECIAL_v6.png",
                ),
            )
            save_train_output(
                model_state=task_network.state_dict(),
                optimizer_state=optimizer.state_dict(),
                clip_model=model.state_dict(),
                tokenizer=tokenizer,
                epoch=epoch,
                train_loss=train_loss_list,
                valid_loss=valid_loss_list,
                external_logic_token_embeddings=external_logic_token_embedding,
                grads=grads_dict,
                savename=os.path.join(
                    general_model_path,
                    "patchnet_MANY"
                    + str(epoch)
                    + "_spw_"
                    + str(no_special_words)
                    + "_batchsize_"
                    + str(batch_size)
                    + "_lr_"
                    + str(lr)
                    + "_drop_"
                    + str(dropout_prob)
                    + "_lambda_"
                    + str(l2_lambda)
                    + "_singlew_"
                    + str(single_words)
                    + "_fix_objaverse_std_NOSPECIAL_v6.pt",
                ),
            )
