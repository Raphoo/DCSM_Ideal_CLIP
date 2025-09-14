import matplotlib.pyplot as plt
import torch
import numpy as np
import os


def visualize_prediction(pred, gt, savename, threshold=0.5):
    plt.figure()
    # actually, include a black and white one that's thresholded.
    # so thresholded
    if threshold != 0:
        pred = torch.sigmoid(pred)
    thresh = pred.detach().cpu() >= threshold

    pred_gt = torch.stack([pred.detach().cpu(), thresh, gt.detach().cpu()])

    plt.imshow(pred_gt, aspect="auto", cmap="gray")
    plt.colorbar()
    plt.ylabel("Gt, Thresholded_pred, Pred")
    plt.xlabel("Batch dimension")
    plt.title("Pred_Label")
    plt.tight_layout()
    plt.savefig(savename + "_preds.png")
    plt.close()


def visualize_whole_set(pred_and_labels, savename):
    """
    Args:
        pred_and_labels (list): nested list of
    """

    pred_and_labels_ = np.array(
        pred_and_labels.detach().cpu()
    )  # this will be of shape set size x 2.
    # let's plot.
    pred_and_labels_ = np.stack(
        [pred_and_labels_[:, 0] for ii in range(50)]
        + [pred_and_labels_[:, 1] for ii in range(50)]
    )
    plt.figure()
    # plt.scatter(np.arange(len(pred_and_labels_)), pred_and_labels_[:, 0])
    # plt.scatter(np.arange(len(pred_and_labels_)), pred_and_labels_[:, 1])

    plt.imshow(pred_and_labels_)
    plt.axis("off")
    plt.ylabel("Preds | Label")
    plt.axhline(50, c="black")
    plt.xlabel("data points")
    plt.colorbar()
    plt.title("All Predictions shown against GT")
    plt.tight_layout()
    plt.savefig(savename + "_allpreds.png")

    plt.close()


def visualize_batch(iter, sample_batch, savename=r"figures\batch_"):
    """
    Create visualization for images in batch, phrase it is being compared with, and label.
    Args:
        iter (int): iteration being visualized
        sample_batch (dict): batch from dataloader
        savename (str, optional): save directory
    """
    batchsize = len(sample_batch["image"])
    oneside = int(np.ceil(batchsize**0.5))
    # figure arrangement dependent on batchsize
    plt.figure(figsize=(oneside * 2, oneside * 2))
    for i in range(batchsize):
        plt.subplot(oneside, oneside, i + 1)
        plt.title(
            "Phrase:\n"
            + sample_batch["phrase"][i]
            + "\nLabel:"
            + str(sample_batch["label"][i].item())
            + "\nimage_idx:"
            + str(sample_batch["image_idx"][i].item())
            + "\nphrase_idx"
            + str(sample_batch["phrase_idx"][i].item())
        )
        plt.imshow(sample_batch["image"][i])

    plt.tight_layout()
    plt.savefig(savename + str(iter) + ".png")
    plt.close()

    plt.figure(figsize=(oneside * 2, oneside * 2))
    for i in range(batchsize):
        plt.subplot(oneside, oneside, i + 1)
        plt.title(
            "for model - Phrase:\n"
            + sample_batch["phrase"][i]
            + "\nLabel:"
            + str(sample_batch["label"][i].item())
            + "\nimage_idx:"
            + str(sample_batch["image_idx"][i].item())
            + "\nphrase_idx"
            + str(sample_batch["phrase_idx"][i].item())
        )
        plt.imshow(sample_batch["image_formodel"][i])

    plt.tight_layout()
    plt.savefig(savename + str(iter) + "formodel.png")
    plt.close()


def visualize_objaverse(
    iter,
    sample_batch,
    images_regular_and_flipped,
    savename=r"figures\cocobatch_",
    pathadd="",
    nameadd="",
):
    nameadd = str(nameadd)
    batchsize = len(sample_batch["image_options" + nameadd])
    oneside = int(np.ceil(batchsize**0.5))
    # figure arrangement dependent on batchsize
    plt.figure(figsize=(oneside * 5, oneside * 5))
    for i in range(0, batchsize):
        plt.subplot(oneside, oneside, i + 1)
        plt.imshow(images_regular_and_flipped[i])
        plt.title(sample_batch["caption_options" + nameadd][i])

    plt.tight_layout()
    plt.savefig(savename + str(iter) + pathadd + "_ver" + nameadd + ".png")
    plt.close()
    return


def visualize_whatsup_data(
    iter,
    sample_batch,
    images_regular_and_flipped,
    savename=r"figures\cocobatch_",
    pathadd="",
    nameadd="",
):
    """
    Create visualization for images in batch, phrase it is being compared with, and label.
    Args:
        iter (int): iteration being visualized
        sample_batch (dict): batch from dataloader
        savename (str, optional): save directory
    """
    nameadd = str(nameadd)
    batchsize = len(sample_batch["image_options" + nameadd]) * 2
    oneside = int(np.ceil(batchsize**0.5))
    # figure arrangement dependent on batchsize
    plt.figure(figsize=(oneside * 5, oneside * 5))
    for i in range(0, batchsize, 2):
        plt.subplot(oneside, oneside, i + 1)
        if len(sample_batch["caption_options" + nameadd]) == 2:
            labels = [1, 0]
        else:
            labels = [1, 0, 0, 1]

        ind = i // 2
        plt.title(
            "Two Phrases:\n"
            + "\n".join(
                [
                    sample_batch["caption_options" + nameadd][l][ind]
                    for l in range(len(sample_batch["caption_options" + nameadd]))
                ]
            )
            + "\nLabel:"
            + str(labels)
        )
        # print(images_regular_and_flipped[ind][0])
        if torch.is_tensor(images_regular_and_flipped[0][ind]):
            plt.imshow(
                images_regular_and_flipped[0][ind].detach().cpu().permute(1, 2, 0)
            )
        else:
            plt.imshow(images_regular_and_flipped[0][ind])

        plt.subplot(oneside, oneside, i + 2)
        # show reversed image.

        if len(sample_batch["caption_options" + nameadd]) == 2:
            labels = [0, 1]
        else:
            labels = [0, 1, 1, 0]
        plt.title(
            "HARD NEG:\n"
            + "\n".join(
                [
                    sample_batch["caption_options" + nameadd][l][ind]
                    for l in range(len(sample_batch["caption_options" + nameadd]))
                ]
            )
            + "\nLabel:"
            + str(labels)
        )
        if torch.is_tensor(images_regular_and_flipped[1][ind]):
            plt.imshow(
                images_regular_and_flipped[1][ind].detach().cpu().permute(1, 2, 0)
            )
        else:
            plt.imshow(images_regular_and_flipped[1][ind])

    plt.tight_layout()
    plt.savefig(savename + str(iter) + pathadd + "_ver" + nameadd + ".png")
    plt.close()


def visualize_toy_data(iter, sample_batch, all_labels, savename=r"figures\cocobatch_"):
    """
    Create visualization for images in batch, phrase it is being compared with, and label.
    Args:
        iter (int): iteration being visualized
        sample_batch (dict): batch from dataloader
        savename (str, optional): save directory
    """
    batchsize = len(sample_batch["image_options"])
    oneside = int(np.ceil(batchsize**0.5))
    # figure arrangement dependent on batchsize
    plt.figure(figsize=(oneside * 5, oneside * 5))
    for i in range(batchsize):
        plt.subplot(oneside, oneside, i + 1)

        # find the correct sentences in caption_options.
        # correct_sentences = ""
        # for k in range(len(sample_batch["caption_options"])):
        #     if sample_batch["label"][k] == 1:
        #         correct_sentences += sample_batch["caption_options"] + "\n"

        plt.title("Label:\n" + str(sample_batch["label"][i]))
        if torch.is_tensor(sample_batch["image_options"][i]):
            plt.imshow(sample_batch["image_options"][i].detach().cpu().permute(1, 2, 0))
        else:
            plt.imshow(sample_batch["image_options"][i])

    plt.tight_layout()
    plt.savefig(savename + str(iter) + "_iter_cocobatch_ANDonly.png")
    plt.close()


def visualize_network_output(output, savename=r"figures\sample"):
    plt.figure()
    out = output.detach().cpu()
    if len(out.shape) > 1:
        plt.imshow(out, aspect="auto")
    else:
        plt.plot(out)
    plt.title("Network output")
    plt.savefig(savename + "_output.png")
    plt.close()


def visualize_gradients(
    grads,
    logic_token_labels=["AND", "OR", "NOT", "NOR", "XOR"],
    savename=r"figures\gradients.png",
):
    logic_len = len(logic_token_labels)
    plt.figure(figsize=(15, 8))

    # now grads is actually a dictionary. I would like to look at the names of each...

    logic_grads = grads["logic_token_embedding.weight"]
    other_grads = []
    for k in grads:
        if k != "logic_token_embedding.weight":
            other_grads.append(grads[k])

    gradlengths = [len(k) for k in other_grads]
    bignum = max(gradlengths)

    for k in range(len(other_grads)):
        # print(len(grads[k]))
        if len(other_grads[k]) != bignum:
            other_grads[k] = other_grads[k] + [0] * (bignum - len(other_grads[k]))

    grads_ = np.array(other_grads)

    plt.subplot(1, 2, 1)
    for i in range(grads_.shape[0]):
        plt.scatter(
            np.arange(grads_[i, :].shape[0]), grads_[i, :], label="Param" + str(i)
        )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Gradient")
    plt.title("Mean Absolute Gradient per Parameter over Epochs")
    logic_grads_ = np.array(logic_grads)

    plt.subplot(1, 2, 2)
    for i in range(logic_len):
        plt.scatter(
            np.arange(logic_grads_[:, i].shape[0]),
            logic_grads_[:, i],
            label="Param" + str(i),
        )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Gradient")
    plt.title("Mean Absolute Gradient per Logic Token over Epochs")
    plt.legend(logic_token_labels)
    # going from innermost (first logical label) to outermost.
    plt.savefig(savename)
    plt.close()


def plot_training_lossparts(
    train_loss_dict,
    valid_loss_dict,
    total_epochs=100,
    losstype="MSE",
    savename=r"figures/losscurve.png",
):
    k1 = len(list(train_loss_dict.values())[0])
    k2 = len(list(valid_loss_dict.values())[0])
    x1 = np.linspace(0, k1, k1)
    x2 = np.linspace(0, k2, k2)
    plt.figure(figsize=(15, 8))
    # plt.subplot(1, 2, 1)
    for subt in train_loss_dict:

        if len(train_loss_dict[subt]) < k1:
            train_loss_dict[subt] = train_loss_dict[subt] + [
                0 for ii in range(k1 - len(train_loss_dict[subt]))
            ]
        plt.scatter(x1, train_loss_dict[subt], label=subt)
        # Fit a line to the data
        # m, b = np.polyfit(x1, train_loss_dict[subt], 1)
        # plt.plot(x1, m * x1 + b, label="Bestfit_" + subt)
    # plt.legend()
    plt.title("training loss subcomponents")

    plt.subplot(1, 2, 2)
    for subt in valid_loss_dict:
        if len(valid_loss_dict[subt]) < k2:
            valid_loss_dict[subt] = valid_loss_dict[subt] + [
                0 for ii in range(k2 - len(valid_loss_dict[subt]))
            ]
        plt.scatter(x2, valid_loss_dict[subt], label=subt)
        # Fit a line to the data
        # m, b = np.polyfit(x1, valid_loss_dict[subt], 1)
        # plt.plot(x1, m * x1 + b, label="Bestfit_" + subt)
    plt.legend()
    plt.title("training acc subcomponents")

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


def plot_training_justlist(
    train_loss_list,
    valid_loss_list,
    train_acc_list=None,
    valid_acc_list=None,
    total_epochs=100,
    losstype="MSE",
    savename=r"figures/losscurve.png",
):
    x1 = np.linspace(0, len(train_loss_list), len(train_loss_list))
    x2 = np.linspace(0, len(valid_loss_list), len(valid_loss_list))
    plt.figure(figsize=(15, 8))
    plt.suptitle(
        "Training: "
        + str(total_epochs)
        + "total epochs, with "
        + str(losstype)
        + " loss"
    )

    plt.subplot(1, 2, 1)

    plt.plot(x1, train_loss_list)
    plt.plot(x2, valid_loss_list)
    plt.title("loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")

    # plt.xticks([0.2 * i for i in range(5)], [0.2 * i * totalepochs for i in range(5)])

    plt.legend(["train loss", "valid loss"])

    x1_ = np.linspace(0, len(train_acc_list), len(train_acc_list))
    x2_ = np.linspace(0, len(valid_acc_list), len(valid_acc_list))

    if train_acc_list:
        plt.subplot(1, 2, 2)
        plt.title("accuracy")
        plt.plot(x1_, train_acc_list)
        plt.plot(x2_, valid_acc_list)

        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        plt.legend(["train acc", "valid acc"])

    # plt.xticks([0.2 * i for i in range(5)], [0.2 * i * totalepochs for i in range(5)])

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


def plot_training(train_loss_list, valid_loss_dict, savename=r"figures/losscurve.png"):

    x1 = np.linspace(0, 1, len(train_loss_list))
    x2 = np.linspace(0, 1, len(valid_loss_dict["valid_loss_all"]))
    plt.figure(figsize=(15, 8))
    plt.suptitle("Training")

    plt.subplot(1, 2, 1)
    plt.scatter(x1, train_loss_list)
    plt.scatter(x2, valid_loss_dict["valid_loss_all"])

    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.legend(["train loss", "valid loss"])

    plt.subplot(1, 2, 2)
    for keyed in list(valid_loss_dict.keys())[1:]:

        plt.plot(valid_loss_dict[keyed])

    plt.xlabel("epoch")
    plt.ylabel("valid loss components")
    plt.legend(list(valid_loss_dict.keys())[1:])

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()
    # plt.show()
