import torch
import os
import argparse

from train import run_training


if __name__ == "__main__":

    torch.manual_seed(15)

    root_dir = os.path.join("whatsup_vlms", "data")
    losstype = "CE"
    general_path = os.path.join("figures", "vis_patchnet")
    parser = argparse.ArgumentParser(
        description="Train patchnet - without learning the special tokens."
    )
    parser.add_argument(
        "-s", "--model_save_path", help="General path for saving models", required=True
    )
    parser.add_argument(
        "-f",
        "--figure_save_path",
        help="General path for saving figures",
        required=False,
        default=os.path.join("figures", "vis_patchnet2"),
    )

    parser.add_argument(
        "-d",
        "--data_path",
        help="General path for saving figures",
        required=False,
        default=os.path.join("whatsup_vlms", "data"),
    )

    parser.add_argument(
        "-do",
        "--dropout",
        help="dropout rate",
        required=False,
        default=0,
    )

    parser.add_argument(
        "-ep",
        "--epochs",
        help="total epochs for training",
        required=False,
        type=int,
        default=99,
    )

    parser.add_argument(
        "-lr",
        "--learn_rate",
        help="learning rate for training",
        required=False,
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        help="batch size for training",
        required=False,
        type=int,
        default=8,
    )

    args = parser.parse_args()  # Use args directly instead of converting to dict

    general_model_path = args.model_save_path
    general_path = args.figure_save_path  # Use the parsed argument directly
    general_data_path = args.data_path
    dro = args.dropout
    lr = args.learn_rate
    epochs = args.epochs

    # root_dir = general_data_path = (
    #     r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\whatsup_vlms\data"
    # )
    losstype = "CE"
    # general_path = os.path.join("figures", "vis_patchnet")
    # general_data_path = (
    #     r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\whatsup_vlms\data"
    # )
    # general_model_path = r"models"
    halfneg_bool = False

    run_training(
        general_path=general_path,
        general_model_path=general_model_path,
        general_data_path=general_data_path,
        epochs=epochs,
        lr=lr,
        batch_size=8,
        l2_lambda=0,
        dropout_prob=dro,
        single_words=False,
        halfneg_bool=halfneg_bool,
        no_replacements=False,
    )
