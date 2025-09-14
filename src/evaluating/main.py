import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import torch
import argparse
from run_evals import evaluate_all_models

# from run_evals import evaluate_model, update_results_csv
from src.util.models_and_dataloaders import get_test_dataloaders
from open_clip import create_model_and_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train patchnet - without learning the special tokens."
    )
    parser.add_argument(
        "-r", "--root_dir", help="General path for data", required=False, default="data"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="General path for saving output results",
        required=False,
        default="vitL_outputs.csv",
    )

    parser.add_argument(
        "-m",
        "--model_path",
        help="General path for model checkpoint",
        required=False,
        default="Naive CLIP",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        help="Name of model",
        required=False,
        default="Naive CLIP",
    )

    parser.add_argument(
        "-tl", "--text_seq", help="Text sequence for model", required=False, default=30
    )

    model_path = parser.parse_args().model_path
    model_name = parser.parse_args().model_name
    text_seq = parser.parse_args().text_seq

    root_dir = parser.parse_args().root_dir
    csv_path = parser.parse_args().output_path
    # root_dir = r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\whatsup_vlms\data"
    # csv_path = "vitL_outputs.csv"

    if model_name is not "all":
        model_dict = {model_name: (model_path, text_seq)}
    else:
        model_dict = {
            # "ViTL objaverse - with standardize and negation - 26 ep -- smaller -- halfneg false -- dropout 0.01": (
            #     r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\models\patchnet_MANY26_spw_True_batchsize_8_lr_0.001_drop_0.01_scaledown_1v1.pt",
            #     30,
            # ),
            # "ViTL coco - with standardize and negation - 26 ep -- smaller -- halfneg false -- dropout 0.01": (
            #     r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\models\patchnet_MANY5_spw_True_batchsize_8_lr_0.001_drop_0.01_lambda_0_scale_1_REBUTTAL_v2.pt",  # patchnet_MANY26_spw_True_batchsize_8_lr_0.001_drop_0.1_lambda_0_singlew_Falsetextseq_30halfneg_False_coco_finetune_std_wneg_v5.pt",
            #     30,
            # ),
            # "NAIVE scoring MLP": (
            #     r"patchnet_MANY20_batchsize_8_lr_0.001_modelkind_MLPclip_transformer_scorer_v7.pt",
            #     20,
            # ),
            # "objaverse - with standardize and negation - 26 ep -- smaller -- halfneg false -- dropout 0.1": (
            #     # r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\models\patchnet_MANY2_spw_True_batchsize_8_lr_0.001_drop_0.01_scaledown_1_REBUTTAL_v3.pt",
            #     r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\models\patchnet_MANY26_spw_True_batchsize_8_lr_0.001_drop_0.1_lambda_0_singlew_Falsehalfneg_False_fix_objaverse_std_wneg_v5.pt",
            #     30,
            # ),
            # "coco - with standardize and negation - 26 ep -- smaller -- halfneg false -- dropout 0.1": (
            #     # r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\models\patchnet_MANY11_spw_True_batchsize_8_lr_0.001_drop_0.01_lambda_0_scale_1_REBUTTAL_v2.pt"
            #     r"D:\temporary_data\art_with_shapes\Logical-CLIP\src\models\patchnet_MANY26_spw_True_batchsize_8_lr_0.001_drop_0.1_lambda_0_singlew_Falsetextseq_30halfneg_False_coco_finetune_std_wneg_v5.pt",
            #     30,
            # ),
            "NAIVE CoCa": ("NAIVE CoCa", 30),
            "NAIVE NegCLIP": ("NAIVE NegCLIP", 30),
            "NAIVE BLIP": ("NAIVE BLIP", 30),
            "NAIVE CLIP": ("NAIVE CLIP-ViT-L14", 30),
            "NAIVE SigLIP": ("NAIVE SigLIP", 30),
        }
    evaluate_all_models(model_dict, root_dir, csv_path)
