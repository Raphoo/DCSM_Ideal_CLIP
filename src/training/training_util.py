from torch.utils.data import DataLoader, Dataset, Subset
from src.training.toydataset_dense import (
    ObjaverseDataset_color,
    ObjaverseDataset_spatial,
    Negation_Dataset,
)
import torch
import os


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


class MultipleDatasets(Dataset):
    def __init__(self, datasets, truncate_to_min=True):
        """
        Initialize the MultipleDatasets instance with a list of datasets.
        If truncate_to_min is True, all datasets are truncated to the size of the smallest dataset.
        """
        self.datasets = datasets
        if truncate_to_min:
            min_length = min(len(dataset) for dataset in datasets)
            self.datasets = [
                self._shuffle_and_truncate(dataset, min_length) for dataset in datasets
            ]
        self.length = len(self.datasets[0])  # All datasets are now the same length

    def __len__(self):
        """
        Length of the combined dataset corresponds to the (possibly truncated) length of individual datasets.
        """
        return self.length

    def __getitem__(self, index):
        """
        Returns a list containing one sample from each dataset.
        """
        return [dataset[index] for dataset in self.datasets]

    @staticmethod
    def _shuffle_and_truncate(dataset, length):
        """
        Shuffles and truncates the dataset to the specified length.
        """
        indices = torch.randperm(len(dataset))[
            :length
        ]  # Generate random shuffled indices
        return Subset(dataset, indices)


def custom_collate_multiple_datasets(batch):
    """
    Collate function for batches from MultipleDatasets.
    Batch is a list of samples from each dataset, so it needs to be split by dataset.
    """
    collated_batch = {f"dataset_{i}": [] for i in range(len(batch[0]))}

    for sample in batch:
        for i, dataset_sample in enumerate(sample):
            collated_batch[f"dataset_{i}"].append(dataset_sample)

    return collated_batch


def make_and_return_train_data(general_data_path, halfneg_bool=False, batch_size=32):
    img_dir_color = os.path.join(
        general_data_path, "objaverse_attributes_overlay_withbg"
    )
    img_dir_relations = os.path.join(
        general_data_path, "objaverse_train_composite_newrelations_withbg"
    )
    img_dir_neg = os.path.join(general_data_path, "negated_pairs_cocotrain_v1.json")
    # Create instances of each dataset
    spatial_dataset_A = ObjaverseDataset_spatial(
        image_dir=img_dir_relations,
        no_special_words=True,
        settype=0,
        add_spw=True,
        cutmix=True,
        notchance=0,
    )
    spatial_dataset_B = ObjaverseDataset_spatial(
        image_dir=img_dir_relations,
        no_special_words=True,
        settype=1,
        add_spw=True,
        cutmix=True,
        notchance=0,
    )

    color_dataset_A = ObjaverseDataset_color(
        image_dir=img_dir_color,
        no_special_words=True,
        settype=0,
        add_spw=True,
        cutmix=True,
        notchance=0,
    )
    color_dataset_B = ObjaverseDataset_color(
        image_dir=img_dir_color,
        no_special_words=False,
        settype=1,
        add_spw=True,
        cutmix=True,
        notchance=0,
    )

    neg_dataset_A = Negation_Dataset(
        json_file=img_dir_neg,
        data_dir=general_data_path,
        settype=0,
    )
    neg_dataset_B = Negation_Dataset(
        json_file=img_dir_neg,
        data_dir=general_data_path,
        settype=1,
    )

    if halfneg_bool:
        neg_dataset_A = Subset(
            neg_dataset_A, [k for k in range(len(neg_dataset_A) // 2)]
        )
        neg_dataset_B = Subset(
            neg_dataset_B, [k for k in range(len(neg_dataset_B) // 2)]
        )

    paired_dataset_train_spatial = PairedDataset(spatial_dataset_A, spatial_dataset_B)
    paired_dataset_train_color = PairedDataset(color_dataset_A, color_dataset_B)
    paired_dataset_train_neg = PairedDataset(neg_dataset_A, neg_dataset_B)

    # Combine paired datasets into MultipleDatasets
    combined_datasets = MultipleDatasets(
        [
            paired_dataset_train_spatial,
            paired_dataset_train_color,
            paired_dataset_train_neg,
        ]
    )

    # Create a dataloader for the combined datasets
    combined_dataloader_train = DataLoader(
        combined_datasets,
        batch_size=batch_size,  # This specifies the number of samples per dataset in each batch
        shuffle=True,
        collate_fn=custom_collate_multiple_datasets,
    )

    return combined_dataloader_train


def standardize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std
