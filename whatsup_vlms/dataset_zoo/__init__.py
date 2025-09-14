from .aro_datasets import (
    VG_Relation,
    VG_Attribution,
    COCO_Order,
    Flickr30k_Order,
    Controlled_Images,
    COCO_QA,
    VG_QA,
)
from .retrieval import COCO_Retrieval, Flickr30k_Retrieval


def get_dataset(
    dataset_name,
    image_preprocess=None,
    text_perturb_fn=None,
    image_perturb_fn=None,
    download=False,
    *args,
    **kwargs,
):
    """
    Factory function to get dataset instances.

    Args:
        dataset_name (str): Name of the dataset
        image_preprocess: Image preprocessing function
        text_perturb_fn: Text perturbation function
        image_perturb_fn: Image perturbation function
        download (bool): Whether to download the dataset
        *args, **kwargs: Additional arguments passed to dataset constructor

    Returns:
        Dataset instance
    """
    if dataset_name == "VG_Relation":
        from .aro_datasets import get_vg_relation

        return get_vg_relation(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "VG_Attribution":
        from .aro_datasets import get_vg_attribution

        return get_vg_attribution(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "COCO_Order":
        from .aro_datasets import get_coco_order

        return get_coco_order(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "Flickr30k_Order":
        from .aro_datasets import get_flickr30k_order

        return get_flickr30k_order(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "Controlled_Images_A":
        from .aro_datasets import get_controlled_images_a

        return get_controlled_images_a(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "Controlled_Images_B":
        from .aro_datasets import get_controlled_images_b

        return get_controlled_images_b(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "COCO_QA_one_obj":
        from .aro_datasets import get_coco_qa_one_obj

        return get_coco_qa_one_obj(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "COCO_QA_two_obj":
        from .aro_datasets import get_coco_qa_two_obj

        return get_coco_qa_two_obj(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "VG_QA_one_obj":
        from .aro_datasets import get_vg_qa_one_obj

        return get_vg_qa_one_obj(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "VG_QA_two_obj":
        from .aro_datasets import get_vg_qa_two_obj

        return get_vg_qa_two_obj(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "COCO_Retrieval":
        from .retrieval import get_coco_retrieval

        return get_coco_retrieval(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    elif dataset_name == "Flickr30k_Retrieval":
        from .retrieval import get_flickr30k_retrieval

        return get_flickr30k_retrieval(
            image_preprocess=image_preprocess,
            text_perturb_fn=text_perturb_fn,
            image_perturb_fn=image_perturb_fn,
            download=download,
            *args,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = [
    "VG_Relation",
    "VG_Attribution",
    "COCO_Order",
    "Flickr30k_Order",
    "Controlled_Images",
    "COCO_QA",
    "VG_QA",
    "COCO_Retrieval",
    "Flickr30k_Retrieval",
    "get_dataset",
]
