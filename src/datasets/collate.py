import torch

# Should allign all audio samples to make proper batch
def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["mix"] = torch.vstack(
        [elem["mix"] for elem in dataset_items]
    )
    if "s1" in dataset_items[0]:
        result_batch["targets"] = torch.stack([torch.stack([elem["s1"], elem["s2"]]) for elem in dataset_items])
    result_batch["lengths"] = torch.tensor([len(elem["mix"]) for elem in dataset_items])
    result_batch["s1_mouth"] = [elem["s1_mouth"] for elem in dataset_items]
    result_batch["s2_mouth"] = [elem["s2_mouth"] for elem in dataset_items]
    result_batch["sample_rate"] = dataset_items[0]["sample_rate"]
    result_batch["video"] = torch.stack([torch.stack([elem["s1_video"], elem["s2_video"]]) for elem in dataset_items])
    return result_batch
