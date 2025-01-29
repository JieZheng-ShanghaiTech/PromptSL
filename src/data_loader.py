import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
import pickle
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import random

class GenePairDataset(Dataset):
    def __init__(self, embeddings, labels):
        # Convert all numpy arrays in embeddings to tensors
        self.embeddings = [
            tuple(torch.from_numpy(e).float() if isinstance(e, np.ndarray) else e.clone().detach().float() for e in emb)
            for emb in embeddings
        ]
        self.labels = [label.clone().detach().long() for label in labels]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class GenePairDataset0(Dataset): ## for check
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        cell_line_emb, token_emb_1, attention_mask_1, genept_emb_1, kge_emb_1, token_emb_2, attention_mask_2, genept_emb_2, kge_emb_2 = self.embeddings[idx]
        label = self.labels[idx]

        # Print types for debugging
        print(f"cell_line_emb type: {type(cell_line_emb)}")
        print(f"token_emb_1 type: {type(token_emb_1)}")
        print(f"attention_mask_1 type: {type(attention_mask_1)}")
        print(f"genept_emb_1 type: {type(genept_emb_1)}")
        print(f"kge_emb_1 type: {type(kge_emb_1)}")
        print(f"token_emb_2 type: {type(token_emb_2)}")
        print(f"genept_emb_2 type: {type(genept_emb_2)}")
        print(f"kge_emb_2 type: {type(kge_emb_2)}")
        print(f"label type: {type(label)}")

        return (cell_line_emb, token_emb_1, genept_emb_1, kge_emb_1, token_emb_2, genept_emb_2, kge_emb_2), label


torch.serialization.add_safe_globals([GenePairDataset])

def load_datasets(cell_line, logger=None):
    folder = "/home/tinglu/MLLM4SL/data/gene_pair_datasets3_ori_omics"
    dataset_paths = {
        "A549": f"{folder}/A549_dataset.pt",
        "PK1": f"{folder}/PK1_dataset.pt",
        "IPC298": f"{folder}/IPC298_dataset.pt",
        "MEWO": f"{folder}/MEWO_dataset.pt",
        "all": [
            f"{folder}/A549_dataset.pt",
            f"{folder}/PK1_dataset.pt",
            f"{folder}/IPC298_dataset.pt",
            f"{folder}/MEWO_dataset.pt"
        ]
    }
    if cell_line == "all":
        datasets = []
        for path in dataset_paths["all"]:
            datasets.append(torch.load(path))
            logger.info(path)
        return datasets
    else:
        if logger:
            logger.info(dataset_paths[cell_line])
        return [torch.load(dataset_paths[cell_line])]

# for overlap data in SLBench
def load_train_test_datasets(cell_line, folder=None, shuffle=True, logger=None):
    dataset_paths = {
        "A549": {
            # "train": f"{folder}/A549_train.pt",
            # "train": f"{folder}/A549_train_overlap.pt",
            "train": f"{folder}/A549_train_subset.pt",
            # "test": f"{folder}/A549_test.pt"
            "test": f"{folder}/A549_conflict_test.pt"
            # "test": f"{folder}/A549_consist_test.pt"
        },
        "PK1": {
            # "train": f"{folder}/PK1_train.pt",
            # "train": f"{folder}/PK1_train_overlap.pt",
            "train": f"{folder}/PK1_train_subset.pt",
            # "test": f"{folder}/PK1_test.pt"
            "test": f"{folder}/PK1_conflict_test.pt"
            # "test": f"{folder}/PK1_consist_test.pt"
        },
        "IPC298": {
            # "train": f"{folder}/IPC298_train.pt",
            # "train": f"{folder}/IPC298_train_overlap.pt",
            "train": f"{folder}/IPC298_train_subset.pt",
            # "test": f"{folder}/IPC298_test.pt"
            "test": f"{folder}/IPC298_conflict_test.pt"
            # "test": f"{folder}/IPC298_consist_test.pt"
        },
        "MEWO": {
            # "train": f"{folder}/MEWO_train.pt",
            # "train": f"{folder}/MEWO_train_overlap.pt",
            "train": f"{folder}/MEWO_train_subset.pt",
            # "test": f"{folder}/MEWO_test.pt"
            # "test": f"{folder}/MEWO_conflict_test.pt"
            "test": f"{folder}/MEWO_consist_test.pt"
        },
        "22RV1": {
            "test": f"{folder}/22RV1_test.pt"
        },
        "GI1": {
            "test": f"{folder}/GI1_test.pt"
        },
        "HS936T": {
            "test": f"{folder}/HS936T_test.pt"
        },
    }
    if cell_line == "all":
        # Load train and test sets for the main 4 cell lines
        train_emb, train_labels, test_emb, test_labels = [], [], [], []
        selected_lines = ["A549", "PK1", "IPC298", "MEWO"]
        for line in selected_lines:
            paths = dataset_paths[line]
            if "train" in paths:
                train_dataset = torch.load(paths["train"])
                train_e, train_l = zip(*[(sample[0], sample[1]) for sample in train_dataset])
                train_emb.extend(train_e)
                train_labels.extend(train_l)
            if "test" in paths:
                test_dataset = torch.load(paths["test"])
                test_e, test_l = zip(*[(sample[0], sample[1]) for sample in test_dataset])
                test_emb.extend(test_e)
                test_labels.extend(test_l)
            if logger:
                logger.info(f"Loaded {paths['train']} and {paths['test']}")

        random.seed(42)  # Ensure reproducibility
        # Shuffle the combined training dataset
        combined_train = list(zip(train_emb, train_labels))
        random.shuffle(combined_train)
        train_emb, train_labels = zip(*combined_train)

        # Shuffle the combined testing dataset
        combined_test = list(zip(test_emb, test_labels))
        random.shuffle(combined_test)
        test_emb, test_labels = zip(*combined_test)

        return train_emb, train_labels, test_emb, test_labels
    else:
        try:
            train_dataset = torch.load(dataset_paths[cell_line]["train"], weights_only=True)
            train_path = dataset_paths[cell_line]["train"]
            train_emb, train_labels = zip(*[(sample[0], sample[1]) for sample in train_dataset])

            # Shuffle the train dataset
            train_data = list(zip(train_emb, train_labels))
            random.seed(42)  # Set seed for reproducibility
            random.shuffle(train_data)
            train_emb, train_labels = zip(*train_data)
        except:
            train_path, train_emb, train_labels = None, None, None

        test_path = dataset_paths[cell_line]["test"]
        test_dataset = torch.load(test_path, weights_only=True)
        test_emb, test_labels = zip(*[(sample[0], sample[1]) for sample in test_dataset])

        # Shuffle the test dataset
        test_data = list(zip(test_emb, test_labels))
        print("len of test data", len(test_data)) 
        if shuffle:
            random.seed(42)  # Set seed for reproducibility
            random.shuffle(test_data)
        test_emb, test_labels = zip(*test_data)

        if logger:
            logger.info(f"Loaded {train_path} and {test_path}")

        return train_emb, train_labels, test_emb, test_labels

# for case study
def load_train_test_datasets_infer(cell_line_condition, folder=None, shuffle=True, logger=None):
    train_path, train_emb, train_labels = None, None, None
    test_path = f"{folder}/{cell_line_condition}_test.pt"
    # Load test dataset
    test_dataset = torch.load(test_path, weights_only=True)
    test_emb, test_labels = zip(*[(sample[0], sample[1]) for sample in test_dataset])

    # Shuffle the test dataset
    test_data = list(zip(test_emb, test_labels))
    print("len of test data", len(test_data)) 
    if shuffle:
        random.seed(42)  # Set seed for reproducibility
        random.shuffle(test_data)
    test_emb, test_labels = zip(*test_data)

    if logger:
        logger.info(f"Loaded {train_path} and {test_path}")

    return train_emb, train_labels, test_emb, test_labels


def kmeans_undersample(dataset, n_clusters=10, neg_pos_ratio=1.0, positive_class=1, random_seed=42):
    """
    Perform K-means undersampling to balance the positive and negative samples.

    Args:
        dataset (GenePairDataset): Original dataset with imbalanced classes.
        n_clusters (int): Number of clusters for K-means.
        positive_class (int): Label for the positive class (default is 1).
        neg_pos_ratio (float): Desired ratio of negative samples to positive samples (default is 1.0).
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        balanced_embeddings (list): Balanced list of embeddings.
        balanced_labels (list): Balanced list of labels.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Separate positive and negative samples
    embeddings, labels = zip(*[(sample[0], sample[1]) for sample in dataset])
    positive_samples = [(emb, label) for emb, label in zip(embeddings, labels) if label == positive_class]
    negative_samples = [(emb, label) for emb, label in zip(embeddings, labels) if label != positive_class]

    # Extract negative sample embeddings for clustering
    negative_embeddings = [emb for emb, _ in negative_samples]
    
    # ## check size
    # for idx, emb in enumerate(negative_embeddings[:3]):  # Check the first 3 samples
    #     print(f"Sample {idx}:")
    #     # Iterate over the elements in emb and print their shapes
    #     for idx, item in enumerate(emb):
    #         print(f"emb[{idx}] shape: {item.shape if hasattr(item, 'shape') else 'No shape attribute'}")
        
    negative_embeddings_concat = [
        # np.concatenate([emb[3], emb[4], emb[7], emb[8]])  # Concatenate GenePT and KG embeddings for clustering
        np.concatenate([emb[9], emb[10], emb[13], emb[14]])
        for emb in negative_embeddings
    ]
    negative_embeddings_concat = np.array(negative_embeddings_concat)

    print("Kmeans undersampling..")
    
    if neg_pos_ratio == -1: # If neg_pos_ratio is -1, take all negative samples without undersampling
        balanced_samples = positive_samples + negative_samples
    else:
        positive_count = len(positive_samples)
        # Perform K-means clustering on negative samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(negative_embeddings_concat)
        target_negative_count = int(positive_count * neg_pos_ratio)
        cluster_sizes = np.bincount(cluster_labels)
        cluster_proportions = cluster_sizes / cluster_sizes.sum()
        samples_per_cluster = (cluster_proportions * target_negative_count).astype(int)

        selected_negatives = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            n_samples_to_draw = min(len(cluster_indices), samples_per_cluster[cluster_id])
            cluster_sample_indices = np.random.choice(cluster_indices, size=n_samples_to_draw, replace=False)
            selected_negatives.extend([negative_samples[idx] for idx in cluster_sample_indices])

        # Combine positive and selected negative samples
        balanced_samples = positive_samples + selected_negatives

    np.random.shuffle(balanced_samples)

    # Unpack embeddings and labels
    balanced_embeddings, balanced_labels = zip(*balanced_samples)

    return list(balanced_embeddings), list(balanced_labels)


def random_undersample(dataset, neg_pos_ratio=1.0, positive_class=1, random_seed=42):
    """
    Perform random undersampling to balance the positive and negative samples.

    Args:
        dataset (GenePairDataset): Original dataset with imbalanced classes.
        neg_pos_ratio (float): Desired ratio of negative samples to positive samples (default is 1.0).
                               If set to -1, all negative samples will be kept, and no undersampling will be done.
        positive_class (int): Label for the positive class (default is 1).
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        balanced_embeddings (list): Balanced list of embeddings.
        balanced_labels (list): Balanced list of labels.
    """
    np.random.seed(random_seed)

    # Separate positive and negative samples
    embeddings, labels = zip(*[(sample[0], sample[1]) for sample in dataset])
    positive_samples = [(emb, label) for emb, label in zip(embeddings, labels) if label == positive_class]
    negative_samples = [(emb, label) for emb, label in zip(embeddings, labels) if label != positive_class]

    positive_count = len(positive_samples)
    print("Random undersampling..")
    
    if neg_pos_ratio == -1: # If neg_pos_ratio is -1, take all negative samples without undersampling
        balanced_samples = positive_samples + negative_samples
    else:
        # Calculate target number of negative samples based on neg_pos_ratio
        target_negative_count = int(positive_count * neg_pos_ratio)

        # Randomly select negative samples to match the ratio
        negative_embeddings, negative_labels = zip(*negative_samples)
        selected_neg_indices = np.random.choice(len(negative_embeddings), size=target_negative_count, replace=False)
        selected_negatives = [(negative_embeddings[idx], negative_labels[idx]) for idx in selected_neg_indices]

        # Combine the positive and selected negative samples
        balanced_samples = positive_samples + selected_negatives

    # Shuffle the combined balanced samples
    np.random.shuffle(balanced_samples)

    # Unzip balanced samples into embeddings and labels
    balanced_embeddings, balanced_labels = zip(*balanced_samples)

    return list(balanced_embeddings), list(balanced_labels)



def prepare_dataloader(embeddings, labels, batch_size=64, train_split=0.8, scenario="C1", shuffle=True, logger=None):
    """
    Prepare DataLoader for training and validation, or for evaluation using the entire dataset.
    This includes different data splitting strategies for training and testing.

    Args:
        embeddings (list): List of embeddings.
        labels (list): List of labels corresponding to the embeddings.
        batch_size (int): Batch size for DataLoader.
        train_split (float): Proportion of training data (default is 0.8).
                            Set to 0.0 for using the entire dataset for evaluation.
        scenario (str): Scenario for splitting the data. Options are "C1", "C2", "C3".

    Returns:
        train_loader (DataLoader): DataLoader for training, or the full dataset for evaluation.
        val_loader (DataLoader): DataLoader for validation, or None if using the full dataset.
    """
    # Create the dataset
    dataset = GenePairDataset(embeddings, labels)
    print("Data loader...")

    # If train_split is 0.0, use the entire dataset for evaluation (no train/val split)
    if train_split == 0.0:
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        print(f"Whole size: {len(dataset)}")
        return None, val_loader

    # Convert gene tensors (genept embedding) into hashable objects
    unique_genes = set()
    for emb in embeddings:
        gene_1 = tuple(emb[9].cpu().numpy())  # Convert tensor to tuple
        gene_2 = tuple(emb[13].cpu().numpy())  # Convert tensor to tuple
        unique_genes.update([gene_1, gene_2])

    # Split based on the scenario
    if scenario == "C1":  # Scenario C1: Completely disjoint gene pairs
        train_dataset, val_dataset = train_test_split(dataset, train_size=train_split, random_state=42)
        print(f"In C1, train size (gene pairs): {len(train_dataset)}, test size (gene pairs): {len(val_dataset)} ({(len(val_dataset) / len(dataset)) * 100:.2f}%).")
        if logger:
            logger.info(f"In C1, train size (gene pairs): {len(train_dataset)}, test size (gene pairs): {len(val_dataset)} ({(len(val_dataset) / len(dataset)) * 100:.2f}%).")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    elif scenario == "C2":  # Scenario C2: One gene in pair is seen, one gene is not seen
        train_genes, test_genes = train_test_split(list(unique_genes), train_size=train_split, random_state=42)

        train_embeddings = []
        train_labels = []
        test_embeddings = []
        test_labels = []
        for emb, label in zip(embeddings, labels):
            gene_1 = tuple(emb[9].cpu().numpy()) # 3
            gene_2 = tuple(emb[13].cpu().numpy()) # 7
            if (gene_1 in train_genes and gene_2 in test_genes) or (gene_1 in test_genes and gene_2 in train_genes):
                test_embeddings.append(emb)
                test_labels.append(label)
            else:
                train_embeddings.append(emb)
                train_labels.append(label)
        
        print(f"In C2, train size (gene pairs): {len(train_embeddings)}, test size (gene pairs): {len(test_embeddings)} ({(len(test_embeddings) / len(dataset)) * 100:.2f}%).")
        logger.info(f"In C2, train size (gene pairs): {len(train_embeddings)}, test size (gene pairs): {len(test_embeddings)} ({(len(test_embeddings) / len(dataset)) * 100:.2f}%).")

        train_loader = DataLoader(GenePairDataset(train_embeddings, train_labels), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(GenePairDataset(test_embeddings, test_labels), batch_size=batch_size, shuffle=False)

    elif scenario == "C3":  # Scenario C3: Completely disjoint genes (no overlap between training and test)
        train_genes, test_genes = train_test_split(list(unique_genes), train_size=train_split, random_state=42)

        train_embeddings = []
        train_labels = []
        test_embeddings = []
        test_labels = []
        for emb, label in zip(embeddings, labels):
            gene_1 = tuple(emb[9].cpu().numpy())
            gene_2 = tuple(emb[13].cpu().numpy())
            if gene_1 in train_genes and gene_2 in train_genes:
                train_embeddings.append(emb)
                train_labels.append(label)
            elif gene_1 in test_genes and gene_2 in test_genes:
                test_embeddings.append(emb)
                test_labels.append(label)
        print(f"In C3, train size (gene pairs): {len(train_embeddings)}, test size (gene pairs): {len(test_embeddings)} ({(len(test_embeddings) / len(dataset)) * 100:.2f}%).")
        logger.info(f"In C3, train size (gene pairs): {len(train_embeddings)}, test size (gene pairs): {len(test_embeddings)} ({(len(test_embeddings) / len(dataset)) * 100:.2f}%).")
        
        train_loader = DataLoader(GenePairDataset(train_embeddings, train_labels), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(GenePairDataset(test_embeddings, test_labels), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

