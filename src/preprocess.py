import pandas as pd
import pickle
import json
from pykeen.datasets import PrimeKG
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer
import os
from src.data_loader import GenePairDataset

"""
    preprocess the data of different modalities,
    perform negative sampling and save data loaders.
    
"""

device = torch.device("cuda:1")
print(device)

cell_line_df_folder = "/home/tinglu/MLLM4SL/data/case_study_compare/KG_test"
# cell_line_df_folder = "/home/tinglu/MLLM4SL/data/SLBench_overlap"
cell_line_dfs = {
    "A549_train_subset": pd.read_csv(f'{cell_line_df_folder}/A549_train_subset.csv').copy(),
    "PK1_train_subset": pd.read_csv(f'{cell_line_df_folder}/PK1_train_subset.csv').copy(),
    "IPC298_train_subset": pd.read_csv(f'{cell_line_df_folder}/IPC298_train_subset.csv').copy(),
    "MEWO_train_subset": pd.read_csv(f'{cell_line_df_folder}/MEWO_train_subset.csv').copy(),
}

cell_lines_info = {
    "A549": {
        "cell_line_id": "ACH-000681",
        "description": "A549 cell line is derived from human lung carcinoma, used in research of lung cancer and chemotherapeutic responses. It is responsible for the study of cancerous transformations and treatment efficacy in pulmonary cells. These cells are characterized by an epithelial-like structure, making them useful for studying ion transport across alveolar surfaces and drug metabolism."
    },
    "PK1": {
        "cell_line_id": "ACH-000307",
        "description": "PK1 cell line is derived from porcine kidney tissue, used in research of viral infections and nephrotoxicity. It is responsible for the diffusion of substances across cellular membranes, vital for understanding pathogenic interactions. These cells exhibit high levels of enzymatic activity, important for studying kidney function and drug-induced renal damage."
    },
    "IPC298": {
        "cell_line_id": "ACH-000915",
        "description": "IPC-298 cell line is derived from human malignant melanoma, used in research of skin cancer mechanisms and drug efficacy. It is responsible for exploring cellular responses to oncogenic stress. These cells provide a model for understanding melanoma biology and testing novel therapeutic strategies."
    },
    "MEWO": {
        "cell_line_id": "ACH-000987",
        "description": "MEWO cell line is derived from human skin (melanoma), used in research of melanoma progression and treatment. It is responsible for modeling melanoma spread and response to therapies. These cells are characterized by their ability to form tumors and metastasize, which are key features for oncological studies."
    },
    "22RV1": {
        "cell_line_id": "ACH-000956",
        "description": "22RV1 cell line is derived from a human prostate carcinoma, used in research of prostate cancer and hormone-responsive cancer therapies. It is responsible for studying tumor growth, androgen receptor signaling, and metastasis in prostate cells. These cells are characterized by their ability to proliferate in the presence of androgens, making them a valuable model for investigating prostate cancer progression and the efficacy of androgen-deprivation therapies."
    },
    "GI1": {
        "cell_line_id": "ACH-000756",
        "description": "GI1 cell line is derived from human glioblastoma multiforme (GBM), a highly malignant tumor of the central nervous system. It is used in research on brain cancer, tumor biology, and therapeutic resistance. These cells exhibit high invasiveness and resistance to radiation, making them an important model for studying the mechanisms of glioma progression and developing targeted therapies for GBM."
    },
    "HS936T": {
        "cell_line_id": "ACH-000801",
        "description": "HS936T cell line is derived from human melanoma, used in research of skin cancer and metastatic mechanisms. It is employed to study the transformation of melanocytes, tumor invasion, and resistance to treatments. These cells exhibit aggressive proliferation and metastatic potential, making them an essential model for understanding melanoma progression and testing targeted therapies."
    },
}

### load GenePT, cell line emb and PrimeKG Emb
with open("/home/tinglu/LLM4SL/GenePT/data_v2/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle.", 'rb') as file:
    genept_gene_embeddings = pickle.load(file) # 3072, text-embedding-3-large embeddings
for gene, emb in genept_gene_embeddings.items():
    genept_gene_embeddings[gene] = np.array(emb, dtype=np.float32)
    
## load cell line embedding 768
with open("/home/tinglu/MLLM4SL/data/cell_line_emb2.pkl", 'rb') as file:
    cell_line_embeddings = pickle.load(file) # 768

### BioBERT
model_name = "/home/tinglu/LLM4SL/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
biobert = AutoModel.from_pretrained(model_name)
biobert.to(device)


#### PrimeKG Emb
kg_dataset = PrimeKG(create_inverse_triples=True)
# Load the embeddings
kg_entity_embeddings = np.load('/home/tinglu/LLM4SL/KG/ConvE/conv_primekg_entity_embeddings.npy')

kg_entity_to_id = kg_dataset.training.entity_to_id



def find_word_indices(tokens, context_before, context_after):
    """
    Locate the indices of a target word based on surrounding context.

    Args:
        tokens (list): List of tokens from tokenized text.
        context_before (list): List of tokens before the target word.
        context_after (list): List of tokens after the target word.

    Returns:
        tuple: (start_index, end_index) of the target word tokens.
    """
    for i in range(len(tokens) - len(context_before) - len(context_after)):
        if tokens[i:i + len(context_before)] == context_before:
            j = i + len(context_before)
            while j < len(tokens) and tokens[j:j + len(context_after)] != context_after:
                j += 1
            if tokens[j:j + len(context_after)] == context_after:
                return i + len(context_before), j
    raise ValueError("Target word not found with the given context.")

def get_token_embeddings(biobert, tokenizer, text, device):
    """
    Generate token embeddings for a given text using BioBERT.

    Args:
        text (str): Input text for embedding.

    Returns:
        token_embeddings (torch.Tensor): Token embeddings of shape [seq_len, hidden_dim].   
        tokens (list): List of tokens corresponding to the embeddings.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    cell_line_startidx, cell_line_endidx = find_word_indices(tokens, context_before=["in", "the"], context_after=["cell", "line", ","])
    gene_startidx, gene_endidx = find_word_indices(tokens, context_before=["cell", "line", ","], context_after=["has", "a", "gene"])
    cell_line_remove_tokens = list(range(cell_line_startidx + 1, cell_line_endidx))
    gene_remove_tokens = list(range(gene_startidx + 1, gene_endidx))
    
    try:
        
        input_ids[0] = torch.cat([
            input_ids[0][:gene_startidx + 1],  
            input_ids[0][gene_endidx:],  
            torch.tensor([tokenizer.pad_token_id] * len(gene_remove_tokens), device=device)  
        ])
    except:
        print(cell_line_startidx, cell_line_endidx, gene_startidx, gene_endidx)
        print(text)
        print(tokens)
    attention_mask[0] = torch.cat([
        attention_mask[0][:gene_startidx + 1],  
        attention_mask[0][gene_endidx:], 
        torch.tensor([0] * len(gene_remove_tokens), device=device)  
    ])
    

    input_ids[0] = torch.cat([
        input_ids[0][:cell_line_startidx + 1],  # Keep tokens up to start of the range
        input_ids[0][cell_line_endidx:],  # Skip tokens in the range
        torch.tensor([tokenizer.pad_token_id] * len(cell_line_remove_tokens), device=device)  # Add PAD tokens
    ])
    attention_mask[0] = torch.cat([
        attention_mask[0][:cell_line_startidx + 1],  # Keep attention mask up to start of the range
        attention_mask[0][cell_line_endidx:],  # Skip mask values for removed tokens
        torch.tensor([0] * len(cell_line_remove_tokens), device=device)  # Add mask for PAD tokens
    ])
    # print(input_ids[0][2:7])

    
    assert len(input_ids[0]) == 512, "Modified sequence length must match the original max length."
    
    with torch.no_grad():
        token_embeddings = biobert.embeddings(input_ids=input_ids).squeeze(0)  # Shape: [seq_len, hidden_dim]

    
    return input_ids, attention_mask.squeeze(0), token_embeddings, cell_line_startidx, gene_startidx

def replace_token_embeddings(token_embeddings, cell_line_startidx, gene_startidx, cell_line_embedding=None, gene_embedding=None):
    """
    Replace the token embeddings for a cell line and a gene with custom embeddings.

    Args:
        token_embeddings (torch.Tensor): Original token embeddings of shape [seq_len, hidden_dim].
        cell_line_startidx (int): Index of the cell line token to be replaced.
        gene_startidx (int): Index of the gene token to be replaced.
        cell_line_embedding (torch.Tensor): Custom embedding for the cell line (default: random).
        gene_embedding (torch.Tensor): Custom embedding for the gene (default: random).

    Returns:
        torch.Tensor: Modified token embeddings of shape [seq_len, hidden_dim].
    """
    # Generate random embeddings if none are provided
    if cell_line_embedding is None:
        cell_line_embedding = torch.zeros(token_embeddings.shape[1]).to(token_embeddings.device)
    if gene_embedding is None:
        gene_embedding = torch.zeros(token_embeddings.shape[1]).to(token_embeddings.device)

    # Replace the token embeddings
    token_embeddings[cell_line_startidx] = cell_line_embedding
    # token_embeddings[gene_startidx] = gene_embedding
    token_embeddings[gene_startidx] = gene_embedding

    return token_embeddings

def add_omics_embedding(token_embeddings, attention_mask, omics_embedding=None, omics_start_idx=None, max_tokens=512):
    """
    Add a 6x768 omics embedding to the token embeddings and adjust the attention mask.
    
    Args:
        token_embeddings (torch.Tensor): Token embeddings of shape (512, 768).
        attention_mask (torch.Tensor): Attention mask of shape (512,). 
        omics_embedding (torch.Tensor): Omics embeddings of shape (6, 768).
        omics_start_idx (int): The index in the token embeddings where the omics embedding should be inserted.
        max_tokens (int, optional): The maximum length of tokens, default is 512.
    
    Returns:
        updated_token_embeddings (torch.Tensor): The token embeddings with the omics embeddings added.
        updated_attention_mask (torch.Tensor): The updated attention mask with the omics embedding added.
    """
    if omics_embedding is None:
        omics_embedding = torch.ones(6, 768).to(token_embeddings.device)
        
    if omics_embedding.shape == (768, 6):
        omics_embedding = omics_embedding.T 
    assert omics_embedding.shape == (6, 768), f"Expected omics_embedding shape (6, 768), but got {omics_embedding.shape}"
    
    # Prepare the final token embeddings with omics
    updated_token_embeddings = torch.cat(
        (token_embeddings[:omics_start_idx, :], omics_embedding, token_embeddings[omics_start_idx:, :]), dim=0
    )
    updated_token_embeddings = updated_token_embeddings.to(device)
    # Ensure the token embeddings are the correct size (512 tokens + 6 for omics = 518 tokens)
    if updated_token_embeddings.size(0) > max_tokens:
        updated_token_embeddings = updated_token_embeddings[:max_tokens, :]

    # Prepare the final attention mask
    updated_attention_mask = torch.cat(
        (attention_mask[:omics_start_idx], torch.ones(6).to(attention_mask.device), attention_mask[omics_start_idx:]), dim=0
    )
    updated_attention_mask = updated_attention_mask.to(device)
    if updated_attention_mask.size(0) > max_tokens:
        updated_attention_mask = updated_attention_mask[:max_tokens]
        
    return updated_token_embeddings, updated_attention_mask

def load_omics_data(omics_data_dir, cell_line_id):
    """
    Load six omics data files and extract the row corresponding to the given cell line ID.
    
    Args:
        omics_data_dir (str): Path to the folder containing omics data files.
        cell_line_id (str): The DepMap ID of the cell line.

    Returns:
        dict: A dictionary containing vectors for six omics data channels.
    """
    omics_files = [
        "selected_cell_cn_raw.csv", 
        "selected_cell_dep_raw.csv",
        "selected_cell_eff_raw.csv",
        "selected_cell_exp_raw.csv",
        "selected_cell_met_raw.csv",
        "selected_cell_mut_raw.csv",
    ]
    default_vector = np.zeros(3456, dtype=np.float32)
    omics_data = {}
    for idx, omics_file in enumerate(omics_files):
        file_path = os.path.join(omics_data_dir, omics_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found!")

        # Load file and filter row for the given cell line ID
        df = pd.read_csv(file_path)
        row = df[df['DepMap_ID'] == cell_line_id]
        if row.empty:
            print(f"Warning: Cell line ID {cell_line_id} not found in {omics_file}. Using default vector for om_{idx}, length {len(default_vector)}.")
            omics_data[f"om_{idx}"] = default_vector
        else:
            # Safely access the row and convert to float32
            omics_data[f"om_{idx}"] = row.iloc[0, 1:].values.astype(np.float32)

    return omics_data

def create_dataset_for_cell_line(
    biobert, tokenizer, device, cell_line_case_name, cell_line_id, cell_line_embedding, omics_data_dir, text_template_path, target_df, genept_embeddings, kg_entity_embeddings, kg_entity_to_id, save_path
):
    """
    Create a GenePairDataset for a specific cell line.

    Args:
        biobert (nn.Module): BioBERT model instance.
        tokenizer (PreTrainedTokenizer): Tokenizer for BioBERT.
        device (torch.device): Device for processing.
        cell_line (str): Cell line name (e.g., "A375").
        cell_line_id (str): Cell line ID (e.g., "ACH-000219").
        cell_line_embedding (torch.Tensor): Embedding for the cell line.
        text_template_path (str): Path to the text templates JSON.
        target_df (pd.DataFrame): DataFrame of gene pairs and their labels.
        genept_embeddings (dict): Preloaded GenePT embeddings for genes.
        kg_embeddings (dict): Preloaded KG embeddings for genes.
        save_path (str): Path to save the dataset.

    Returns:
        None
    """
    # Load omics data for the current cell line
    omics_data = load_omics_data(omics_data_dir, cell_line_id)
    # print(omics_data)
    # print(omics_data["om_0"])
    
    # Load text templates 
    cell_line_name = cell_line_case_name.split("_")[0]
    with open(os.path.join(text_template_path, f"{cell_line_name}_gene_omics_descriptions.json"), "r") as file:
        text_templates = json.load(file)

    # Prepare data
    embeddings = []
    labels = []
    filtered_rows = []

    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc=f"Processing {cell_line_case_name}"):
        gene_1 = row["gene_1"]
        gene_2 = row["gene_2"]
        label = 1 if row["SL_or_not"] == "SL" else 0
        
        # Skip if genes are missing embeddings
        if gene_1 not in genept_embeddings or gene_2 not in genept_embeddings:
            # print(gene_1, gene_2)
            continue
        if gene_1 not in kg_entity_to_id or gene_2 not in kg_entity_to_id:
            # print(gene_1, gene_2)
            continue
        
        if gene_1 not in text_templates or gene_2 not in text_templates:
            # print(gene_1, gene_2)
            continue
        
        filtered_rows.append(row)
        
        text_1 = text_templates[gene_1]
        _, attention_mask_1, token_embeddings_1, cell_line_startidx_1, gene_startidx_1 = get_token_embeddings(biobert, tokenizer, text_1, device)
        text_2 = text_templates[gene_2]
        _, attention_mask_2, token_embeddings_2, cell_line_startidx_2, gene_startidx_2 = get_token_embeddings(biobert, tokenizer, text_2, device)

        #### if add omics:
        omics_start_idx = 7
        token_emb_1, attention_mask_1 = add_omics_embedding(token_embeddings_1, attention_mask_1, omics_embedding=None, omics_start_idx=omics_start_idx, max_tokens=512)
        token_emb_2, attention_mask_2 = add_omics_embedding(token_embeddings_2, attention_mask_2, omics_embedding=None, omics_start_idx=omics_start_idx, max_tokens=512)
        
        #### Replace target embeddings
        omics_length = 6
        token_emb_1 = replace_token_embeddings(token_emb_1, cell_line_startidx_1, gene_startidx=(omics_start_idx+omics_length), cell_line_embedding=cell_line_embedding )
        token_emb_2 = replace_token_embeddings(token_emb_2, cell_line_startidx_2, gene_startidx=(omics_start_idx+omics_length), cell_line_embedding=cell_line_embedding )
        ####
        
        embedding = (
            cell_line_embedding.cpu().numpy(),
            omics_data["om_0"],  # CN data
            omics_data["om_1"],  # DEP data
            omics_data["om_2"],  # EFF data
            omics_data["om_3"],  # EXP data
            omics_data["om_4"],  # MET data
            omics_data["om_5"],  # MUT data
            token_emb_1.cpu().numpy(),
            attention_mask_1.cpu().numpy(),
            genept_embeddings[gene_1],
            kg_entity_embeddings[kg_entity_to_id.get(gene_1)],
            # kg_emb1,
            token_emb_2.cpu().numpy(),
            attention_mask_2.cpu().numpy(),
            genept_embeddings[gene_2],
            kg_entity_embeddings[kg_entity_to_id.get(gene_2)],
            # kg_emb2,
        )
        embeddings.append(embedding)
        label_tensor = torch.tensor(label).long()
        labels.append(label_tensor)
        
    filtered_df = pd.DataFrame(filtered_rows)
    output_path = f"{cell_line_df_folder}/{cell_line_case_name}_filtered.csv"
    filtered_df.to_csv(output_path, index=False)
    ## Save dataset
    dataset = GenePairDataset(embeddings, labels)
    dataset_name = f"{cell_line_case_name}.pt"
    torch.save(dataset, os.path.join(save_path, dataset_name))
    print(f"Dataset for {cell_line_case_name} saved at {os.path.join(save_path, dataset_name)}")


########################################################

# save_path = "/home/tinglu/MLLM4SL/data/gene_pair_datasets3_ori_omics"
save_path = "/home/tinglu/MLLM4SL/data/case_study_compare/datasets"
# save_path = "/home/tinglu/MLLM4SL/data/SLBench_overlap_datasets/remain_datasets"



for cell_line_case_name, target_df in cell_line_dfs.items():
    cell_line_id = cell_lines_info[cell_line_case_name.split("_")[0]]["cell_line_id"]
    cell_line_embedding = cell_line_embeddings[cell_line_case_name.split("_")[0]]["embedding"]
    
    create_dataset_for_cell_line(
        biobert,
        tokenizer,
        device,
        cell_line_case_name,
        cell_line_id=cell_line_id,
        cell_line_embedding=cell_line_embedding,
        omics_data_dir="/home/tinglu/MLLM4SL/data/omics_data/selected",
        text_template_path="/home/tinglu/MLLM4SL/data/text_template",
        target_df=target_df,
        genept_embeddings=genept_gene_embeddings,
        kg_entity_embeddings=kg_entity_embeddings,
        kg_entity_to_id=kg_entity_to_id,
        save_path=save_path,
    )
    # break

#  python -m src.preprocess

