from tqdm import tqdm
from data.utils import get_omics_data, format_cell_line_description
import pandas as pd
import json
from pykeen.datasets import PrimeKG
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import os
import pickle

"""
    prepare the cell line embedding file, and the text template for each gene pair in each cell line,
    which we can directly use in data preprocess.
    output file: 
        /home/tinglu/MLLM4SL/data/text_template/...
        /home/tinglu/MLLM4SL/data/cell_line_emb2.pkl...
"""
device = torch.device("cuda:2")
print(device)

get_cell_line_emb = 0
prepare_text_template = 0

### Load BioBERT and get cell line embedding (768)
model_name = "/home/tinglu/LLM4SL/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
biobert = AutoModel.from_pretrained(model_name)
biobert.to(device)

### get_biobert_embedding of cell lines

def get_biobert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=False, padding="max_length")
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = biobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
    return pooled_output.detach()

# # Directory to save embeddings
# output_dir = "/home/tinglu/MLLM4SL/data/cell_line_emb/"
# os.makedirs(output_dir, exist_ok=True)

### prepare omics data
# Function to load omics data
def load_omics_data():
    # Load required omics data
    with open('/home/tinglu/LLM4SL/CCLE/name2ens.json', 'r') as file:
        name_id_mapping = json.load(file)

    ge_data = pd.read_csv('/home/tinglu/LLM4SL/CCLE/GE.csv')
    mu_data = pd.read_csv('/home/tinglu/LLM4SL/CCLE/MU.csv', low_memory=False)
    cn_gene_data = pd.read_csv('/home/tinglu/LLM4SL/CCLE/CNGene.csv')

    # Set index for GE and CNV data
    ge_data.set_index('Unnamed: 0', inplace=True)
    cn_gene_data.set_index('Unnamed: 0', inplace=True)

    return name_id_mapping, ge_data, mu_data, cn_gene_data

# Load labels and omics data
name_id_mapping, ge_data, mu_data, cn_gene_data = load_omics_data()

labels_df = pd.read_csv('/home/tinglu/LLM4SL/SLKB/rawSL.csv')
# Define cell line DataFrames
cell_line_dfs = {
    "PK1": labels_df[labels_df['cell_line_origin'] == 'PK1'].copy(),
    "IPC298": labels_df[labels_df['cell_line_origin'] == 'IPC298'].copy(),
    "MEWO": labels_df[labels_df['cell_line_origin'] == 'MEWO'].copy(),
    "A549": labels_df[labels_df['cell_line_origin'] == 'A549'].copy(),
    "22RV1": labels_df[labels_df['cell_line_origin'] == '22RV1'].copy(),
    "GI1": labels_df[labels_df['cell_line_origin'] == 'GI1'].copy(),
    "HS936T": labels_df[labels_df['cell_line_origin'] == 'HS936T'].copy(),
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


if get_cell_line_emb:
    # Generate embeddings for all cell lines
    cell_line_embeddings = {}
    for cell_line, details in cell_lines_info.items():
        description = details["description"]
        embedding = get_biobert_embedding(description)[0]  # Get the 768-dimensional embedding
        cell_line_embeddings[cell_line] = {
            "cell_line_id": details["cell_line_id"],
            "embedding": embedding
        }
        print(f"Generated embedding for {cell_line}")

    # Save the dictionary to a file
    output_file = "/home/tinglu/MLLM4SL/data/cell_line_emb2.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(cell_line_embeddings, f)

    print(f"Saved all embeddings to {output_file}")


def process_cell_line_data(cell_line, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data, save_path):
    
    # Get unique genes
    genes_in_df = pd.unique(target_df[['gene_1', 'gene_2']].values.ravel())
    print(f"Processing {cell_line}...{genes_in_df.size}")
    
    # Dictionary to store gene descriptions
    gene_descriptions = {}
    for gene_name in tqdm(genes_in_df, desc=f"Processing genes for {cell_line}"):
        gene_id = name_id_mapping.get(gene_name)
        
        if gene_id is None:
            continue

        # Retrieve omics data
        omics_data = get_omics_data(ge_data, cn_gene_data, mu_data, cell_line_id, gene_name, gene_id)
        if any(data is None for data in omics_data):
            continue

        # Generate description
        gene_descriptions[gene_name] = format_cell_line_description(cell_line, gene_name, omics_data)
        
    # Save the dictionary to a file
    save_file = f"{save_path}/{cell_line}_gene_omics_descriptions.json"
    with open(save_file, "w", encoding="utf-8") as file:
        json.dump(gene_descriptions, file, indent=4, ensure_ascii=False)
    
    print(f"Saved gene descriptions for {cell_line} to {save_file}")


if prepare_text_template:

    # Directory to save outputs
    save_path = "/home/tinglu/MLLM4SL/data/text_template"
    os.makedirs(save_path, exist_ok=True)

    cell_line_done = ["A375", "A549", "PK1", "IPC298", "MEWO"]
    # Process each cell line
    for cell_line_case_name, target_df in cell_line_dfs.items():
        # if cell_line in cell_line_done:
        #     continue
        cell_line_name = cell_line_case_name.split("_")[0]
        cell_line_id = cell_lines_info[cell_line_name]["cell_line_id"]
        process_cell_line_data(cell_line_name, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data, save_path)






