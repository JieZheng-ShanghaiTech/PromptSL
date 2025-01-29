import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
from sklearn.metrics import auc, roc_auc_score, f1_score, balanced_accuracy_score,accuracy_score, precision_recall_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from .model import Projector, FuseModel, Predictor, BioBERTEncoder, EarlyStopping, Encoder_om
from .data_loader import kmeans_undersample, random_undersample, prepare_dataloader, load_train_test_datasets_infer
from .train import validate_model, load_models, validate_model_with_reference, eval_model
from .main import load_train_test_datasets, load_datasets
import logging
import pickle
import ast

device = torch.device("cuda:1")
print(device)
print(torch.cuda.is_available()) # If False and AssertionError: Torch not compiled with CUDA enabled: conda activate only one env: llm4sl


model_ids = [
    "2025-01-28_11-47-22_cuda1"
]

# cell_lines = ["22RV1", "GI1", 'HS936T']
# cell_lines = ['A549', 'IPC298', 'MEWO', 'PK1']
cell_lines = ['MEWO']

## SLBench
# model_folder = "train_results-4in1-SLBench"
# result_dir = "/home/tinglu/MLLM4SL/save/eval_results-4in1-SLBench" 
# # result_dir = "/home/tinglu/MLLM4SL/save/eval_results-4to1-SLBench" 

## SLKB - full data
# model_folder = "train_results-4in1-noprompt"
# result_dir = "/home/tinglu/MLLM4SL/save/eval_results-4in1-ablation" 
model_folder = "train_results"
result_dir = "./save/eval_results" 

## Case study
# cell_lines = ['A549_conflict', 'IPC298_conflict', 'MEWO_conflict', 'PK1_conflict']
# # cell_lines = ['A549_consist', 'IPC298_consist', 'MEWO_consist', 'PK1_consist']
# model_folder = "train_results-4in1-mlp_compare"
# # model_folder = "train_results-4in1"
# result_dir = "/home/tinglu/MLLM4SL/save/case_study_reference_mlp_compare/conflict_test"
# # result_dir = "/home/tinglu/MLLM4SL/save/case_study_reference_mlp_compare/consist_test"

# datasets_folder = "/home/tinglu/MLLM4SL/data/case_study_compare" # ground truth data

# ## when integarte 4 cell line results and form matching.csv
# target_file_path = "/home/tinglu/MLLM4SL/data/case_study_compare/conflict_gene_pairs250.csv"
# # target_file_path = "/home/tinglu/MLLM4SL/data/case_study_compare/consist_gene_pairs136.csv"



gene_startidx = 13
neg_pos_ratio = 1.0 # default = 1.0, for all samples: use -1
biobert_model_name = "/home/tinglu/LLM4SL/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
biobert = AutoModel.from_pretrained(biobert_model_name)
biobert.to(device)

def evaluate_cell_lines(cell_lines, model_ids, neg_pos_ratio=1.0, random_seeds=[42, 123, 456, 789, 1010]):
    """
    Evaluates a model on multiple cell lines systematically.
    
    Args:
        cell_lines (list): List of cell line identifiers (e.g., ["A375", "A549"]).
        model_id (str): The ID of the model checkpoint to load.
        neg_pos_ratio (float): The negative-to-positive ratio for undersampling. Default is 1.0.
        random_seeds (list): List of seeds for random undersampling. Default is 5 predefined seeds.
    """
    for model_id in model_ids:
        log_file_path = os.path.join(result_dir, f"{model_id}.txt")

        # Ensure the directory exists
        os.makedirs(result_dir, exist_ok=True)
        logging.shutdown()  # Close any previously configured loggers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)  # Clear existing handlers
        logging.basicConfig(
            filename=log_file_path,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        def log_and_print(message):
            print(message)
            logging.info(message)
            
        log_and_print(f"\nEvaluating Model ID: **{model_id}**\n{'='*50}")

        # Define paths and initialize models
        save_path = f"/home/tinglu/MLLM4SL/save/{model_folder}/{model_id}/model/model_ckpt.pt"
        projector = Projector().to(device)
        fuse_model = FuseModel().to(device)
        predictor = Predictor().to(device)
        bio_encoder = BioBERTEncoder(biobert).to(device)
        encoder_om_cn = Encoder_om(input_dim=4078).to(device)
        encoder_om_dep = Encoder_om(input_dim=3456).to(device)
        encoder_om_eff = Encoder_om(input_dim=3456).to(device)
        encoder_om_exp = Encoder_om(input_dim=4078).to(device)
        encoder_om_met = Encoder_om(input_dim=2279).to(device)
        encoder_om_mut = Encoder_om(input_dim=3937).to(device)

        # Load the models from the checkpoint
        # projector, bio_encoder, fuse_model, predictor = load_models(save_path, projector, bio_encoder, fuse_model, predictor)
        projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut = load_models(save_path, projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut, device)

        
        # Evaluate each cell line
        for eval_cell_line in cell_lines:
            log_and_print("********************************")
            log_and_print(f"Model id: **{model_id}**. Eval on cell line: **{eval_cell_line}**. Neg_pos_ratio: **{neg_pos_ratio}**.")

            # datasets = load_datasets(eval_cell_line)
            # embeddings, labels = kmeans_undersample(datasets[0], neg_pos_ratio=neg_pos_ratio)
            # log_and_print("Kmeans undersampling")
            # _, eval_loader = prepare_dataloader(embeddings, labels, batch_size=16, train_split=0.8)
            
            # val_loss, val_metrics = eval_model(
            #     eval_loader, projector, bio_encoder, fuse_model, predictor, 
            #     encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut,
            #     criterion=nn.BCELoss(),
            #     omics_startidx=7,
            #     gene_startidx=gene_startidx,
            #     device=device
            # )
            
            # log_and_print(f"Validation Loss: {val_loss:.4f}")
            # log_and_print("Validation Metrics:")
            # log_and_print(f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}")
            
            # ############################
            # with open(f"/home/tinglu/MLLM4SL/data/gene_pair_datasets3_ori_omics/{eval_cell_line}_emb_labels_km_test.pkl", "rb") as f:
            #     data = pickle.load(f)
            #     embeddings, labels = data["embeddings"], data["labels"]
            # _, eval_loader = prepare_dataloader(embeddings, labels, batch_size=16, train_split=0.0)
            # log_and_print("Kmeans test sets")
            # val_loss, val_metrics = eval_model(
            #     eval_loader, projector, bio_encoder, fuse_model, predictor, 
            #     encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut,
            #     criterion=nn.BCELoss(),
            #     omics_startidx=7,
            #     gene_startidx=gene_startidx,
            #     device=device
            # )
            
            # log_and_print(f"Validation Loss: {val_loss:.4f}")
            # log_and_print("Validation Metrics:")
            # log_and_print(f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}")
            
            ############################
            results = []
            if neg_pos_ratio != -1:
                for seed in random_seeds:
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    embeddings, labels = random_undersample(datasets[0], neg_pos_ratio=neg_pos_ratio, random_seed=seed)
                    _, eval_loader = prepare_dataloader(embeddings, labels, batch_size=16, train_split=0.8)
                    
                    val_loss, val_metrics = eval_model(
                        eval_loader, projector, bio_encoder, fuse_model, predictor, 
                        encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut,
                        criterion=nn.BCELoss(),
                        omics_startidx=7,
                        gene_startidx=gene_startidx,
                        device=device
                    )

                    log_and_print(f"Random Seed: {seed}")
                    log_and_print(f"Validation Loss: {val_loss:.4f}")
                    log_and_print(f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}")
                    
                    results.append({
                        "loss": val_loss,
                        "accuracy": val_metrics['accuracy'],
                        "f1": val_metrics['f1'],
                        "roc_auc": val_metrics['roc_auc'],
                        "aupr": val_metrics['aupr'],
                        "bacc": val_metrics['bacc'],
                        "recall": val_metrics['recall']
                    })
            
            # Compute average metrics across seeds
            if results:
                avg_results = {key: np.mean([result[key] for result in results]) for key in results[0].keys()}
                log_and_print(f"\nAverage Results for Cell Line: **{eval_cell_line}**, {model_id}")
                log_and_print(f"Validation Loss: {avg_results['loss']:.4f}")
                log_and_print(f"Acc: {avg_results['accuracy']:.4f}, F1: {avg_results['f1']:.4f}, BACC: {avg_results['bacc']:.4f}, AUC: {avg_results['roc_auc']:.4f}, AUPR: {avg_results['aupr']:.4f}, Recall: {avg_results['recall']:.4f}")
                log_and_print("\n\n")
            

def evaluate_on_fixed_cell_lines(cell_lines, model_ids):
    """
    Evaluates a model on multiple cell lines systematically.
    
    Args:
        cell_lines (list): List of cell line identifiers (e.g., ["A375", "A549"]).
        model_id (str): The ID of the model checkpoint to load.
        neg_pos_ratio (float): The negative-to-positive ratio for undersampling. Default is 1.0.
        random_seeds (list): List of seeds for random undersampling. Default is 5 predefined seeds.
    """
    for model_id in model_ids:
        log_file_path = os.path.join(result_dir, f"{model_id}.txt")

        # Ensure the directory exists
        os.makedirs(result_dir, exist_ok=True)
        logging.shutdown()  # Close any previously configured loggers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)  # Clear existing handlers
        logging.basicConfig(
            filename=log_file_path,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        def log_and_print(message):
            print(message)
            logging.info(message)
            
        log_and_print(f"\nEvaluating Model ID: **{model_id}**\n{'='*50}")

        # Define paths and initialize models
        save_path = f"./save/{model_folder}/{model_id}/model/model_ckpt.pt"
        projector = Projector(dropout_rate=0.1).to(device)
        fuse_model = FuseModel().to(device)
        predictor = Predictor().to(device)
        bio_encoder = BioBERTEncoder(biobert).to(device)
        encoder_om_cn = Encoder_om(input_dim=4078).to(device)
        encoder_om_dep = Encoder_om(input_dim=3456).to(device)
        encoder_om_eff = Encoder_om(input_dim=3456).to(device)
        encoder_om_exp = Encoder_om(input_dim=4078).to(device)
        encoder_om_met = Encoder_om(input_dim=2279).to(device)
        encoder_om_mut = Encoder_om(input_dim=3937).to(device)

        # Load the models from the checkpoint
        # projector, bio_encoder, fuse_model, predictor = load_models(save_path, projector, bio_encoder, fuse_model, predictor)
        projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut = load_models(save_path, projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut, device)

        
        # Evaluate each cell line
        for eval_cell_line in cell_lines:
            log_and_print(f"Model id: **{model_id}**. Eval on cell line: **{eval_cell_line}**.")
            
            # Evaluate using the specified neg_pos_ratio
            # train_emb, train_labels, test_emb, test_labels = load_train_test_datasets(eval_cell_line, folder="/home/tinglu/MLLM4SL/data/SLBench_overlap_datasets")
            train_emb, train_labels, test_emb, test_labels = load_train_test_datasets(eval_cell_line, folder="./data/dataset_example")
            _, eval_loader = prepare_dataloader(test_emb, test_labels, batch_size=16, train_split=0.0)

                
            val_loss, val_metrics = eval_model(
                eval_loader, projector, bio_encoder, fuse_model, predictor, 
                encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_o1m_exp, encoder_om_met, encoder_om_mut,
                criterion=nn.BCELoss(),
                omics_startidx=7,
                gene_startidx=gene_startidx,
                device=device
            )
            
            log_and_print(f"Validation Loss: {val_loss:.4f}")
            log_and_print("Validation Metrics:")
            log_and_print(f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}\n\n")
            
def evaluate_on_fixed_cell_lines_reference(cell_lines, model_ids):

    for model_id in model_ids:
        model_folder_path = os.path.join(result_dir, model_id)
        os.makedirs(model_folder_path, exist_ok=True)  # Create folder if it doesn't exist

        log_file_path = os.path.join(model_folder_path, f"{model_id}.txt")
        logging.shutdown()  # Close any previously configured loggers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)  # Clear existing handlers
        logging.basicConfig(
            filename=log_file_path,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        def log_and_print(message):
            print(message)
            logging.info(message)
            
        log_and_print(f"\nEvaluating Model ID: **{model_id}**\n{'='*50}")

        # Define paths and initialize models
        save_path = f"/home/tinglu/MLLM4SL/save/{model_folder}/{model_id}/model/model_ckpt.pt"
        projector = Projector(dropout_rate=0.1).to(device)
        fuse_model = FuseModel().to(device)
        predictor = Predictor().to(device)
        bio_encoder = BioBERTEncoder(biobert).to(device)
        encoder_om_cn = Encoder_om(input_dim=4078).to(device)
        encoder_om_dep = Encoder_om(input_dim=3456).to(device)
        encoder_om_eff = Encoder_om(input_dim=3456).to(device)
        encoder_om_exp = Encoder_om(input_dim=4078).to(device)
        encoder_om_met = Encoder_om(input_dim=2279).to(device)
        encoder_om_mut = Encoder_om(input_dim=3937).to(device)

        # Load the models from the checkpoint
        # projector, bio_encoder, fuse_model, predictor = load_models(save_path, projector, bio_encoder, fuse_model, predictor)
        projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut = load_models(save_path, projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut, device)

        # Evaluate each cell line
        for eval_cell_line in cell_lines:
            log_and_print(f"Model id: **{model_id}**. Eval on cell line: **{eval_cell_line}**.")
            
            ground_truth_file = f"{datasets_folder}/{eval_cell_line}_test_genept_f5.csv"
            ground_truth_df = pd.read_csv(ground_truth_file)
            ground_truth_df['genept_1'] = ground_truth_df['genept_1'].apply(ast.literal_eval)
            ground_truth_df['genept_2'] = ground_truth_df['genept_2'].apply(ast.literal_eval)
            
            # Evaluate using the specified neg_pos_ratio
            train_emb, train_labels, test_emb, test_labels = load_train_test_datasets_infer(eval_cell_line, folder=f"{datasets_folder}/datasets", shuffle=False)
            _, eval_loader = prepare_dataloader(test_emb, test_labels, batch_size=16, shuffle=False, train_split=0.0)

            val_loss, val_metrics, reference = validate_model_with_reference(
                eval_loader, projector, bio_encoder, fuse_model, predictor, 
                encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut,
                criterion=nn.BCELoss(),
                omics_startidx=7,
                gene_startidx=gene_startidx,
                device=device,
                output_reference_labels=True
            )
            # Correct gene pairs in reference
            def match_gene_pair(row):
                for _, ground_row in ground_truth_df.iterrows():
                    if (row['gene_pair_1_f5'] == ground_row['genept_1'] and 
                        row['gene_pair_2_f5'] == ground_row['genept_2']):
                        # print(ground_row['gene_pair'])
                        return ground_row['gene_pair']
                return None 
            
            reference_df = pd.DataFrame(reference)
            reference_df['gene_pair_map_back'] = reference_df.apply(match_gene_pair, axis=1)
            reference_df = reference_df.drop(columns=['gene_pair_1_f5', 'gene_pair_2_f5'])
            
            output_reference_path = f"{model_folder_path}/{eval_cell_line.split('_')[0]}_inference.csv"
            reference_df.to_csv(output_reference_path, index=False)
            
            log_and_print(f"Validation Loss: {val_loss:.4f}")
            log_and_print("Validation Metrics:")
            log_and_print(f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}\n\n")

        # Step 1: Create {cell_line}_infer.csv files with correct gene_pair from {cell_line}_test_filtered.csv
        for cell_line in cell_lines:
            test_filtered_path = f"{datasets_folder}/{cell_line}_test_filtered.csv"
            inference_path = f"{model_folder_path}/{cell_line.split('_')[0]}_inference.csv"
            df_test_filtered = pd.read_csv(test_filtered_path)
            df_inference = pd.read_csv(inference_path)
            
            df_test_filtered = df_test_filtered[['gene_pair', 'cell_line_origin', 'SL_or_not']]

            # df_merged = pd.concat([df_test_filtered.reset_index(drop=True), df_inference[['predicted_label', 'true_label', 'gene_pair_map_back']].reset_index(drop=True)], axis=1)
            df_merged = pd.concat([df_test_filtered.reset_index(drop=True), df_inference[['predicted_label', 'true_label']].reset_index(drop=True)], axis=1)
            
            df_merged.rename(
                columns={
                    'predicted_label': f"predicted_{cell_line.split('_')[0]}",
                    'true_label': f"true_label_{cell_line.split('_')[0]}",
                    'gene_pair_map_back': f"map_back_{cell_line.split('_')[0]}"
                },
                inplace=True
            )
            # Save the merged DataFrame
            output_path = f"{model_folder_path}/{cell_line.split('_')[0]}_infer.csv"
            df_merged.to_csv(output_path, index=False)
            print(f"Merged file saved for {cell_line} to {output_path}")
            
        # Step 2: 
        df_target = pd.read_csv(target_file_path)
        df_target['con_gene_pairs'] = df_target['con_gene_pairs'].astype(str)  # Ensure string type for comparison
        
        # Map back the inference results
        for cell_line in cell_lines:
            # Load the {cell_line}_infer.csv file
            infer_path = f"{model_folder_path}/{cell_line.split('_')[0]}_infer.csv"
            df_infer = pd.read_csv(infer_path)
            df_infer.rename(
                columns={
                    'gene_pair': 'mapped_gene_pair',
                    f"predicted_{cell_line.split('_')[0]}": f"predicted_{cell_line.split('_')[0]}",
                    f"true_label_{cell_line.split('_')[0]}": f"true_label_{cell_line.split('_')[0]}",
                },
                inplace=True
            )
            df_target = pd.merge(
                df_target,
                df_infer[['mapped_gene_pair', f"predicted_{cell_line.split('_')[0]}", f"true_label_{cell_line.split('_')[0]}"]],
                how='left',  # Ensure we keep all rows from the target file
                left_on='con_gene_pairs',
                right_on='mapped_gene_pair'
            )
            df_target.drop(columns=['mapped_gene_pair'], inplace=True)
            
        # Save the updated DataFrame
        final_output_path = f"{model_folder_path}/{model_id}_matching.csv"
        df_target.to_csv(final_output_path, index=False)
        print(f"Final mapped results saved to {final_output_path}")
        

# evaluate_cell_lines(cell_lines, model_ids, neg_pos_ratio=neg_pos_ratio) # full datasets
evaluate_on_fixed_cell_lines(cell_lines, model_ids) # SLBench datasets
# evaluate_on_fixed_cell_lines_reference(cell_lines, model_ids) # SLBench case study test datasets

