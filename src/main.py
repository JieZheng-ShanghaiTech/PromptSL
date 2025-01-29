import torch
import os
import yaml
import pickle
import logging
import numpy as np
from src.data_loader import kmeans_undersample, random_undersample, prepare_dataloader
from src.data_loader import GenePairDataset
import argparse
from src.train import train_model
from src.model import BioBERTEncoder, Projector, FuseModel, Predictor
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from .data_loader import load_train_test_datasets, load_datasets
from sklearn.model_selection import train_test_split

def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logger(name, log_file_path, level=logging.INFO):
    """Set up a logger that writes to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Training configuration for MLLM4SL.")
    
    # Dataset and model paths
    parser.add_argument("--dataset_path", type=str, default="/home/tinglu/MLLM4SL/data/gene_pair_datasets/A549_dataset.pt", help="Path to the dataset file.")
    parser.add_argument("--biobert_path", type=str, default="/home/tinglu/LLM4SL/biobert-base-cased-v1.2", help="Path to the pretrained BioBERT model.")
    # parser.add_argument("--save_folder", type=str, default="/home/tinglu/MLLM4SL/save/train_results", help="Path to the save train results.")
    parser.add_argument("--save_folder", type=str, default="train_results", help="Path to the save train results.")
    parser.add_argument("--cell_line", type=str, choices=["A375", "A549", "PK1", "IPC298", "MEWO", "all"], default="A549", 
                        help="Specify the cell line to train on (or 'all' to use all datasets combined).")
    parser.add_argument("--scenario", type=str, choices=["C1", "C2", "C3"], default="C1", help="Specify the scenario (C1, C2 or C3) to train on.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train test split rate.")
    
    # Hyperparameters
    parser.add_argument("--predictor_lr", type=float, default=1e-4, help="Learning rate for the predictor.")
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="Learning rate for the encoder.")
    parser.add_argument("--projector_lr", type=float, default=1e-4, help="Learning rate for the projector.")
    parser.add_argument("--om_encoder_lr", type=float, default=1e-4, help="Learning rate for the omics encoder.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--early_stop_patience", type=int, default=30, help="Early stop patience.")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Learning rate reducing factor.")
    parser.add_argument("--lr_patience", type=int, default=5, help="Patience for learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--device", type=int, default=0, help="Select the device.")
    
    return parser.parse_args()



def create_save_folder(device, save_folder):
    current_time = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_cuda{device}")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_folder_path = os.path.join(base_path, "save", save_folder, current_time)
    os.makedirs(save_folder_path, exist_ok=True)
    return save_folder_path

def main():
    args = parse_arguments()

    # Create save folder
    save_folder = create_save_folder(args.device, args.save_folder)
    print(f"Results will be saved in: {save_folder}")
    # Set up logging
    log_file_path = os.path.join(save_folder, "training.log")
    logger = setup_logger("training_logger", log_file_path)
    logger.info("Starting training process.")

    # Log the hyperparameters
    logger.info("Hyperparameters:")
    logger.info(f"  Predictor learning rate: {args.predictor_lr}")
    logger.info(f"  Encoder learning rate: {args.encoder_lr}")
    logger.info(f"  Projector learning rate: {args.projector_lr}")
    logger.info(f"  Omics Encoder learning rate: {args.om_encoder_lr}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Number of epochs: {args.num_epochs}")
    logger.info(f"  Early stop patience: {args.early_stop_patience}")
    logger.info(f"  Learning rate patience: {args.lr_patience}")
    logger.info(f"  Learning rate reduce factor: {args.lr_factor}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Dropout rate: {args.dropout_rate}")
    logger.info(f"  Device: {args.device}")

    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load dataset
    logger.info(f"Loading dataset for {args.cell_line}.")
    
    ############# If full data in SLKB: #############
    
    # logger.info(f"Scenario {args.scenario}.")
    # logger.info(f"Train test split rate {args.train_split}.")
    
    # datasets = load_datasets(args.cell_line, logger=logger)
    # if args.cell_line == "all":
    #     combined_dataset = []
    #     for dataset in datasets:
    #         combined_dataset.extend(dataset)
    #     dataset = combined_dataset
    # else:
    #     dataset = datasets[0]

    # logger.info("Performing undersampling.")
    # embeddings, labels = kmeans_undersample(dataset, neg_pos_ratio=1.0)
    # # embeddings, labels = random_undersample(dataset, neg_pos_ratio=1.0, random_seed=42)
    
    # train_loader, val_loader = prepare_dataloader(embeddings, labels, batch_size=args.batch_size, train_split=args.train_split, scenario=args.scenario, logger=logger)

    #############
    # emb_train, emb_test, labels_train, labels_test = train_test_split(embeddings, labels, train_size=0.8, random_state=42)
    # with open(f"/home/tinglu/MLLM4SL/data/gene_pair_datasets3_ori_omics/{args.cell_line}_emb_labels_km_train.pkl", "wb") as f:
    #     pickle.dump({"embeddings": emb_train, "labels": labels_train}, f)
    # with open(f"/home/tinglu/MLLM4SL/data/gene_pair_datasets3_ori_omics/{args.cell_line}_emb_labels_km_test.pkl", "wb") as f:
    #     pickle.dump({"embeddings": emb_test, "labels": labels_test}, f)
    
    # ############# If SLBench overlap data: #############
    
    # train_emb, train_labels, test_emb, test_labels = load_train_test_datasets(args.cell_line, folder="/home/tinglu/MLLM4SL/data/SLBench_overlap_datasets", logger=logger)
    # # train_emb, train_labels, test_emb, test_labels = load_train_test_datasets(args.cell_line, folder="/home/tinglu/MLLM4SL/data/SLBench_overlap_datasets/remain_datasets", logger=logger)
    # _, train_loader = prepare_dataloader(train_emb, train_labels, batch_size=args.batch_size, train_split=0.0, scenario=args.scenario, logger=logger)
    # _, val_loader = prepare_dataloader(test_emb, test_labels, batch_size=args.batch_size, train_split=0.0, scenario=args.scenario, logger=logger)

    ############# If fixed full data (case study): #############
    
    # train_emb, train_labels, test_emb, test_labels = load_train_test_datasets(args.cell_line, folder="/home/tinglu/MLLM4SL/data/case_study_compare/datasets", logger=logger) # datasets_ori_with_sf500
    train_emb, train_labels, test_emb, test_labels = load_train_test_datasets(args.cell_line, folder="./data/dataset_example", logger=logger)
    _, train_loader = prepare_dataloader(train_emb, train_labels, batch_size=args.batch_size, train_split=0.0, scenario=args.scenario, logger=logger)
    _, val_loader = prepare_dataloader(test_emb, test_labels, batch_size=args.batch_size, train_split=0.0, scenario=args.scenario, logger=logger)

    ############# End if #############
    
    
    # Load BioBERT model
    logger.info(f"Loading BioBERT model from {args.biobert_path}.")
    biobert = AutoModel.from_pretrained(args.biobert_path).to(device)

    # Train model
    logger.info("Starting training.")
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        biobert=biobert,
        projector_lr=args.projector_lr,
        encoder_lr=args.encoder_lr,
        predictor_lr=args.predictor_lr,
        om_encoder_lr=args.om_encoder_lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        early_stop_patience=args.early_stop_patience,
        device=device,
        omics_startidx=7,
        gene_startidx=13,
        save_path=os.path.join(save_folder, "model", "model_ckpt.pt"),
        logger=logger
    )

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()




