import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .model import Projector, FuseModel, Predictor, BioBERTEncoder, EarlyStopping, Encoder_om
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import auc, roc_auc_score, f1_score, balanced_accuracy_score,accuracy_score, precision_recall_curve, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pandas as pd
import pickle



def train_model(
    train_loader, val_loader, biobert, projector_lr, encoder_lr, predictor_lr, om_encoder_lr, lr_factor=0.5, lr_patience=3, 
    num_epochs=100, weight_decay=1e-4, dropout_rate=0.1,early_stop_patience=10, device="cuda", omics_startidx=7, gene_startidx=13, save_path="model_ckpt.pt", logger=None
):
    # Initialize models
    projector = Projector(dropout_rate=dropout_rate).to(device)
    fuse_model = FuseModel().to(device)
    predictor = Predictor(dropout_rate=dropout_rate).to(device)
    bio_encoder = BioBERTEncoder(biobert).to(device)
    encoder_om_cn = Encoder_om(input_dim=4078,dropout_rate=dropout_rate).to(device)
    encoder_om_dep = Encoder_om(input_dim=3456,dropout_rate=dropout_rate).to(device)
    encoder_om_eff = Encoder_om(input_dim=3456,dropout_rate=dropout_rate).to(device)
    encoder_om_exp = Encoder_om(input_dim=4078,dropout_rate=dropout_rate).to(device)
    encoder_om_met = Encoder_om(input_dim=2279,dropout_rate=dropout_rate).to(device)
    encoder_om_mut = Encoder_om(input_dim=3937,dropout_rate=dropout_rate).to(device)
    
    # Define loss function
    criterion = nn.BCELoss()

    # Optimizer with different learning rates
    optimizer = Adam(
        [
            {"params": projector.parameters(), "lr": projector_lr, "weight_decay": weight_decay},
            {"params": fuse_model.parameters(), "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": predictor.parameters(), "lr": predictor_lr, "weight_decay": weight_decay},
            {"params": bio_encoder.parameters(), "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_om_cn.parameters(), "lr": om_encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_om_dep.parameters(), "lr": om_encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_om_eff.parameters(), "lr": om_encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_om_exp.parameters(), "lr": om_encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_om_met.parameters(), "lr": om_encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_om_mut.parameters(), "lr": om_encoder_lr, "weight_decay": weight_decay},
        ]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True)
    
    
    # Early stopping mechanism
    early_stopping = EarlyStopping(
        patience=early_stop_patience,
        verbose=True,
        save_model=True
    )

    for epoch in range(num_epochs):
        projector.train()
        fuse_model.train()
        predictor.train()
        bio_encoder.train()
        encoder_om_cn.train()
        encoder_om_dep.train()
        encoder_om_eff.train()
        encoder_om_exp.train()
        encoder_om_met.train()
        encoder_om_mut.train()

        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            # Unpack the batch
            # (cell_line_emb, token_emb_1, att_mask_1, genept_emb_1, kge_emb_1, token_emb_2, att_mask_2, genept_emb_2, kge_emb_2), labels = batch
            (cell_line_emb, om_cn, om_dep, om_eff, om_exp, om_met, om_mut, token_emb_1, att_mask_1, genept_emb_1, kge_emb_1,
             token_emb_2, att_mask_2, genept_emb_2, kge_emb_2), labels = batch

            cell_line_emb = cell_line_emb.to(device)
            om_cn = om_cn.to(device)
            om_dep = om_dep.to(device)
            om_eff = om_eff.to(device)
            om_exp = om_exp.to(device)
            om_met = om_met.to(device)
            om_mut = om_mut.to(device)
            token_emb_1 = token_emb_1.to(device)
            att_mask_1 = att_mask_1.to(device)
            genept_emb_1 = genept_emb_1.to(device)
            kge_emb_1 = kge_emb_1.to(device)
            token_emb_2 = token_emb_2.to(device)
            att_mask_2 = att_mask_2.to(device)
            genept_emb_2 = genept_emb_2.to(device)
            kge_emb_2 = kge_emb_2.to(device)
            labels = labels.float().to(device)
            

            ## Step 1: Project GenePT embeddings & Omics data to 768-dim embeddings
            projected_emb_1 = projector(genept_emb_1)
            projected_emb_2 = projector(genept_emb_2)
            # print("projected_emb_1", projected_emb_1.shape) # [64, 768]
            # print("token_emb_1", token_emb_1.shape) # [64, 512, 768]
            # print("check if gene start idx:", torch.all(token_emb_1[:, 7, :] == 0).item())
            om_cn_emb = encoder_om_cn(om_cn)
            om_dep_emb = encoder_om_dep(om_dep)
            om_eff_emb = encoder_om_eff(om_eff)
            om_exp_emb = encoder_om_exp(om_exp)
            om_met_emb = encoder_om_met(om_met)
            om_mut_emb = encoder_om_mut(om_mut)
            
            ## Step 2: Replace token embeddings
            # token_emb_1[:, 7, :] = projected_emb_1  # Replace gene_startidx for gene 12
            token_emb_1[:, gene_startidx, :] = projected_emb_1  # add omics emb (6*768)
            token_emb_2[:, gene_startidx, :] = projected_emb_2  # add omics emb (6*768)
            
            token_emb_1[:, omics_startidx + 0, :] = om_cn_emb  # add omics channel 1 embedding (768)
            token_emb_1[:, omics_startidx + 1, :] = om_dep_emb  # add omics channel 2 embedding (768)
            token_emb_1[:, omics_startidx + 2, :] = om_eff_emb  # add omics channel 3 embedding (768)
            token_emb_1[:, omics_startidx + 3, :] = om_exp_emb  # add omics channel 4 embedding (768)
            token_emb_1[:, omics_startidx + 4, :] = om_met_emb  # add omics channel 5 embedding (768)
            token_emb_1[:, omics_startidx + 5, :] = om_mut_emb  # add omics channel 6 embedding (768)

            token_emb_2[:, omics_startidx + 0, :] = om_cn_emb  # add omics channel 1 embedding (768)
            token_emb_2[:, omics_startidx + 1, :] = om_dep_emb  # add omics channel 2 embedding (768)
            token_emb_2[:, omics_startidx + 2, :] = om_eff_emb  # add omics channel 3 embedding (768)
            token_emb_2[:, omics_startidx + 3, :] = om_exp_emb  # add omics channel 4 embedding (768)
            token_emb_2[:, omics_startidx + 4, :] = om_met_emb  # add omics channel 5 embedding (768)
            token_emb_2[:, omics_startidx + 5, :] = om_mut_emb  # add omics channel 6 embedding (768)
            
            ## no prompt: :13; no cell line emb: :7; no 6 omics emb: 7:13
            # token_emb_1[:, :13, :] = 0  
            # token_emb_2[:, :13, :] = 0 

            ## Step 3: Get gene representation from BioBERT
            # print("check before bio encoder:", token_emb_1.shape, att_mask_1.shape) # [16, 512, 768] [16, 512]
            gene_repr_1 = bio_encoder(token_emb_1) ####
            gene_repr_2 = bio_encoder(token_emb_2) ####

            ## Step 4: Fuse gene representation with KG embedding
            # print("check for fusion:", gene_repr_1.shape, kge_emb_1.shape) # [16, 768] [16, 768]
            
            fused_emb_1 = fuse_model(gene_repr_1, kge_emb_1) ####
            fused_emb_2 = fuse_model(gene_repr_2, kge_emb_2) ####

            ## Step 5: Predict binary output
            # print("check before predictor:", fused_emb_1.shape) # [16, 768]
            predictions = predictor(fused_emb_1, fused_emb_2).squeeze() ####

            # Compute loss
            predictions = predictions.view(-1)  # Flatten to ensure it's (batch_size,)
            loss = criterion(predictions, labels.view(-1)) 
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation Step
        val_loss, val_metrics = validate_model(
            val_loader, projector, bio_encoder, fuse_model, predictor, 
            encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut,
            criterion, omics_startidx, gene_startidx, device
        )
        
        scheduler.step(val_loss)
        
        train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Early stopping
        early_stopping(
            val_metrics['accuracy'], 
            val_metrics['f1'], 
            val_metrics['roc_auc'], 
            val_metrics['aupr'], 
            val_metrics['bacc'], 
            val_metrics['recall'],
            model=[projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut], 
            model_path=save_path,
            logger=logger
        )
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            save_final_results(early_stopping.best_metrics, save_path)
            break
    if not early_stopping.early_stop:
        save_final_results(early_stopping.best_metrics, save_path)


def validate_model(
    val_loader, projector, bio_encoder, fuse_model, predictor, 
    encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut,
    criterion, omics_startidx=7, gene_startidx=13, device=None
):
    """
    Validate the model on the validation dataset.

    Args:
        val_loader (DataLoader): Validation data loader.
        projector (nn.Module): Projector model.
        bio_encoder (nn.Module): BioBERT encoder model.
        fuse_model (nn.Module): Fusion model.
        predictor (nn.Module): Predictor model.
        criterion (nn.Module): Loss function.
        device (str): Device for computation.

    Returns:
        val_loss (float): Validation loss.
        val_metrics (dict): Dictionary containing evaluation metrics (accuracy, F1, ROC-AUC, AUPR).
    """
    projector.eval()
    fuse_model.eval()
    predictor.eval()
    bio_encoder.eval()
    encoder_om_cn.eval()
    encoder_om_dep.eval()
    encoder_om_eff.eval()
    encoder_om_exp.eval()
    encoder_om_met.eval()
    encoder_om_mut.eval()
    
    val_loss = 0
    all_labels = []
    all_predictions = []
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            # Unpack the batch
            # (cell_line_emb, token_emb_1, att_mask_1, genept_emb_1, kge_emb_1, token_emb_2, att_mask_2, genept_emb_2, kge_emb_2), labels = batch
            (cell_line_emb, om_cn, om_dep, om_eff, om_exp, om_met, om_mut, token_emb_1, att_mask_1, genept_emb_1, kge_emb_1,
             token_emb_2, att_mask_2, genept_emb_2, kge_emb_2), labels = batch
            
            cell_line_emb = cell_line_emb.to(device)
            om_cn = om_cn.to(device)
            om_dep = om_dep.to(device)
            om_eff = om_eff.to(device)
            om_exp = om_exp.to(device)
            om_met = om_met.to(device)
            om_mut = om_mut.to(device)
            token_emb_1 = token_emb_1.to(device)
            att_mask_1 = att_mask_1.to(device)
            genept_emb_1 = genept_emb_1.to(device)
            kge_emb_1 = kge_emb_1.to(device)
            token_emb_2 = token_emb_2.to(device)
            att_mask_2 = att_mask_2.to(device)
            genept_emb_2 = genept_emb_2.to(device)
            kge_emb_2 = kge_emb_2.to(device)
            labels = labels.float().to(device)

            # Project embeddings
            projected_emb_1 = projector(genept_emb_1)
            projected_emb_2 = projector(genept_emb_2)
            om_cn_emb = encoder_om_cn(om_cn)
            om_dep_emb = encoder_om_dep(om_dep)
            om_eff_emb = encoder_om_eff(om_eff)
            om_exp_emb = encoder_om_exp(om_exp)
            om_met_emb = encoder_om_met(om_met)
            om_mut_emb = encoder_om_mut(om_mut)

            # Replace token embeddings
            token_emb_1[:, gene_startidx, :] = projected_emb_1
            token_emb_2[:, gene_startidx, :] = projected_emb_2
            token_emb_1[:, omics_startidx + 0, :] = om_cn_emb  # add omics channel 1 embedding (768)
            token_emb_1[:, omics_startidx + 1, :] = om_dep_emb  # add omics channel 2 embedding (768)
            token_emb_1[:, omics_startidx + 2, :] = om_eff_emb  # add omics channel 3 embedding (768)
            token_emb_1[:, omics_startidx + 3, :] = om_exp_emb  # add omics channel 4 embedding (768)
            token_emb_1[:, omics_startidx + 4, :] = om_met_emb  # add omics channel 5 embedding (768)
            token_emb_1[:, omics_startidx + 5, :] = om_mut_emb  # add omics channel 6 embedding (768)

            token_emb_2[:, omics_startidx + 0, :] = om_cn_emb  # add omics channel 1 embedding (768)
            token_emb_2[:, omics_startidx + 1, :] = om_dep_emb  # add omics channel 2 embedding (768)
            token_emb_2[:, omics_startidx + 2, :] = om_eff_emb  # add omics channel 3 embedding (768)
            token_emb_2[:, omics_startidx + 3, :] = om_exp_emb  # add omics channel 4 embedding (768)
            token_emb_2[:, omics_startidx + 4, :] = om_met_emb  # add omics channel 5 embedding (768)
            token_emb_2[:, omics_startidx + 5, :] = om_mut_emb 

            ## no prompt: :13; no cell line emb: :7; no 6 omics emb: 7:13
            # token_emb_1[:, :13, :] = 0  
            # token_emb_2[:, :13, :] = 0 
            
            # Gene representations
            gene_repr_1 = bio_encoder(token_emb_1) ####
            gene_repr_2 = bio_encoder(token_emb_2) ####

            
            # Fuse embeddings
            fused_emb_1 = fuse_model(gene_repr_1, kge_emb_1) ####
            fused_emb_2 = fuse_model(gene_repr_2, kge_emb_2) ####

            # Predictions
            predictions = predictor(fused_emb_1, fused_emb_2).squeeze() ####

            # Compute loss
            predictions = predictions.view(-1)  # Flatten to ensure it's (batch_size,)
            loss = criterion(predictions, labels.view(-1)) 
            val_loss += loss.item()

            # Collect predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            total += labels.size(0)

    # Convert predictions to binary for threshold 0.5
    binary_predictions = (torch.tensor(all_predictions) > 0.5).float().numpy()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_predictions)
    
    recall_scr = recall_score(all_labels, binary_predictions)
    
    # Calculate F1 score across different thresholds and choose the best
    best_f1 = 0
    for threshold in [x * 0.05 for x in range(20)]:
        threshold_predictions = (torch.tensor(all_predictions) > threshold).float().numpy()
        f1 = f1_score(all_labels, threshold_predictions)
        best_f1 = max(best_f1, f1)
    
    # Balanced accuracy
    bacc = balanced_accuracy_score(all_labels, binary_predictions)

    roc_auc = roc_auc_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    aupr = auc(recall, precision)

    val_metrics = {
        "accuracy": accuracy,
        "f1": best_f1,  # Return the best F1 score
        "roc_auc": roc_auc,
        "aupr": aupr,
        "bacc": bacc,  # Include balanced accuracy
        "recall": recall_scr
    }

    return val_loss / len(val_loader), val_metrics

def eval_model(
    val_loader, projector, bio_encoder, fuse_model, predictor, 
    encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut,
    criterion, omics_startidx=7, gene_startidx=13, device=None
):
    """
    Validate the model on the validation dataset.

    Args:
        val_loader (DataLoader): Validation data loader.
        projector (nn.Module): Projector model.
        bio_encoder (nn.Module): BioBERT encoder model.
        fuse_model (nn.Module): Fusion model.
        predictor (nn.Module): Predictor model.
        criterion (nn.Module): Loss function.
        device (str): Device for computation.

    Returns:
        val_loss (float): Validation loss.
        val_metrics (dict): Dictionary containing evaluation metrics (accuracy, F1, ROC-AUC, AUPR).
    """
    projector.eval()
    fuse_model.eval()
    predictor.eval()
    bio_encoder.eval()
    encoder_om_cn.eval()
    encoder_om_dep.eval()
    encoder_om_eff.eval()
    encoder_om_exp.eval()
    encoder_om_met.eval()
    encoder_om_mut.eval()
    
    val_loss = 0
    all_labels = []
    all_predictions = []
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            # Unpack the batch
            # (cell_line_emb, token_emb_1, att_mask_1, genept_emb_1, kge_emb_1, token_emb_2, att_mask_2, genept_emb_2, kge_emb_2), labels = batch
            (cell_line_emb, om_cn, om_dep, om_eff, om_exp, om_met, om_mut, token_emb_1, att_mask_1, genept_emb_1, kge_emb_1,
             token_emb_2, att_mask_2, genept_emb_2, kge_emb_2), labels = batch
            
            cell_line_emb = cell_line_emb.to(device)
            om_cn = om_cn.to(device)
            om_dep = om_dep.to(device)
            om_eff = om_eff.to(device)
            om_exp = om_exp.to(device)
            om_met = om_met.to(device)
            om_mut = om_mut.to(device)
            token_emb_1 = token_emb_1.to(device)
            att_mask_1 = att_mask_1.to(device)
            genept_emb_1 = genept_emb_1.to(device)
            kge_emb_1 = kge_emb_1.to(device)
            token_emb_2 = token_emb_2.to(device)
            att_mask_2 = att_mask_2.to(device)
            genept_emb_2 = genept_emb_2.to(device)
            kge_emb_2 = kge_emb_2.to(device)
            labels = labels.float().to(device)

            # Project embeddings
            projected_emb_1 = projector(genept_emb_1)
            projected_emb_2 = projector(genept_emb_2)
            om_cn_emb = encoder_om_cn(om_cn)
            om_dep_emb = encoder_om_dep(om_dep)
            om_eff_emb = encoder_om_eff(om_eff)
            om_exp_emb = encoder_om_exp(om_exp)
            om_met_emb = encoder_om_met(om_met)
            om_mut_emb = encoder_om_mut(om_mut)

            # Replace token embeddings
            token_emb_1[:, gene_startidx, :] = projected_emb_1
            token_emb_2[:, gene_startidx, :] = projected_emb_2
            token_emb_1[:, omics_startidx + 0, :] = om_cn_emb  # add omics channel 1 embedding (768)
            token_emb_1[:, omics_startidx + 1, :] = om_dep_emb  # add omics channel 2 embedding (768)
            token_emb_1[:, omics_startidx + 2, :] = om_eff_emb  # add omics channel 3 embedding (768)
            token_emb_1[:, omics_startidx + 3, :] = om_exp_emb  # add omics channel 4 embedding (768)
            token_emb_1[:, omics_startidx + 4, :] = om_met_emb  # add omics channel 5 embedding (768)
            token_emb_1[:, omics_startidx + 5, :] = om_mut_emb  # add omics channel 6 embedding (768)

            token_emb_2[:, omics_startidx + 0, :] = om_cn_emb  # add omics channel 1 embedding (768)
            token_emb_2[:, omics_startidx + 1, :] = om_dep_emb  # add omics channel 2 embedding (768)
            token_emb_2[:, omics_startidx + 2, :] = om_eff_emb  # add omics channel 3 embedding (768)
            token_emb_2[:, omics_startidx + 3, :] = om_exp_emb  # add omics channel 4 embedding (768)
            token_emb_2[:, omics_startidx + 4, :] = om_met_emb  # add omics channel 5 embedding (768)
            token_emb_2[:, omics_startidx + 5, :] = om_mut_emb 

            # no prompt: :13; no cell line emb: :7; no 6 omics emb: 7:13
            # token_emb_1[:, :7, :] = 0  
            # token_emb_2[:, :7, :] = 0 
            
            # Gene representations
            gene_repr_1 = bio_encoder(token_emb_1)
            gene_repr_2 = bio_encoder(token_emb_2)
            
            # Fuse embeddings
            fused_emb_1 = fuse_model(gene_repr_1, kge_emb_1)
            fused_emb_2 = fuse_model(gene_repr_2, kge_emb_2)

            # Predictions
            predictions = predictor(fused_emb_1, fused_emb_2).squeeze()

            # Compute loss
            predictions = predictions.view(-1)  # Flatten to ensure it's (batch_size,)
            loss = criterion(predictions, labels.view(-1)) 
            val_loss += loss.item()

            # Collect predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            total += labels.size(0)

    # Convert predictions to binary for threshold 0.5
    binary_predictions = (torch.tensor(all_predictions) > 0.5).float().numpy()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_predictions)
    
    recall_scr = recall_score(all_labels, binary_predictions)
    
    # Calculate F1 score across different thresholds and choose the best
    best_f1 = 0
    for threshold in [x * 0.05 for x in range(20)]:
        threshold_predictions = (torch.tensor(all_predictions) > threshold).float().numpy()
        f1 = f1_score(all_labels, threshold_predictions)
        best_f1 = max(best_f1, f1)
    
    # Balanced accuracy
    bacc = balanced_accuracy_score(all_labels, binary_predictions)

    roc_auc = roc_auc_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    aupr = auc(recall, precision)

    val_metrics = {
        "accuracy": accuracy,
        "f1": best_f1,  # Return the best F1 score
        "roc_auc": roc_auc,
        "aupr": aupr,
        "bacc": bacc,  # Include balanced accuracy
        "recall": recall_scr
    }

    return val_loss / len(val_loader), val_metrics
    
def validate_model_with_reference(
    val_loader, projector, bio_encoder, fuse_model, predictor, 
    encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut,
    criterion, omics_startidx=7, gene_startidx=13, device="cuda", output_reference_labels=True
):
    projector.eval()
    fuse_model.eval()
    predictor.eval()
    bio_encoder.eval()
    encoder_om_cn.eval()
    encoder_om_dep.eval()
    encoder_om_eff.eval()
    encoder_om_exp.eval()
    encoder_om_met.eval()
    encoder_om_mut.eval()
    
    val_loss = 0
    all_labels = []
    all_predictions = []
    inference_results = [] 
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            # Unpack the batch
            # (cell_line_emb, token_emb_1, att_mask_1, genept_emb_1, kge_emb_1, token_emb_2, att_mask_2, genept_emb_2, kge_emb_2), labels = batch
            (cell_line_emb, om_cn, om_dep, om_eff, om_exp, om_met, om_mut, token_emb_1, att_mask_1, genept_emb_1, kge_emb_1,
             token_emb_2, att_mask_2, genept_emb_2, kge_emb_2), labels = batch
            
            cell_line_emb = cell_line_emb.to(device)
            om_cn = om_cn.to(device)
            om_dep = om_dep.to(device)
            om_eff = om_eff.to(device)
            om_exp = om_exp.to(device)
            om_met = om_met.to(device)
            om_mut = om_mut.to(device)
            token_emb_1 = token_emb_1.to(device)
            att_mask_1 = att_mask_1.to(device)
            genept_emb_1 = genept_emb_1.to(device)
            kge_emb_1 = kge_emb_1.to(device)
            token_emb_2 = token_emb_2.to(device)
            att_mask_2 = att_mask_2.to(device)
            genept_emb_2 = genept_emb_2.to(device)
            kge_emb_2 = kge_emb_2.to(device)
            labels = labels.float().to(device)

            # Project embeddings
            projected_emb_1 = projector(genept_emb_1)
            projected_emb_2 = projector(genept_emb_2)
            om_cn_emb = encoder_om_cn(om_cn)
            om_dep_emb = encoder_om_dep(om_dep)
            om_eff_emb = encoder_om_eff(om_eff)
            om_exp_emb = encoder_om_exp(om_exp)
            om_met_emb = encoder_om_met(om_met)
            om_mut_emb = encoder_om_mut(om_mut)

            # Replace token embeddings
            token_emb_1[:, gene_startidx, :] = projected_emb_1
            token_emb_2[:, gene_startidx, :] = projected_emb_2
            token_emb_1[:, omics_startidx + 0, :] = om_cn_emb  # add omics channel 1 embedding (768)
            token_emb_1[:, omics_startidx + 1, :] = om_dep_emb  # add omics channel 2 embedding (768)
            token_emb_1[:, omics_startidx + 2, :] = om_eff_emb  # add omics channel 3 embedding (768)
            token_emb_1[:, omics_startidx + 3, :] = om_exp_emb  # add omics channel 4 embedding (768)
            token_emb_1[:, omics_startidx + 4, :] = om_met_emb  # add omics channel 5 embedding (768)
            token_emb_1[:, omics_startidx + 5, :] = om_mut_emb  # add omics channel 6 embedding (768)

            token_emb_2[:, omics_startidx + 0, :] = om_cn_emb  # add omics channel 1 embedding (768)
            token_emb_2[:, omics_startidx + 1, :] = om_dep_emb  # add omics channel 2 embedding (768)
            token_emb_2[:, omics_startidx + 2, :] = om_eff_emb  # add omics channel 3 embedding (768)
            token_emb_2[:, omics_startidx + 3, :] = om_exp_emb  # add omics channel 4 embedding (768)
            token_emb_2[:, omics_startidx + 4, :] = om_met_emb  # add omics channel 5 embedding (768)
            token_emb_2[:, omics_startidx + 5, :] = om_mut_emb 
            
            
            ## no prompt: :13; no cell line emb: :7; no 6 omics emb: 7:13
            # token_emb_1[:, :13, :] = 0 
            # token_emb_2[:, :13, :] = 0 
            
            # Gene representations
            gene_repr_1 = bio_encoder(token_emb_1)
            gene_repr_2 = bio_encoder(token_emb_2)

            # Fuse embeddings
            fused_emb_1  = fuse_model(gene_repr_1, kge_emb_1)
            fused_emb_2  = fuse_model(gene_repr_2, kge_emb_2)
            
            # print(f"Attention Weights for Gene Emb 1: {attn_weights_1.squeeze().detach().cpu().numpy()}")
            # print(f"Attention Weights for Gene Emb 2: {attn_weights_2.squeeze().detach().cpu().numpy()}")
            
            # Predictions
            predictions = predictor(fused_emb_1, fused_emb_2).squeeze()

            # Compute loss
            predictions = predictions.view(-1)  # Flatten to ensure it's (batch_size,)
            loss = criterion(predictions, labels.view(-1)) 
            val_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            
            if output_reference_labels:
                for i in range(len(predictions)):
                    inference_results.append({
                        "gene_pair_1_f5": (genept_emb_1[i].cpu().numpy())[:].tolist(),
                        "gene_pair_2_f5": (genept_emb_2[i].cpu().numpy())[:].tolist(),
                        "biobert_gene_emb_1": gene_repr_1[i].cpu().numpy().tolist(),
                        "biobert_gene_emb_2": gene_repr_2[i].cpu().numpy().tolist(),
                        # "kg_gene_emb_1": kge_emb_1[i].cpu().numpy().tolist(),
                        # "kg_gene_emb_2": kge_emb_2[i].cpu().numpy().tolist(),
                        "fused_emb_1": fused_emb_1[i].cpu().numpy().tolist(),
                        "fused_emb_2": fused_emb_2[i].cpu().numpy().tolist(),
                        "predicted_label": predictions[i].item(),
                        "true_label": labels[i].item()
                    })
                    
            total += labels.size(0)

    # Convert predictions to binary for threshold 0.5
    binary_predictions = (torch.tensor(all_predictions) > 0.5).float().numpy()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_predictions)
    
    recall_scr = recall_score(all_labels, binary_predictions)
    
    # Calculate F1 score across different thresholds and choose the best
    best_f1 = 0
    for threshold in [x * 0.05 for x in range(20)]:
        threshold_predictions = (torch.tensor(all_predictions) > threshold).float().numpy()
        f1 = f1_score(all_labels, threshold_predictions)
        best_f1 = max(best_f1, f1)
    
    # Balanced accuracy
    bacc = balanced_accuracy_score(all_labels, binary_predictions)

    roc_auc = roc_auc_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    aupr = auc(recall, precision)

    val_metrics = {
        "accuracy": accuracy,
        "f1": best_f1,  # Return the best F1 score
        "roc_auc": roc_auc,
        "aupr": aupr,
        "bacc": bacc,  # Include balanced accuracy
        "recall": recall_scr
    }


    return val_loss / len(val_loader), val_metrics, inference_results 


def save_final_results(val_metrics, save_path, cell_line=None):
    # Prepare the data for this run as a dictionary
    data = {
        'Val Accuracy': [val_metrics['accuracy']],
        'Val F1': [val_metrics['f1']],
        'Val ROC-AUC': [val_metrics['roc_auc']],
        'Val AUPR': [val_metrics['aupr']],
        'Val BACC': [val_metrics['bacc']],
        'Val Recall': [val_metrics['recall']],
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    result_file = os.path.join(os.path.dirname(os.path.dirname(save_path)), 'final_results.csv')
    
    # Write to CSV (append mode, with header only if the file doesn't exist)
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, mode='w', header=True, index=False)
        
        
def load_models(model_path, projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut, device):
    model_dict = torch.load(model_path, map_location=device) 

    # Load the state dict into the models
    projector.load_state_dict(model_dict["projector"])
    bio_encoder.load_state_dict(model_dict["bio_encoder"])
    fuse_model.load_state_dict(model_dict["fuse_model"])
    predictor.load_state_dict(model_dict["predictor"])
    encoder_om_cn.load_state_dict(model_dict["encoder_om_cn"])
    encoder_om_dep.load_state_dict(model_dict["encoder_om_dep"])
    encoder_om_eff.load_state_dict(model_dict["encoder_om_eff"])
    encoder_om_exp.load_state_dict(model_dict["encoder_om_exp"])
    encoder_om_met.load_state_dict(model_dict["encoder_om_met"])
    encoder_om_mut.load_state_dict(model_dict["encoder_om_mut"])

    return projector, bio_encoder, fuse_model, predictor, encoder_om_cn, encoder_om_dep, encoder_om_eff, encoder_om_exp, encoder_om_met, encoder_om_mut




