import torch
import torch.nn as nn
import os

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_model=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.last_best = 0.0
        self.best_metrics = {
            'accuracy': 0,
            'f1': 0,
            'roc_auc': 0,
            'aupr': 0,
            'bacc': 0,
            'recall':0,
        }
        self.save_model = save_model

    def __call__(self, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, logger=None):
        score = val_roc_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                pass
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metrics['accuracy'] = val_accuracy
            self.best_metrics['f1'] = val_f1
            self.best_metrics['roc_auc'] = val_roc_auc
            self.best_metrics['aupr'] = aupr
            self.best_metrics['bacc'] = val_bacc
            self.best_metrics['recall'] = val_recall
            self.save_checkpoint(score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, logger)
            self.counter = 0
        return self.best_metrics

    def save_checkpoint(self, score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, logger=None):
        '''Saves all models in a single file when AUPR improves.'''
        if self.verbose:
            print(f'Validation AUC ({self.last_best:.6f} --> {score:.6f}). '
                  f'Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}, BACC: {val_bacc:.4f}, '
                  f'AUC: {val_roc_auc:.4f}, AUPR: {aupr:.4f}, Recall: {val_recall:4f}')
            logger.info(f'Validation AUC ({self.last_best:.6f} --> {score:.6f}). '
                  f'Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}, BACC: {val_bacc:.4f}, '
                  f'AUC: {val_roc_auc:.4f}, AUPR: {aupr:.4f}, Recall: {val_recall:4f}')
        model_folder = os.path.dirname(model_path)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True) 

        if self.save_model:
            model_dict = {
                "projector": model[0].state_dict(),
                "bio_encoder": model[1].state_dict(),
                "fuse_model": model[2].state_dict(),
                "predictor": model[3].state_dict(),
                "encoder_om_cn": model[4].state_dict(),
                "encoder_om_dep": model[5].state_dict(),
                "encoder_om_eff": model[6].state_dict(),
                "encoder_om_exp": model[7].state_dict(),
                "encoder_om_met": model[8].state_dict(),
                "encoder_om_mut": model[9].state_dict()
            }
            torch.save(model_dict, model_path)  

        self.last_best = score
        


#########

class Projector(nn.Module):
    def __init__(self, input_dim=3072, output_dim=768, dropout_rate=0.1):
        super(Projector, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))

class Encoder_om(nn.Module):
    def __init__(self, input_dim=4079, output_dim=768, dropout_rate=0.1):
        super(Encoder_om, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))


class FuseModel(nn.Module): # [768, 768]
    def __init__(self, input_dim=768, hidden_dim=768):
        super(FuseModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, gene_emb, kg_emb):
        # Add batch dimension if missing
        gene_emb = gene_emb.unsqueeze(1) if gene_emb.dim() == 2 else gene_emb
        kg_emb = kg_emb.unsqueeze(1) if kg_emb.dim() == 2 else kg_emb

        # Fuse with attention
        fused, _  = self.attention(gene_emb, kg_emb, kg_emb)
        return self.fc(fused.squeeze(1)) 

# class FuseModel(nn.Module): # [768, 1536]
#     def __init__(self, input_dim=768, hidden_dim=768):
#         super(FuseModel, self).__init__()
#         self.fc = nn.Linear(input_dim * 2, hidden_dim)  # Concatenation

#     def forward(self, gene_emb, kg_emb):
#         combined_emb = torch.cat([gene_emb, kg_emb], dim=-1)  # [batch_size, 768 * 2]
#         fused_emb = self.fc(combined_emb)
#         return fused_emb


class Predictor(nn.Module):
    def __init__(self, input_dim=768 * 2, hidden_dim=512, output_dim=1, dropout_rate=0.1):
        super(Predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, fused_emb_1, fused_emb_2):
        combined = torch.cat([fused_emb_1, fused_emb_2], dim=-1)
        return self.mlp(combined)


class BioBERTEncoder(nn.Module):
    def __init__(self, biobert):
        super(BioBERTEncoder, self).__init__()
        self.encoder = biobert.encoder
        self.pooler = biobert.pooler
        
    def forward(self, token_emb, attention_mask=None):
        # Get encoder outputs
        encoder_output = self.encoder(token_emb, attention_mask=attention_mask)[0]  # Shape: [batch_size, seq_len, hidden_size]
        
        # Extract CLS token (first token) output
        cls_token_output = encoder_output[:, 0, :]  # Shape: [batch_size, hidden_size]
        
        return cls_token_output



