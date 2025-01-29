from pykeen.datasets.base import PathDataset
from pykeen.models import ConvE
import random
import torch
import numpy as np
from pykeen.models import ConvE
from pykeen.training import LCWATrainingLoop
from torch.optim import Adam
import os
from pykeen.triples import TriplesFactory
import pandas as pd
from sklearn.model_selection import train_test_split


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

seed = 42  # You can choose any number here
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from pykeen.datasets import PrimeKG

dataset = PrimeKG(create_inverse_triples=True)
# Step 2: Configure the model
from pykeen.models import ConvE
model = ConvE(
    triples_factory=dataset.training,
    embedding_dim=256, # 768
    input_channels=1,
    output_channels=32,
    embedding_height=16, # 24
    embedding_width=16, # 32
    kernel_height=3,
    kernel_width=3,
    input_dropout=0.2,
    feature_map_dropout=0.2,
    output_dropout=0.3,
).to(device)
print("model")


#########################################################

# model.load_state_dict(torch.load('conv_primekg_model.pth'))
# model.eval() 


# # Get gene index and embedding
# gene_name = "PRKDC"  # Replace with your specific gene name
# gene_index = dataset.validation.entity_to_id[gene_name]  # Get the index of the gene
# print(gene_index) # 43190
# # gene_embedding = model.entity_representations[0].weight[gene_index].detach().cpu().numpy()
# gene_embedding = model.entity_representations[0](torch.tensor([gene_index], device=model.device)).detach().cpu().numpy()

# # Use the embedding
# print(gene_embedding)

#########################################################

# Step 3: Configure the loop
from torch.optim import Adam
optimizer = Adam(params=model.get_grad_params())
from pykeen.training import LCWATrainingLoop
training_loop = LCWATrainingLoop(model=model, optimizer=optimizer,triples_factory=dataset.training)
# Step 4: Train
losses = training_loop.train(num_epochs=10, batch_size=256, triples_factory=dataset.training, use_tqdm=True)
print("trained")


# Step 5: Evaluate the model
from pykeen.evaluation import RankBasedEvaluator
evaluator = RankBasedEvaluator()
metric_result = evaluator.evaluate(
    model=model,
    mapped_triples=dataset.validation.mapped_triples,
    additional_filter_triples=dataset.training.mapped_triples,
    batch_size=8192,
    use_tqdm=True
)

# Convert results to dictionary
results_dict = metric_result.to_dict()

# Print all available metrics
for key, value in results_dict.items():
    print(f"{key}: {value}")


## save

torch.save(model.state_dict(), 'conv_primekg_model_ep10_dim256.pth')

# Optionally, save the optimizer state
torch.save(optimizer.state_dict(), 'conv_primekg_optimizer_ep10_dim256.pth')

# Get embeddings directly from the model's embedding layers
if hasattr(model.entity_representations[0], 'weight'):
    entity_embeddings = model.entity_representations[0].weight.detach().cpu().numpy()
else:
    # If the embedding representation is not directly an embedding layer, handle accordingly
    entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()

if hasattr(model.relation_representations[0], 'weight'):
    relation_embeddings = model.relation_representations[0].weight.detach().cpu().numpy()
else:
    # If the embedding representation is not directly an embedding layer, handle accordingly
    relation_embeddings = model.relation_representations[0]().detach().cpu().numpy()

# Save embeddings to a file
np.save('conv_primekg_entity_emb256.npy', entity_embeddings)
np.save('conv_primekg_relation_emb256.npy', relation_embeddings)



