# %% [markdown]
# ## Example of GRACE

# %% [markdown]
# ### Node-level representation learning on CORA

# %%
from dig.sslgraph.utils import Encoder
from dig.sslgraph.evaluation import NodeUnsupervised
from dig.sslgraph.dataset import get_node_dataset
from dig.sslgraph.method import pGRACE
from dig.sslgraph.utils.pgrace import generate_split

# %% [markdown]
# #### Loading dataset

# %%
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
import torch
path = '/tmp/dig-experiments/'
dataset = Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
device = torch.device('cuda:0')
data = dataset[0].to(device=device)

# %%
split = generate_split(data.num_nodes, train_ratio = 0.1, val_ratio = 0.1)

# %% [markdown]
# #### Define encoders and contrastive model
# 
# You can refer to [https://github.com/CRIPAC-DIG/GCA/blob/master/config.yaml](https://github.com/CRIPAC-DIG/GCA/blob/master/config.yaml) for detailed training configs.
# 
# ***Note***: Results in the GRACE paper uses different training-test splits to the public splits, due to which you may see different results in DIG and the original implementation of GRACE.

# %%
embed_dim = 128
encoder = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, 
                  n_layers=2, gnn='gcn', act='prelu', node_level=True, graph_level=False).to(device)
grace = pGRACE(dim=128, prob_edge_1 = 0.5, prob_edge_2 = 0.5, prob_feature_1 = 0.2, prob_feature_2 = 0.1).to(device)

# %%
evaluator = NodeUnsupervised(dataset, train_mask=split[0], test_mask = split[1], val_mask = split[2], device=0, log_interval=100)
evaluator.setup_train_config(p_lr=0.01, p_epoch=1500, p_weight_decay=1e-5, comp_embed_on='cuda:0')
evaluator.evaluate(learning_model=grace, encoder=encoder)


