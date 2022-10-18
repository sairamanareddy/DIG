from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import GCANodeAttrMask, GCAEdgePerturbation, Sequential
from dig.sslgraph.method.contrastive.objectives import pGRACE_loss


class pGRACE(Contrastive):
    r"""
    Contrastive learning method proposed in the paper `Deep Graph Contrastive Representation 
    Learning <https://arxiv.org/abs/2006.04131>`_. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_grace.ipynb>`_ for
    an example of usage.
    
    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`GRACE`.
        
    Args:
        dim (int): The embedding dimension.
        prob_edge_1, prob_edge_2 (float): The probability factor for calculating edge-drop probability
        prob_feature_1, prob_feature_2 (float): The probability factor for calculating feature-masking probability
        tau (float, optional): The temperature parameter used for contrastive objective.
        p_tau (float, optional): The upper-bound probability for dropping edges or removing nodes.
        **kwargs (optional): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(self, dim: int, prob_edge_1: float, prob_edge_2: float, prob_feature_1: float, prob_feature_2: float,
                 tau: float = 0.1, p_tau: float = 0.7, **kwargs):
        view_fn_1 = Sequential([GCAEdgePerturbation(centrality_measure='degree', prob=prob_edge_1, threshold=p_tau),
                                GCANodeAttrMask(centrality_measure='degree', prob=prob_feature_1, threshold=p_tau)])
        view_fn_2 = Sequential([GCAEdgePerturbation(centrality_measure='degree', prob=prob_edge_2, threshold=p_tau),
                                GCANodeAttrMask(centrality_measure='degree', prob=prob_feature_2, threshold=p_tau)])
        views_fn = [view_fn_1, view_fn_2]
        
        super(pGRACE, self).__init__(objective=pGRACE_loss,
                                    views_fn=views_fn,
                                    graph_level=False,
                                    node_level=True,
                                    z_n_dim=dim,
                                    tau=tau,
                                    proj_n='linear',
                                    **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GRACE removes projection heads after pre-training
        for enc, proj in super().train(encoders, data_loader, 
                                       optimizer, epochs, per_epoch_out):
            yield enc
