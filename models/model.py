import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .local_moe import LocalMoEModule
from .global_moe import GlobalMoEModule


class EnhancedEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class EnhancedDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class TransformerAggregator(nn.Module):
    def __init__(self, num_views, input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerAggregator, self).__init__()
        self.num_views = num_views
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.view_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, dim_feedforward),
                nn.LayerNorm(dim_feedforward),
                nn.ReLU(inplace=True)
            ) for _ in range(num_views)
        ])
        self.view_embeddings = nn.Parameter(torch.randn(num_views, dim_feedforward))
        self.pos_embeddings = nn.Parameter(torch.randn(num_views + 1, dim_feedforward))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.cls_token = nn.Parameter(torch.randn(1, dim_feedforward))

        self.output_projection = nn.Sequential(
            nn.Linear(dim_feedforward, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )
        self.direct_aggregation = nn.Sequential(
            nn.Linear(input_dim * num_views, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, H_vs):
        batch_size = H_vs[0].size(0)
        projected_views = [self.view_projections[i](h_v) + self.view_embeddings[i].unsqueeze(0) for i, h_v in enumerate(H_vs)]
        view_sequence = torch.stack(projected_views, dim=1)
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        transformer_input = torch.cat([cls_tokens, view_sequence], dim=1) + self.pos_embeddings.unsqueeze(0)
        transformer_output = self.transformer_encoder(transformer_input)
        cls_output = transformer_output[:, 0, :]
        aggregated = self.output_projection(cls_output)
        direct_agg = self.direct_aggregation(torch.cat(H_vs, dim=1))
        return aggregated + direct_agg


class FinalEnhancedModel(nn.Module):
    def __init__(self, cfg, num_views: int, view_dims: List[int]):
        super(FinalEnhancedModel, self).__init__()
        self.num_views = num_views
        self.view_dims = view_dims
        self.feature_dim = cfg.model.feature_dim
        self.local_dim = cfg.model.local_output_dim
        self.num_clusters = cfg.model.num_clusters

        self.encoders = nn.ModuleList([EnhancedEncoder(self.view_dims[v], self.feature_dim) for v in range(self.num_views)])
        self.decoders = nn.ModuleList([EnhancedDecoder(self.view_dims[v], self.feature_dim) for v in range(self.num_views)])

        self.local_moe = LocalMoEModule(
            num_views=self.num_views,
            input_dim=self.feature_dim,
            num_experts=cfg.local_moe.num_experts,
            top_k=cfg.local_moe.top_k,
            expert_hidden_dims=cfg.local_moe.expert_hidden_dims,
            expert_output_dim=cfg.local_moe.expert_output_dim,
            router_hidden_dim=cfg.local_moe.router_hidden_dim,
            router_tau=cfg.local_moe.router_tau
        )

        self.aggregator = TransformerAggregator(
            num_views=self.num_views,
            input_dim=cfg.local_moe.expert_output_dim,
            output_dim=self.local_dim,
            nhead=cfg.transformer.nhead,
            num_encoder_layers=cfg.transformer.num_encoder_layers,
            dim_feedforward=cfg.transformer.dim_feedforward
        )

        self.global_moe = GlobalMoEModule(
            num_views=self.num_views,
            view_dim=self.feature_dim,
            num_experts=cfg.global_moe.num_experts,
            top_k=cfg.global_moe.top_k,
            expert_hidden_dims=cfg.global_moe.expert_hidden_dims,
            expert_output_dim=cfg.global_moe.expert_output_dim,
            router_tau=cfg.global_moe.router_tau,
            router_hidden_dim=cfg.global_moe.router_hidden_dim
        )

        self.cluster_head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(self.local_dim, self.num_clusters)
        )

        proj_dim = getattr(cfg.model, "proj_dim", 256)
        self.proj_glo = nn.Sequential(
            nn.Linear(cfg.global_moe.expert_output_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        self.proj_loc = nn.Sequential(
            nn.Linear(self.local_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x_views: List[torch.Tensor]) -> Tuple:
        Z_views = [self.encoders[v](x) for v, x in enumerate(x_views)]
        recon_views = [self.decoders[v](z) for v, z in enumerate(Z_views)]

        H_vs, local_scores, local_expert_outs = self.local_moe(Z_views)
        H_loc_list = H_vs
        H_loc_agg = self.aggregator(H_vs)
        H_glo, global_scores, global_expert_outs = self.global_moe(Z_views)

        logits = self.cluster_head(H_loc_agg)

        return (
            recon_views,
            H_loc_list,
            H_glo,
            logits,
            local_scores,
            local_expert_outs,
            global_scores,
            global_expert_outs
        )
