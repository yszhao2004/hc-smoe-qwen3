import gc
import os
import json
import pickle
import time
import sys
import numpy as np
from copy import deepcopy
from pickle import dump
from types import MethodType
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from .utils import generate_random_group_labels
from hcsmoe.utils.constants import FP32_EPS
from hcsmoe.models.mixtral.utils import merged_moe_forward, MoEWrapper
from hcsmoe.merging.clustering import compute_silhouette_score, group_experts_by_clustering
from hcsmoe.merging.overlap import compute_kl_divergence, get_prob_distributions, compute_wasserstein_distance

import sys
sys.path.append('/home/zhaozhr/moe_1')  

from modelings.configuration_deepseek import DeepseekConfig
from modelings.modeling_deepseek import DeepseekForCausalLM

from modelings.modeling_deepseek import DeepseekMoE, DeepseekMLP

SIMILARITY_MAPPING_FUNCTION = {
    "cosine": lambda x, y: (F.cosine_similarity(x, y, dim=-1, eps=FP32_EPS) + 1).item() / 2,
    "mse": lambda x, y: 1 / (1 + 0.1 * torch.log(F.mse_loss(x, y, reduction="sum"))).item(),
}

LEGAL_SIMILARITY_BASES = ["weight", "feature", "feature.abs", "weight-feature", "gradient", "weight-gradient",
                          "router-logits", "router-weight", "router-weight-feature", "mse", "random", "no",
                          "feature-correlation.lsa", "feature-correlation.max", "expert-output", "weight+expert-output",
                          "router-logits+weight", "router-logits+expert-output", "router-logits+weight+expert-output"]


class ExpertsGrouperForDeepseek(object):
    def __init__(
            self,
            config: DeepseekConfig,
            start_layer: int = 0,
            similarity_fn: str = "cosine",
            similarity_base: str = "router-logits",
            group_limit: int = 4,
            data_limit: int = 50000,
            random_start_center: bool = False,
            cluster: str = "kmeans",
            linkage: str = "ward",
            hierarchical_stopping_metric: str = "silhouette",
            overlap_metric: str = "cosine",
            dynamic_group: bool = False,
    ):
        if similarity_fn not in SIMILARITY_MAPPING_FUNCTION:
            raise ValueError(
                f"[HC-SMoE]similarity_fn should be one of {SIMILARITY_MAPPING_FUNCTION.keys()}, got {similarity_fn} instead."
            )
        if similarity_base not in LEGAL_SIMILARITY_BASES:
            raise ValueError(
                f"[HC-SMoE] similarity_base should be one of {LEGAL_SIMILARITY_BASES}, got {similarity_base} instead.")

        self.num_experts = config.n_routed_experts
        self.d_model = config.hidden_size
        self.d_ff = config.moe_intermediate_size
        self.topk = config.num_experts_per_tok
        self.sparse_layer_indices = list(range(start_layer, config.num_hidden_layers))
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.random_start_center = random_start_center
        self.cluster = cluster
        self.linkage = linkage
        self.hierarchical_stopping_metric = hierarchical_stopping_metric
        self.overlap_metric = overlap_metric
        self.dynamic_group = dynamic_group

        self.similarity_fn = SIMILARITY_MAPPING_FUNCTION[similarity_fn]
        self.similarity_base = similarity_base
        self._group_state_dict = None
        self._similarity_state_dict = None
        self._usage_frequency_state_dict = None
        self._init_center_state_dict = None
        self.moe_scores = None
        self.reset_all()

    def reset_all(self):
        if self.similarity_base == "mse":
            self.similarity_fn = SIMILARITY_MAPPING_FUNCTION["mse"]
            print("[HC-SMoE]Set similarity_fn to mse for mse similarity_base.")
        self._group_state_dict = dict()
        self._similarity_state_dict = dict()
        self._usage_frequency_state_dict = dict()
        self._init_center_state_dict = dict()
        self.moe_scores = torch.zeros(len(self.sparse_layer_indices), self.num_experts, self.d_ff)
        # Similarity range: [0, 2]
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            self._group_state_dict[ffn_name] = torch.arange(self.num_experts, device="cpu")
            self._similarity_state_dict[ffn_name] = torch.zeros(
                (self.num_experts, self.num_experts), device="cpu") + torch.eye(self.num_experts, device="cpu")
            self._usage_frequency_state_dict[ffn_name] = torch.zeros(self.num_experts, device="cpu")

    def similarity_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._similarity_state_dict)

    def group_state_dict(self) -> Dict[str, torch.LongTensor]:
        return deepcopy(self._group_state_dict)

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._usage_frequency_state_dict)

    def save_similarity(self, mlp_name: str, i: int, j: int, similarity: float):
        self._similarity_state_dict[mlp_name][i, j] = similarity
        self._similarity_state_dict[mlp_name][j, i] = similarity

    def get_similarity(self, mlp_name: str, i: int, j: int) -> float:
        return self._similarity_state_dict[mlp_name][i, j].item()

    def get_similarity_matrix(self, mlp_name: str) -> torch.Tensor:
        return deepcopy(self._similarity_state_dict[mlp_name])

    def save_group_state_dict(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._group_state_dict, os.path.join(save_dir, "group_state_dict.pt"))

    def load_group_state_dict(self, load_dir: str):
        self._group_state_dict = torch.load(os.path.join(load_dir, "group_state_dict.pt"))
    
    def load_init_center_state_dict(self, load_path: str):
        init_centers = pickle.load(open(load_path, "rb"))
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            self._init_center_state_dict[ffn_name] = torch.tensor(init_centers[layer_idx])

    def _get_moe_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = input[0].detach().reshape(
                -1, self.d_model)  # of shape (batch_size*sequence_length, hidden_size)
        return hook

    def _assign_num_groups_per_layer(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, int]:
        num_grouping_layers = len(merging_layers)
        total_num_groups = num_average_groups * num_grouping_layers + self.num_experts * (
                len(self.sparse_layer_indices) - num_grouping_layers
        )
        all_usage_frequency = []
        usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            all_usage_frequency.append(usage_frequency_dict[ffn_name])

        all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
        sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
        num_groups_per_layer = dict()

        # Note: When threshold is 0.0, the actual number of groups is smaller than total_num_groups.
        if num_average_groups == self.num_experts:
            total_num_groups = total_num_groups - 1
        frequency_threshold = sorted_usage_frequency[total_num_groups]
        print(f"[HC-SMoE] Frequency threshold: {frequency_threshold}")

        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups_per_layer[ffn_name] = torch.sum(
                (usage_frequency_dict[ffn_name] >= frequency_threshold).long()
            ).item()

        return num_groups_per_layer

    def group_experts_randomly(
        self,
        num_groups: int,
    ):
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Randomly merging experts into {num_groups} clusters"):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            group_labels = generate_random_group_labels(self.num_experts, num_groups)
            self._group_state_dict[ffn_name] = group_labels

    #NOTE: Compute Sihouette Score
    def compute_sihouette_score(self, model, dataloader):
        if self.similarity_base == "expert-output":
            # collect expert outputs
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            forwarded_hidden_states = {}
            handles = []
            def _get_activation_hook(name):
                def hook(module, input, output):
                    forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
                return hook
            
            for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging] Registering forward hook..."):
                ffn_name = f"model.layers.{layer_idx}.mlp"
                forwarded_hidden_states[ffn_name] = []
                moe = model.model.layers[layer_idx].mlp
                handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

            for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to collect moe inputs"):
                batch = {k: v.cuda() for k, v in batch.items()}
                if "labels" in batch:
                    batch.pop("labels")
                with torch.no_grad():
                    outputs = model(**batch)
                    del outputs
            
            for handle in handles:
                handle.remove()
            torch.cuda.empty_cache()

            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                layer_input = torch.cat(forwarded_hidden_states[ffn_name]) # .cuda()
                expert_outputs = [] # (E, #T, D) -> average -> (E, D)
                with torch.no_grad():
                    for i in range(self.num_experts):
                        expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                    expert_outputs = torch.stack(expert_outputs)
                    score = compute_silhouette_score(expert_outputs, self._group_state_dict[ffn_name])
                    print(f"layer {layer_idx}: {score}")
                del layer_input
            del forwarded_hidden_states
        elif self.similarity_base == "weight":
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                moe = model.model.layers[layer_idx].mlp
                experts = []
                for i in range(self.num_experts):
                    weight_flat = torch.cat(
                        [moe.experts[i].w1.weight.flatten(),
                        moe.experts[i].w2.weight.flatten(),
                        moe.experts[i].w3.weight.flatten()],
                        dim=0
                    )
                    experts.append(weight_flat)
                experts = torch.stack(experts).to("cuda:7")
                score = compute_silhouette_score(experts, self._group_state_dict[ffn_name])
                del experts
                print(f"layer {layer_idx}: {score}")
        elif self.similarity_base == "router-logits":
            all_router_logits = []
            for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to get routing logits"):
                batch = {k: v.cuda() for k, v in batch.items()}
                if "labels" in batch:
                    batch.pop("labels")
                with torch.no_grad():
                    outputs = model(**batch, output_router_logits=True)
                batch_router_logits = outputs.router_logits
                batch_router_logits = torch.stack(batch_router_logits)
                all_router_logits.append(batch_router_logits)
                del outputs
            all_router_logits = torch.cat(all_router_logits, dim=1)
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T
                score = compute_silhouette_score(layer_router_logits, self._group_state_dict[ffn_name])
                print(f"layer {layer_idx}: {score}")
        else:
            pass



    #NOTE: Clustering
    def cluster_experts(
            self,
            model: DeepseekForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        if self.similarity_base == "weight": 
            dom_experts = self.group_experts_by_clustering_weight(
                model=model,
                num_groups=num_groups
            )
        elif self.similarity_base == "expert-output":
            dom_experts = self.group_experts_by_clustering_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "weight+expert-output":
            dom_experts = self.group_experts_by_clustering_weight_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "router-logits":
            dom_experts = self.group_experts_by_clustering_router_score(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "router-logits+weight":
            dom_experts = self.group_experts_by_clustering_router_score_weight(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "router-logits+expert-output":
            dom_experts = self.group_experts_by_clustering_router_score_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "router-logits+weight+expert-output":
            dom_experts = self.group_experts_by_clustering_router_score_weight_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        else:
            raise ValueError(
                f"Accepted similarity bases are `weight`, `expert-output`, `weight+expert-output`, `router-logits`, `router-logits+weight`, `router-logits+expert-output`, `router-logits+weight+expert-output`, but the input is `{self.similarity_base}`")
        return dom_experts
    

    def group_experts_by_clustering_weight_layerwise(
            self,
            moe: DeepseekMoE,
            ffn_name: str,
            num_groups: int,
    ):
        experts = []
        for i in range(self.num_experts):
            weight_flat = torch.cat(
                [moe.experts[i].w1.weight.flatten(),
                moe.experts[i].w2.weight.flatten(),
                moe.experts[i].w3.weight.flatten()],
                dim=0
            )
            experts.append(weight_flat)
        experts = torch.stack(experts)
        dom_experts, label = group_experts_by_clustering(
            model="deepseek",
            num_groups=num_groups,
            cluster=self.cluster,
            linkage=self.linkage,
            hierarchical_stopping_metric=self.hierarchical_stopping_metric,
            num_experts=self.num_experts,
            experts=experts,
            init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
        self._group_state_dict[ffn_name] = label.cpu()
        return dom_experts

    def group_experts_by_clustering_weight(
        self,
        model: DeepseekForCausalLM,
        num_groups: int,
    ):
        model.eval()
        dom_experts = dict()
        if self.dynamic_group:
            num_groups_per_layer = self._assign_num_groups_per_layer(
                num_groups, self.sparse_layer_indices
            )
        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"Clustering experts by weight"):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            print(ffn_name)
            moe = model.model.layers[layer_idx].mlp
            num_groups_in_layer = num_groups_per_layer[ffn_name] if self.dynamic_group else num_groups
            dom_experts[ffn_name] = self.group_experts_by_clustering_weight_layerwise(moe, ffn_name, num_groups_in_layer)
        return dom_experts
    
    def group_experts_by_clustering_output_layerwise(
            self,
            model: DeepseekForCausalLM,
            dataloader: DataLoader,
            layer_idx: int,
            ffn_name: str,
            num_groups: int,
    ):
        model.eval()
        moe_input = []
        moe = model.model.layers[layer_idx].mlp
        
        def _get_hook(_, input, __): # module, input, output
            moe_input.append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
        
        with torch.no_grad():
            handle = moe.register_forward_hook(_get_hook)
            for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to collect moe inputs"):
                batch = {k: v.cuda() for k, v in batch.items()}
                if "labels" in batch:
                    batch.pop("labels")
                outputs = model(**batch)
            handle.remove()
            torch.cuda.empty_cache()
            
            layer_input = torch.cat(moe_input)
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            for i in range(self.num_experts):
                expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
            expert_outputs = torch.stack(expert_outputs)

            dom_experts, label = group_experts_by_clustering(
                model="deepseek",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=expert_outputs,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
            return dom_experts

    def group_experts_by_clustering_output(
        self,
        model: DeepseekForCausalLM,
        dataloader: DataLoader,
        num_groups: int,
    ):
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging] Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        if self.dynamic_group:
            num_groups_per_layer = self._assign_num_groups_per_layer(
                num_groups, self.sparse_layer_indices
            )

        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]) # .cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            with torch.no_grad():
                for i in range(self.num_experts):
                    expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                expert_outputs = torch.stack(expert_outputs)
                print(expert_outputs.shape)
                num_groups_in_layer = num_groups_per_layer[ffn_name] if self.dynamic_group else num_groups
                dom_experts[ffn_name], label = group_experts_by_clustering(
                    model="deepseek",
                    num_groups=num_groups_in_layer,
                    cluster=self.cluster,
                    linkage=self.linkage,
                    hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                    num_experts=self.num_experts,
                    experts=expert_outputs,
                    init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
                self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_weight_output(
        self,
        model: DeepseekForCausalLM,
        dataloader: DataLoader,
        num_groups: int,
    ):
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging] Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by weight and expert outputs..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]) # .cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            weights = []
            with torch.no_grad():
                for i in range(self.num_experts):
                    output = model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0)
                    weight_flat = torch.cat(
                        [moe.experts[i].w1.weight.flatten(),
                        moe.experts[i].w2.weight.flatten(),
                        moe.experts[i].w3.weight.flatten()],
                        dim=0
                    )
                    weights.append(weight_flat)
                    expert_outputs.append(output)
                expert_outputs = torch.stack(expert_outputs)
                weights = torch.stack(weights)
                dom_experts[ffn_name], label = group_experts_by_clustering(
                    model="deepseek",
                    num_groups=num_groups,
                    cluster=self.cluster,
                    linkage=self.linkage,
                    hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                    num_experts=self.num_experts,
                    experts=weights,
                    experts2=expert_outputs,
                    init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
                self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_router_score(
            self,
            model: DeepseekForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        print("group_experts_by_clustering_router_score")
        model.eval()
        dom_experts = dict()
        all_router_logits = []
        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs 
        
        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, num_tokens, num_experts)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T
            dom_experts[ffn_name], label = group_experts_by_clustering(
                model="deepseek",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=layer_router_logits,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_router_score_weight(
            self,
            model: DeepseekForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        print("group_experts_by_clustering_router_score_weight")
        model.eval()
        dom_experts = dict()
        all_router_logits = []
        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs 
        
        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, num_tokens, num_experts)
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            all_weights = []
            for i in range(self.num_experts):
                weight_flat = torch.cat(
                        [moe.experts[i].w1.weight.flatten(),
                        moe.experts[i].w2.weight.flatten(),
                        moe.experts[i].w3.weight.flatten()],
                        dim=0
                    )
                all_weights.append(weight_flat)
            weights = torch.stack(all_weights)
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T
            all_weights.clear()
            dom_experts[ffn_name], label = group_experts_by_clustering(
                model="deepseek",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=layer_router_logits,
                experts2=weights,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_router_score_output(
            self,
            model: DeepseekForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        print("group_experts_by_clustering_router_score_output")
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []
        all_router_logits = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook

        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging] Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))
        

        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs 
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, num_tokens, num_experts)
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            layer_input = torch.cat(forwarded_hidden_states[ffn_name])
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            for i in range(self.num_experts):
                output = model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0)
                expert_outputs.append(output)
            expert_outputs = torch.stack(expert_outputs)
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T
            dom_experts[ffn_name], label = group_experts_by_clustering(
                model="deepseek",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=layer_router_logits,
                experts2=expert_outputs,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_router_score_weight_output(
            self,
            model: DeepseekForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        print("group_experts_by_clustering_router_score_output")
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []
        all_router_logits = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook

        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging] Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))
        

        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs 
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, num_tokens, num_experts)
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            layer_input = torch.cat(forwarded_hidden_states[ffn_name])
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            weights = []
            for i in range(self.num_experts):
                output = model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0)
                weight_flat = torch.cat(
                        [moe.experts[i].w1.weight.flatten(),
                        moe.experts[i].w2.weight.flatten(),
                        moe.experts[i].w3.weight.flatten()],
                        dim=0
                    )
                weights.append(weight_flat)
                expert_outputs.append(output)
            expert_outputs = torch.stack(expert_outputs)
            weights = torch.stack(weights)
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T
            dom_experts[ffn_name], label = group_experts_by_clustering(
                model="deepseek",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=layer_router_logits,
                experts2=weights,
                experts3=expert_outputs,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts


    def group_experts_globally_from_dominant_experts(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, List[int]]:
        """
        Globally group experts into clusters by routing-guided clustering, each layer will have different number of
         clusters. The total number of clusters is determined by num_average_groups.

        Parameters
        ----------
        num_average_groups: int
            The average number of clusters for all layers.
        merging_layers: List[int]
            The layers of decoder that are excluded from merging.

        Returns
        -------
        core_experts: Dict[str, List[int]]
            The core experts of each cluster
        """

        # 1. Assign num_groups respectively for each layer according to num_average_groups
        num_groups_per_layer = self._assign_num_groups_per_layer(
            num_average_groups, merging_layers
        )
        print(f"[HC-SMoE] Number of groups per layer: {num_groups_per_layer}")

        # 2. Group experts into clusters for each layer
        dom_experts = dict()
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[HC-SMoE] Globally grouping experts into average {num_average_groups} clusters"
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups = num_groups_per_layer[ffn_name]
            group_member_count = torch.zeros(num_groups)

            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[ffn_name], descending=True)
            # 1 Assign top-K most-used experts with label 0 to K-1 respectively
            group_dict = {} 
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            dom_experts[ffn_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[ffn_name][indices_sorted_by_usage[i]] = i
                group_member_count[i] += 1
                group_dict[i] = [core_expert_indices[i].item()]
            # 2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(ffn_name)
            print(similarity_matrix)
            print(core_expert_indices)
            for i in range(0, self.num_experts):
                if i in core_expert_indices:
                    continue
                # Find the most similar core
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                self._group_state_dict[ffn_name][i] = most_similar_group_label
                group_member_count[most_similar_group_label] += 1
                group_dict[most_similar_group_label.item()].append(i)
                print(f"--expert {i} is assigned to group {most_similar_group_label}, the core expert is {most_similar_core}")
                if group_member_count[self._group_state_dict[ffn_name][i]] > self.group_limit:
                    if len(core_expert_indices) == 1 and self.group_limit < self.num_experts:
                        raise ValueError(
                            f"[Merging]The number of groups at Encoder layer {layer_idx} is too small!"
                        )
                    while group_member_count[most_similar_group_label] > self.group_limit:
                        print(f"----meet group limit {self.group_limit} with group {most_similar_group_label} (core: {most_similar_core})")
                        # Find the most unsimilar expert in the exceed group
                        sim = similarity_matrix[most_similar_core, group_dict[most_similar_group_label.item()]]
                        unsimilar_pos = torch.argmin(sim).item()
                        if (unsimilar_pos == 0):
                            unsimilar_pos = 1
                        unsimilar_idx = group_dict[most_similar_group_label.item()][unsimilar_pos]
                    
                        group_member_count[most_similar_group_label] -= 1
                        group_dict[most_similar_group_label.item()].remove(unsimilar_idx)
                        similarity_matrix[unsimilar_idx, most_similar_core] = -100
                        similarity_matrix[most_similar_core, unsimilar_idx] = -100
                        print(f"----kick out {unsimilar_idx} from group ")
                        # Reassign group label
                        most_similar_core = core_expert_indices[
                            torch.argmax(similarity_matrix[unsimilar_idx, core_expert_indices])
                        ]
                        most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                        self._group_state_dict[ffn_name][unsimilar_idx] = most_similar_group_label
                        group_member_count[most_similar_group_label] += 1
                        group_dict[most_similar_group_label.item()].append(unsimilar_idx)
                        print(f"--expert {unsimilar_idx} is assigned to group {most_similar_group_label}, the core expert is {most_similar_core}")
        return dom_experts
    
    
    def group_experts_layerwise_by_freq(
        self,
        num_groups: int,
    ) -> Dict[str, List[int]]:
        core_experts = dict()
        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"Grouping experts layerwise by frequency"):
            moe_name = f"model.layers.{layer_idx}.mlp"
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[moe_name], descending=True)
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            core_experts[moe_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[moe_name][core_expert_indices[i]] = i
            similarity_matrix = self.get_similarity_matrix(moe_name)
            for i in range(num_groups, self.num_experts):
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[moe_name][most_similar_core]
                self._group_state_dict[moe_name][i] = most_similar_group_label
        return core_experts

    def compute_all_usages(
            self,
            model: DeepseekForCausalLM,
            dataloader: DataLoader,
            mode: str = "frequency", # frequency, routing-score
    ):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        config = model.config
        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Evaluating routing distribution"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            all_router_logits = outputs.router_logits
            if mode == "frequency":
                all_router_logits = torch.stack(all_router_logits)  # of shape (num_hidden_layers, num_tokens, num_experts)
                selected_experts = torch.topk(all_router_logits, 2, dim=-1)[1].reshape(
                    config.num_hidden_layers-1, -1
                )  # of shape (num_hidden_layers, num_tokens * 2)
                for layer_idx in self.sparse_layer_indices:
                    ffn_name = f"model.layers.{layer_idx}.mlp"
                    unique, counts = torch.unique(selected_experts[layer_idx-1], return_counts=True)
                    self._usage_frequency_state_dict[ffn_name][unique.cpu()] += counts.cpu()
            else: # routing-score
                for layer_idx in self.sparse_layer_indices:
                    ffn_name = f"model.layers.{layer_idx}.mlp"
                    router_score = F.softmax(all_router_logits[layer_idx], dim=1)
                    scores = router_score.float().sum(0) / router_score.shape[0]
                    self._usage_frequency_state_dict[ffn_name] += scores.cpu()
        if mode == "frequency":
            self._usage_frequency_state_dict = {
                k: v / torch.sum(v) for k, v in self._usage_frequency_state_dict.items()
            }


    #NOTE: Compute similarities
    def compute_all_similarities(
            self,
            model: DeepseekForCausalLM,
            dataloader: DataLoader = None
    ):
        similarity_list = ["weight", "router-weight", "router-logits", "expert-output"]
        if self.similarity_base not in similarity_list and dataloader is None:
            raise ValueError(
                "[HC-SMoE] `dataloader` should be provided when similarity_base is not 'weight' or 'router-weight'")
        model = model.eval()
        if self.similarity_base == "weight":
            self._compute_all_similarities_by_weight(model.state_dict())
        elif self.similarity_base == 'router-weight':
            self._compute_all_similarities_by_router_weight(model.state_dict())
        elif self.similarity_base == 'router-logits':
            self._compute_all_similarities_by_router_logits(model, dataloader)
        elif self.similarity_base == 'expert-output':
            self._compute_all_similarities_by_expert_outputs(model, dataloader)
        else:
            raise NotImplementedError
        
    def _compute_all_similarities_by_weight(self, state_dict: Dict[str, torch.Tensor]):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by weight..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_flat = torch.cat(
                        [state_dict[f"{ffn_name}.experts.{i}.w1.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{i}.w2.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{i}.w3.weight"].flatten()],
                        dim=0
                    )
                    j_flat = torch.cat(
                        [state_dict[f"{ffn_name}.experts.{j}.w1.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{j}.w2.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{j}.w3.weight"].flatten()],
                        dim=0
                    )
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_router_weight(self, state_dict: Dict[str, torch.Tensor]):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by router rows..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_flat = state_dict[f"{ffn_name}.gate.weight"][i]
                    j_flat = state_dict[f"{ffn_name}.gate.weight"][j]
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_router_logits(self, model: DeepseekForCausalLM, dataloader: DataLoader):
        model.eval()
        all_router_logits = []
        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs

        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, *, num_experts)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts)
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        i_flat = layer_router_logits[:, i].flatten()
                        j_flat = layer_router_logits[:, j].flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)
                
    def _compute_layer_similarities_by_weight(self, state_dict: Dict[str, torch.Tensor], layer_idx: int):
        ffn_name = f"model.layers.{layer_idx}.mlp"
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                i_flat = torch.cat(
                    [state_dict[f"{ffn_name}.experts.{i}.w1.weight"].flatten(),
                        state_dict[f"{ffn_name}.experts.{i}.w2.weight"].flatten(),
                        state_dict[f"{ffn_name}.experts.{i}.w3.weight"].flatten()],
                    dim=0
                )
                j_flat = torch.cat(
                    [state_dict[f"{ffn_name}.experts.{j}.w1.weight"].flatten(),
                        state_dict[f"{ffn_name}.experts.{j}.w2.weight"].flatten(),
                        state_dict[f"{ffn_name}.experts.{j}.w3.weight"].flatten()],
                    dim=0
                )
                similarity = self.similarity_fn(i_flat, j_flat)
                self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_expert_outputs(self, model: DeepseekForCausalLM, dataloader: DataLoader):
        model.eval()
        forwarded_hidden_states = {} # moe input
        handles = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook
        
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[Merging] Registering forward hook..."
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            handles.append(model.model.layers[layer_idx].mlp.register_forward_hook(
                _get_activation_hook(ffn_name))
            )

        for batch in tqdm(dataloader, desc=f"[HC-SMoE] Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[HC-SMoE] Computing similarities by expert outputs..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]) # .cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            with torch.no_grad():
                for i in range(self.num_experts):
                    if self.overlap_metric == "cosine":
                        expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                    else:
                        expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input))
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        if i == j:
                            self.save_similarity(ffn_name, i, j, 1.0)
                            continue
                        if self.overlap_metric == "kl-divergence":
                            p = get_prob_distributions(expert_outputs[i])
                            q = get_prob_distributions(expert_outputs[j])
                            similarity = compute_kl_divergence(p, q)
                        elif self.overlap_metric == "wasserstein": # wasserstein
                            similarity = compute_wasserstein_distance(expert_outputs[i], expert_outputs[j])
                        else: # cosine
                            i_flat = expert_outputs[i].flatten()
                            j_flat = expert_outputs[j].flatten()
                            similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)
                        self.save_similarity(ffn_name, j, i, similarity)
            del layer_input
        torch.cuda.empty_cache()
    

#######################
####    Merging    ####
#######################

### Helper Functions for Merging
def apply_mask(module, _mask):
    # applying masks to the input to compute gradients
    def masking(_, i):
        return _mask * i[0]

    handle = module.register_forward_pre_hook(masking)
    return handle

def hijack(module, _list, _hijack_input, _stop_forward=False):
    # if _stop_forward=True, then it raise error after forwarding the module
    if _hijack_input:
        def input_hook(_, inputs, __):
            _list.append(inputs[0].detach().cpu()) # .clone().data
            # if _stop_forward:
                # raise StopFowardException

        handle = module.register_forward_hook(input_hook)
    else:
        def output_hook(_, __, outputs):
            if isinstance(outputs, tuple):
                _list.append(outputs[0].detach().cpu())
            else:
                _list.append(outputs.detach()) # .clone().data
            # if _stop_forward:
                # raise StopFowardException
        handle = module.register_forward_hook(output_hook)
    return handle  

def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)

def remove_row(x, idx):
    return torch.cat([x[:idx], x[idx+1:]], dim=0)   

@torch.no_grad()
def collect_act(data, weight1, weight3=None):
    activations = []
    act = torch.nn.SiLU()
    if weight3 is not None:
        cur = act(torch.matmul(data, weight1.T)) * torch.matmul(data, weight3.T)
    else:
        cur = torch.matmul(data, weight1.T)
    activations.append(cur.reshape(-1, cur.shape[-1]))
    return torch.cat(activations, dim=0) # T x N, T is the number of tokens, N is the intermediate size

@torch.no_grad()
def collect_feature(ingredient, data, weight1, weight2, weight3):
    if ingredient == "act":
        return collect_act(data, weight1, weight3)
    elif ingredient == "weight":
        # weigh1, weigh3: NxD, weigh2: DxN
        return torch.cat([weight1.T, weight2, weight3.T], dim=0)
    else: # both
        return collect_act(data, weight1, weight3), torch.cat([weight1.T, weight2, weight3.T], dim=0)

@torch.no_grad()
def compute_covariance(act1, act2):
    with torch.no_grad():
        print(f"compute covariance: {act1.shape}, {act2.shape}")
        mean1 = act1.mean(dim=0, keepdim=True)
        mean2 = act2.mean(dim=0, keepdim=True)
        std1 = act1.std(dim=0, keepdim=True)
        std2 = act2.std(dim=0, keepdim=True)
        corr_matrix = torch.matmul((act1 - mean1).T, act2 - mean2) / (act1.shape[0] - 1)
        mean1 = mean1.to("cpu")
        mean2 = mean2.to("cpu")
        del mean1, mean2
        torch.cuda.empty_cache()
        corr_matrix = corr_matrix / (std1.T * std2 + FP32_EPS)
        del std1, std2
        torch.cuda.empty_cache()
    return corr_matrix # N x N

@torch.no_grad()
def compute_feature_covariance(ingredient, data1, data2):
    if ingredient == "act+weight":
        corr1 = compute_covariance(data1[0], data2[0])
        corr2 = compute_covariance(data1[1], data2[1])
        return corr1 + corr2
    else:
        return compute_covariance(data1, data2)

def get_coef(num_ffn, input_weight, average_coefs, d_ff=None):
    if d_ff == None: # fix-dom-same
        if input_weight is not None:
            coef = input_weight
        elif average_coefs is None:
            coef = [1.0] * num_ffn
        elif len(average_coefs) == num_ffn:
            coef = average_coefs
        else:
            coef = [1.0] * num_ffn
    else: # zipit
        if input_weight is not None:
            coef = []
            for w in input_weight:
                coef = [w] * d_ff
                coef.extend(coef)
        elif average_coefs is None:
            coef = [1.0] * num_ffn * d_ff
        elif len(average_coefs) == num_ffn:
            coef = [coef for coef in average_coefs for _ in range(d_ff)]
        elif len(average_coefs) != num_ffn * d_ff:
            raise ValueError(
                f"The length of average_coefs should be either {num_ffn} or {num_ffn * d_ff}, "
                f"but got {len(average_coefs)}."
            )
    return coef


@torch.no_grad()
def _merge_mlp_experts_by_usage_frequency_weighting(
        ffn: DeepseekMoE,
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor,
) -> DeepseekMoE:

    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        # print(f"group {label} with experts {expert_indices}, weight {usage_frequencies[expert_indices]}")
        w1_weight_list = torch.stack(
            [ffn.experts[expert_idx].up_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        w2_weight_list = torch.stack(
            [ffn.experts[expert_idx].gate_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        w3_weight_list = torch.stack(
            [ffn.experts[expert_idx].down_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        w1_weight = torch.sum(w1_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
        w2_weight = torch.sum(w2_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
        w3_weight = torch.sum(w3_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)

        ffn.experts[expert_indices[0]].up_proj.weight.copy_(w1_weight)
        ffn.experts[expert_indices[0]].gate_proj.weight.copy_(w2_weight)
        ffn.experts[expert_indices[0]].down_proj.weight.copy_(w3_weight)

        for expert_idx in expert_indices[1:]:
            # Binding merged experts to the first of them
            ffn.experts[expert_idx] = ffn.experts[expert_indices[0]]

    return ffn


@torch.no_grad()
def _zipit_merge(temp_dim, target_dim, weight1, weight3, data,):
    permutation_matrix = torch.eye(temp_dim, temp_dim).to(weight1.dtype)
    ROUND = 0
    act = torch.nn.SiLU()
    while temp_dim > target_dim:
        ROUND += 1
        odd = temp_dim % 2
        target_dim_this_round = max(target_dim, temp_dim // 2 + odd)
        print(f"ROUND {ROUND}. From {temp_dim} to {target_dim_this_round}")
        
        ### Collect activations
        activations = []
        if weight3 is None:
            cur = torch.matmul(data, weight1.T)
        else:
            cur = act(torch.matmul(data, weight1.T)) * torch.matmul(data, weight3.T)
        activations.append(cur.reshape(-1, cur.shape[-1]))
        activations = torch.cat(activations, dim=0)
        print("Activations: ", activations.shape)
        ### Compute covariance
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True)
        covar = torch.matmul((activations - mean).T, activations - mean) / (activations.shape[0] - 1)
        corr_matrix = covar / (std.T * std + FP32_EPS)
        del mean, std, covar
        torch.cuda.empty_cache()
        corr_matrix[torch.arange(temp_dim), torch.arange(temp_dim)] = -1 # Remove self-correlation
        print(corr_matrix)
        ### Merge temp_dim / 2 times
        for _ in range(temp_dim - target_dim_this_round):
            max_index = torch.argmax(corr_matrix)
            row, col = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]
            permutation_matrix[:, row] += permutation_matrix[:, col]
            permutation_matrix = remove_col(permutation_matrix, col)

            # row_coef, col_coef = average_coefs[row], average_coefs[col]
            row_coef, col_coef = 1.0, 1.0
            weight1[row] = (row_coef * weight1[row] + col_coef * weight1[col]) / (row_coef + col_coef + FP32_EPS)
            if weight3 is not None:
                weight3[row] = (row_coef * weight3[row] + col_coef * weight3[col]) / (row_coef + col_coef + FP32_EPS)
                weight3 = remove_row(weight3, col)
            weight1 = remove_row(weight1, col)
            
            corr_matrix[row] = FP32_EPS # set very small number to avoid repeated merging
            corr_matrix[:, row] = FP32_EPS
            corr_matrix[row, row] = -1
            corr_matrix = remove_col(corr_matrix, col)
            corr_matrix = remove_row(corr_matrix, col)
        temp_dim = weight1.shape[0]
    for i in range(20): # permutation_matrix.shape[1]
        print(permutation_matrix[:, i].nonzero().squeeze())
    return permutation_matrix

@torch.no_grad()
def _merge_moe_experts_by_zipit(
    ffn_list: List[DeepseekMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = None,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
) -> DeepseekMLP:
    print("zipit-recompute-correlation")
    d_ff, d_model = ffn_list[0].w1.out_features, ffn_list[0].w1.in_features
    num_ffn = len(ffn_list)
    temp_dim = d_ff * num_ffn
    average_coefs = [1.0] * temp_dim
    act = torch.nn.SiLU()

    _device = ffn_list[0].w1.weight.device
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {temp_dim}, target_dim: {d_ff}")

    ### Merge W1 and W3
    ffn_all_w1 = torch.cat([ffn.w1.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_w3 = torch.cat([ffn.w3.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    first_permutation_matrix = _zipit_merge(d_ff * num_ffn, d_ff, ffn_all_w1, ffn_all_w3, forwarded_hidden_states).to(_device)
    first_unmerge_matrix = first_permutation_matrix
    first_merge_matrix = torch.div(first_permutation_matrix, torch.sum(first_permutation_matrix, dim=0, keepdim=True))

    ffn_all_w1 = torch.cat([ffn.w1.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_w3 = torch.cat([ffn.w3.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_w1 = torch.matmul(first_merge_matrix.T, ffn_all_w1)
    ffn_w3 = torch.matmul(first_merge_matrix.T, ffn_all_w3)

    ### Merge W2
    new_data = act(torch.matmul(forwarded_hidden_states, ffn_w1.T)) * torch.matmul(forwarded_hidden_states, ffn_w3.T)
    ffn_all_w2 = torch.cat([ffn.w2.weight.data for ffn in ffn_list], dim=0) # (d_model * num_ffn, d_ff)
    second_permutation_matrix = _zipit_merge(d_model * num_ffn, d_model, ffn_all_w2, None, new_data).to(_device)
    second_merge_matrix = torch.div(second_permutation_matrix, torch.sum(second_permutation_matrix, dim=0, keepdim=True))
    ffn_w2 = torch.zeros(d_model, d_ff).to(_device)
    for i in range(num_ffn):
        ffn_w2 += torch.matmul(second_merge_matrix.T[:, i*d_model:(i+1)*d_model], torch.matmul(ffn_all_w2[i*d_model:(i+1)*d_model], first_unmerge_matrix.T[:, i*d_ff:(i+1)*d_ff]))

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.w1.weight.data = ffn_w1
    merged_ffn.w2.weight.data = ffn_w2
    merged_ffn.w3.weight.data = ffn_w3

    return merged_ffn


@torch.no_grad()
def _merge_moe_experts_with_dominant_same_rule(
        ffn_list: List[DeepseekMLP],
        forwarded_hidden_states: torch.Tensor,
        average_coefs: Optional[List[float]] = None,
        input_weight: Optional[List[float]] = None,
        dominant_index: Optional[int] = 0,
        ingredient: Optional[str] = "act+weight", # act, weight, act+weight
        mode: Optional[str] = "normal", # normal, cluster
):
    print("merge: fix-dom-same-rule-without-unmerge")
    d_ff = ffn_list[0].w1.out_features
    num_ffn = len(ffn_list)
    _device = ffn_list[0].w1.weight.device
    _dtype = ffn_list[0].w1.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)

    # Move dominant expert to the first
    if dominant_index != 0:
        if input_weight != None:
            input_weight[0], input_weight[dominant_index] = input_weight[dominant_index], input_weight[0]
        ffn_list[0], ffn_list[dominant_index] = ffn_list[dominant_index], ffn_list[0]
    
    coef = get_coef(num_ffn, input_weight, average_coefs)
    print("coef=", coef)
    
    print("dominant_index: ", dominant_index)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {d_ff * num_ffn}, target_dim: {d_ff}, dominant_index: {dominant_index}, dtype: {_dtype}. device: {_device}")
    
    ### Kmeans clustering
    if mode == "cluster":
        # 1. Concat activations
        activations = []
        for ffn in ffn_list:
            cur = collect_act(forwarded_hidden_states, ffn.w1.weight.data, ffn.w3.weight.data)
            activations.append(cur)
        activations = torch.cat(activations, dim=1).T.to("cuda:7")
        print(activations.shape)

        # 2. Kmeans clustering
        centers = activations[:d_ff]
        min_points_per_cluster = 1

        for _ in range(100):
            distance = torch.cdist(activations, centers)
            assignments = torch.argmin(distance, dim=1)
            del distance
            for i in range(d_ff):
                num_points_in_cluster = torch.sum(assignments == i)
                if num_points_in_cluster < min_points_per_cluster:
                    # Find overpopulated clusters
                    for j in range(d_ff):
                        if i != j and torch.sum(assignments == j) > num_ffn:
                            # Move points from overpopulated cluster j to underpopulated cluster i
                            diff = torch.sum(assignments == j) - min_points_per_cluster

                            # Select `num_to_move` points from cluster j and reassign them to cluster i
                            reassign_indices = torch.where(assignments == j)[0][0]
                            assignments[reassign_indices] = i
                            print(f"Group {i} has {num_points_in_cluster} points, move 1 point from group {j}")
                            break
                            
            # Recompute the centers after ensuring the minimum number of points
            group_members = []
            for i in range(d_ff):
                group_member = activations[assignments == i].mean(dim=0)
                if torch.isnan(group_member).sum().item() > 0:
                    print(f"Group {i}: {torch.nonzero(assignments == i).squeeze()} {group_member.shape} {torch.isnan(group_member).sum()}")
                group_members.append(group_member)
            new_centers = torch.stack(group_members)
            max_diff = 0
            for i in range(d_ff):
                diff = torch.max(torch.abs(centers[i] - new_centers[i]))
                max_diff = max(max_diff, diff.item())
            print(f"max_diff: {max_diff}")
            if max_diff < 1e-4:
                print("Converged!")
                break
            centers = new_centers
        
        permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, dtype=torch.float16, device="cuda:7")
        for i in range(d_ff):
            index_in_this_group = (assignments == i).nonzero().squeeze()
            permutation_matrix[i, index_in_this_group] = 1
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True)).to(_dtype)
        for i in range(5): # permutation_matrix.shape[1]
            print(permutation_matrix[:, i].nonzero().squeeze())
        permutation_matrix = permutation_matrix.to(_device)
    else:
        # dom_act = collect_act(forwarded_hidden_states, ffn_list[0].w1.weight.data, ffn_list[0].w3.weight.data) # T x d_ff
        dom_act = collect_feature(ingredient, forwarded_hidden_states, ffn_list[0].w1.weight.data, ffn_list[0].w2.weight.data, ffn_list[0].w3.weight.data)
        group_indexes = [[]]
        for i in range(1, num_ffn):
            other_act = collect_feature(ingredient, forwarded_hidden_states, ffn_list[i].w1.weight.data, ffn_list[i].w2.weight.data, ffn_list[i].w3.weight.data)
            corr_matrix = compute_feature_covariance(ingredient, dom_act, other_act)
            print(f"corr_matrix: {i}, {corr_matrix[0]}")
            # corr_matrix = d_ff x d_ff, first dimension is the index of dominant expert, second dimension is the index of other experts
            # we want to find the maximum value for each dim in `other experts` = find maximum value for each column -> dim = 0
            max_index = torch.argmax(corr_matrix, dim=0)
            print(f"max_index: {max_index.shape} {max_index}")
            group_indexes.append(max_index)
            
            del other_act, corr_matrix
        
        permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, dtype=torch.float16, device=_device) * coef[0]
        for i in range(d_ff):
            for j in range(1, num_ffn):
                index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_ff * j
                permutation_matrix[i, index_in_this_group] = coef[j]
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True)).to(_dtype)
        print(f"first permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
        del dom_act

    # merge weight
    ffn_all_w1 = torch.cat([ffn.w1.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_w2 = torch.cat([ffn.w2.weight.data for ffn in ffn_list], dim=1) # (d_model, d_ff * num_ffn)
    ffn_all_w3 = torch.cat([ffn.w3.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_w1 = torch.matmul(permutation_matrix, ffn_all_w1)
    ffn_w2 = torch.matmul(permutation_matrix, ffn_all_w2.T)
    ffn_w3 = torch.matmul(permutation_matrix, ffn_all_w3)
    
    # clear memory
    ffn_all_w1 = ffn_all_w1.to('cpu')
    ffn_all_w2 = ffn_all_w2.to('cpu')
    ffn_all_w3 = ffn_all_w3.to('cpu')
    # dom_act = dom_act.to('cpu')
    permutation_matrix = permutation_matrix.to('cpu')
    forwarded_hidden_states = forwarded_hidden_states.to('cpu')
    del ffn_all_w1, ffn_all_w2, ffn_all_w3, permutation_matrix, forwarded_hidden_states
    torch.cuda.empty_cache()

    # save result
    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.w1.weight.data = ffn_w1
    merged_ffn.w2.weight.data = ffn_w2.T
    merged_ffn.w3.weight.data = ffn_w3

    return merged_ffn

@torch.no_grad()
def _merge_moe_experts_with_dominant(
        ffn_list: List[DeepseekMLP],
        forwarded_hidden_states: torch.Tensor,
        mini_batch_size: Optional[int] = None,
        alpha_for_repeated_merging: Optional[float] = 0.1,
        average_coefs: Optional[List[float]] = None,
        input_weight: Optional[List[float]] = None,
        dominant_index: Optional[int] = 0,
):
    print("merge: fix-dom-independent-rule-without-unmerge")
    d_ff, d_model = ffn_list[0].w1.out_features, ffn_list[0].w1.in_features
    num_ffn = len(ffn_list)
    need_pinv = False
    if input_weight is not None:
        coef = input_weight
        need_pinv = True
    elif average_coefs is None:
        coef = [1.0] * num_ffn
    elif len(average_coefs) == num_ffn:
        coef = average_coefs
        need_pinv = True
    else:
        coef = [1.0] * num_ffn
    
    if dominant_index != 0:
        ffn_list[0], ffn_list[dominant_index] = ffn_list[dominant_index], ffn_list[0]
    print("dominant_index: ", dominant_index)
    _device = ffn_list[0].w1.weight.device
    _dtype = ffn_list[0].w1.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {d_ff * num_ffn}, target_dim: {d_ff}, dominant_index: {dominant_index}")
    # Compute Permutation Matrix for w1 and w3
    permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, device=_device, dtype=_dtype) * coef[0]
    dom_act = collect_act(forwarded_hidden_states, ffn_list[dominant_index].w1.weight.data, ffn_list[dominant_index].w3.weight.data)
    group_indexes = []
    for i in range(num_ffn):
        if i == dominant_index:
            continue
        other_act = collect_act(forwarded_hidden_states, ffn_list[i].w1.weight.data, ffn_list[i].w3.weight.data)
        corr_matrix = compute_covariance(dom_act, other_act)
        max_index = torch.argmax(corr_matrix, dim=1)
        group_indexes.append(max_index)
    for i in range(d_ff):
        for j in range(num_ffn - 1):
            index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_ff * (j + 1)
            permutation_matrix[i, index_in_this_group] = coef[j]
    if not need_pinv:
        unmerge_1 = permutation_matrix
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
    else:
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
        unmerge_1 = torch.linalg.pinv(permutation_matrix.to(torch.float)).to(_dtype).T
        permutation_matrix = permutation_matrix.to(_dtype)
    
    print(f"first permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
    ffn_all_w1 = torch.cat([ffn.w1.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_w3 = torch.cat([ffn.w3.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_w1 = torch.matmul(permutation_matrix, ffn_all_w1)
    ffn_w3 = torch.matmul(permutation_matrix, ffn_all_w3)

    del ffn_all_w1, ffn_all_w3

    # Compute Permutation Matrix for w2
    permutation_matrix = torch.eye(d_model, d_model * num_ffn, dtype=_dtype, device=_device) * coef[0]
    new_data = collect_act(forwarded_hidden_states, ffn_w1, ffn_w3)
    dom_act = collect_act(new_data, ffn_list[dominant_index].w2.weight.data, None)
    group_indexes.clear()
    for i in range(num_ffn):
        if i == dominant_index:
            continue
        other_act = collect_act(new_data, ffn_list[i].w2.weight.data, None)
        corr_matrix = compute_covariance(dom_act, other_act)
        max_index = torch.argmax(corr_matrix, dim=1)
        group_indexes.append(max_index)
    for i in range(d_model):
        for j in range(num_ffn - 1):
            index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_model * (j + 1)
            permutation_matrix[i, index_in_this_group] = coef[j]
    permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
    print(f"second permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
    ffn_all_w2 = torch.cat([ffn.w2.weight.data for ffn in ffn_list], dim=0) # (d_model * num_ffn, d_ff)
    ffn_w2 = torch.zeros(d_model, d_ff).to(_device)
    for i in range(num_ffn):
        ffn_w2 += torch.matmul(permutation_matrix[:, i*d_model:(i+1)*d_model],
            torch.matmul(ffn_all_w2[i*d_model:(i+1)*d_model], 
                         unmerge_1[:, i*d_ff:(i+1)*d_ff])
        )

    del ffn_all_w2

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.w1.weight.data = ffn_w1
    merged_ffn.w2.weight.data = ffn_w2
    merged_ffn.w3.weight.data = ffn_w3

    return merged_ffn

def _merge_deepseek_moe_by_activation_matching_within_and_across_models_same_rule_with_unmerge(
    ffn_list: List[DeepseekMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = None,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
    ingredient: Optional[str] = "act", # act, weight, act+weight
) -> DeepseekMLP:
    print("merge: zipit-same-rule-with-unmerge")
    ffn_list = [ffn.eval() for ffn in ffn_list]
    concat_ffn = deepcopy(ffn_list[0])
    
    d_ff, d_model = ffn_list[0].w1.out_features, ffn_list[0].w1.in_features
    num_ffn = len(ffn_list)
    _device = concat_ffn.w1.weight.device
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    
    average_coefs = get_coef(num_ffn, input_weight, average_coefs, d_ff)
    
    if mini_batch_size is None:
        mini_batch_size = forwarded_hidden_states.shape[0]

    
    ffn_all_w1 = torch.cat([ffn.w1.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_w2 = torch.cat([ffn.w2.weight.data for ffn in ffn_list], dim=1) # (d_model, d_ff * num_ffn)
    ffn_all_w3 = torch.cat([ffn.w3.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    concat_ffn.w1 = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.w2 = torch.nn.Linear(d_ff * num_ffn, d_model, bias=False)
    concat_ffn.w3 = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.w1.weight.data = ffn_all_w1
    concat_ffn.w2.weight.data = ffn_all_w2
    concat_ffn.w3.weight.data = ffn_all_w3

    activations, weights = None, None
    if "act" in ingredient:
        handles = []
        activations = []
        def _activation_hook(module, input, output):
            activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))
            return _activation_hook
        handle = concat_ffn.w2.register_forward_hook(_activation_hook) 
        print(f"Collect activations with batch size {mini_batch_size} with original data length {forwarded_hidden_states.shape}")
        concat_ffn = concat_ffn.eval().to(forwarded_hidden_states.device)
        for i in range(0, forwarded_hidden_states.shape[0], mini_batch_size): # mini_batch_size = 10000
            concat_ffn(forwarded_hidden_states[i:i + mini_batch_size])  # mini_batch_size * 14336 -> activation: mini_batch_size * 32768 * num_ffn
        for handle in handles:
            handle.remove()
        del handles, forwarded_hidden_states
        activations = torch.cat(activations, dim=0)
        print(f"Collected activations: {activations.shape} {activations.device}")
    if "weight" in ingredient:
        if "act" not in ingredient:
            del forwarded_hidden_states
        weights = []
        for ffn in ffn_list:
            concat_weight = torch.cat([ffn.w1.weight.data.T, ffn.w2.weight.data, ffn.w3.weight.data.T], dim=0) # 3DxN
            weights.append(concat_weight)
        weights = torch.cat(weights, dim=1) # 3Dx(N*num_ffn)
        
        
    if ingredient == "act":
        corr_matrix = compute_covariance(activations, activations)
    elif ingredient == "weight":
        corr_matrix = compute_covariance(weights, weights)
    else:
        corr_matrix = compute_covariance(activations, activations) + compute_covariance(weights, weights)
    torch.cuda.empty_cache()

    corr_matrix[torch.arange(d_ff * num_ffn), torch.arange(d_ff * num_ffn)] = -1  # Remove self-correlation
    print(f"corr_matrix: {corr_matrix.shape}")
    permutation_matrix = torch.eye(d_ff * num_ffn, d_ff * num_ffn, device=_device, dtype=ffn_all_w1.dtype)

    # Greedy Merging!
    while ffn_all_w1.shape[0] > d_ff:
        # Select the most correlated pair
        max_index = torch.argmax(corr_matrix)
        max_i, max_j = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]

        # Merge the most correlated pair, replace the first feature with the merged one
        i_coef, j_coef = average_coefs[max_i], average_coefs[max_j]
        ffn_all_w1[max_i] = (i_coef * ffn_all_w1[max_i] + j_coef * ffn_all_w1[max_j]) / (i_coef + j_coef + FP32_EPS)
        ffn_all_w3[max_i] = (i_coef * ffn_all_w3[max_i] + j_coef * ffn_all_w3[max_j]) / (i_coef + j_coef + FP32_EPS)
        ffn_all_w2[:, max_i] = (i_coef * ffn_all_w2[:, max_i] + j_coef * ffn_all_w2[:, max_j]) / (
                i_coef + j_coef + FP32_EPS)
        permutation_matrix[:, max_i] += permutation_matrix[:, max_j]
        permutation_matrix = remove_col(permutation_matrix, max_j)
       
        # Remove the second feature
        ffn_all_w1 = torch.cat([
            ffn_all_w1[:max_j],
            ffn_all_w1[max_j + 1:]
        ], dim=0)
        ffn_all_w3 = torch.cat([
            ffn_all_w3[:max_j],
            ffn_all_w3[max_j + 1:]
        ], dim=0)
        ffn_all_w2 = torch.cat([
            ffn_all_w2[:, :max_j],
            ffn_all_w2[:, max_j + 1:]
        ], dim=1)

        # Update the correlation matrix
        updated_corr_vec = alpha_for_repeated_merging * torch.min(
            torch.stack([corr_matrix[max_i], corr_matrix[max_j]]), dim=0
        ).values
        corr_matrix[max_i] = updated_corr_vec
        corr_matrix[:, max_i] = updated_corr_vec
        corr_matrix[max_i, max_i] = -1  # Remove self-correlation

        # Remove the second feature from the correlation matrix
        corr_matrix = torch.cat([
            corr_matrix[:, :max_j],
            corr_matrix[:, max_j + 1:]
        ], dim=1)
        corr_matrix = torch.cat([
            corr_matrix[:max_j],
            corr_matrix[max_j + 1:]
        ], dim=0)

        # Update the average coefs
        average_coefs[max_i] += average_coefs[max_j]
        average_coefs = average_coefs[:max_j] + average_coefs[max_j + 1:]

    permutation_matrix = permutation_matrix / torch.sum(permutation_matrix, dim=0, keepdim=True) # 3N x N
    print(f"permutation_matrix: {permutation_matrix.shape} {permutation_matrix}")
    unmerge_matrix = torch.linalg.pinv(permutation_matrix.to(torch.float)).to(ffn_all_w1.dtype)  # N x 3N
    print(f"unmerge_matrix: {unmerge_matrix.shape} {unmerge_matrix}")

    print(f"original ffn w1: {ffn_all_w1.shape} {ffn_all_w1}")
    print(f"original ffn w2: {ffn_all_w2.shape} {ffn_all_w2}")
    print(f"original ffn w3: {ffn_all_w3.shape} {ffn_all_w3}")

    ffn_w1 = torch.zeros(d_model, d_ff, dtype=ffn_all_w1.dtype, device=_device)
    ffn_w3 = torch.zeros(d_model, d_ff, dtype=ffn_all_w3.dtype, device=_device)
    ffn_w2 = torch.zeros(d_model, d_ff, dtype=ffn_all_w2.dtype, device=_device)
    for i in range(num_ffn):
        ffn_w1 += torch.matmul(ffn_list[i].w1.weight.data.T, torch.matmul(permutation_matrix[i * d_ff:(i + 1) * d_ff], unmerge_matrix[:, i * d_ff:(i + 1) * d_ff]))
        ffn_w3 += torch.matmul(ffn_list[i].w3.weight.data.T, torch.matmul(permutation_matrix[i * d_ff:(i + 1) * d_ff], unmerge_matrix[:, i * d_ff:(i + 1) * d_ff]))
        ffn_w2 += torch.matmul(ffn_list[i].w2.weight.data, torch.matmul(permutation_matrix[i * d_ff:(i + 1) * d_ff], unmerge_matrix[:, i * d_ff:(i + 1) * d_ff]))

    print(f"unmerge ffn w1: {ffn_w1.shape} {torch.sum(torch.abs(ffn_all_w1 - ffn_w1.T))} {ffn_w1}")
    print(f"unmerge ffn w2: {ffn_w2.shape} {torch.sum(torch.abs(ffn_all_w2 - ffn_w2))} {ffn_w2}")
    print(f"unmerge ffn w3: {ffn_w3.shape} {torch.sum(torch.abs(ffn_all_w3 - ffn_w3.T))} {ffn_w3}")

    # handle.remove()
    del corr_matrix
    merged_ffn = deepcopy(ffn_list[0])
   
    merged_ffn.w1.weight.data = ffn_w1.T
    merged_ffn.w2.weight.data = ffn_w2
    merged_ffn.w3.weight.data = ffn_w3.T

    return merged_ffn

@torch.no_grad()
def _merge_deepseek_moe_by_activation_matching_within_and_across_models(
    ffn_list: List[DeepseekMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = None,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
    ingredient: Optional[str] = "act", # act, weight, act+weight
    mode: Optional[str] = "normal", # normal, cluster
) -> DeepseekMLP:
    print("merge: zipit-same-rule-without-unmerge")
    ffn_list = [ffn.eval() for ffn in ffn_list]
    concat_ffn = deepcopy(ffn_list[0])
    d_ff, d_model = ffn_list[0].w1.out_features, ffn_list[0].w1.in_features
    num_ffn = len(ffn_list)
    _device = concat_ffn.w1.weight.device
    _dtype = concat_ffn.w1.weight.dtype
    average_coefs = get_coef(num_ffn, input_weight, average_coefs, d_ff)
    forwarded_hidden_states = forwarded_hidden_states.to(_device)

    if mini_batch_size is None:
        mini_batch_size = forwarded_hidden_states.shape[0]

    
    ffn_all_w1 = torch.cat([ffn.w1.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_w2 = torch.cat([ffn.w2.weight.data for ffn in ffn_list], dim=1) # (d_model, d_ff * num_ffn)
    ffn_all_w3 = torch.cat([ffn.w3.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    concat_ffn.w1 = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.w2 = torch.nn.Linear(d_ff * num_ffn, d_model, bias=False)
    concat_ffn.w3 = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.w1.weight.data = ffn_all_w1
    concat_ffn.w2.weight.data = ffn_all_w2
    concat_ffn.w3.weight.data = ffn_all_w3
    
    if "act" in ingredient:
        handles = []
        activations = []
        def _activation_hook(module, input, output):
            activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))
            return _activation_hook
        handle = concat_ffn.w2.register_forward_hook(_activation_hook) 
        print(f"Collect activations with batch size {mini_batch_size} with original data length {forwarded_hidden_states.shape}")
        concat_ffn = concat_ffn.eval().to(forwarded_hidden_states.device)
        for i in range(0, forwarded_hidden_states.shape[0], mini_batch_size): # mini_batch_size = 10000
            concat_ffn(forwarded_hidden_states[i:i + mini_batch_size])  # mini_batch_size * 14336 -> activation: mini_batch_size * 32768 * num_ffn
        for handle in handles:
            handle.remove()
        del handles, forwarded_hidden_states
        activations = torch.cat(activations, dim=0)
        activations = activations.to("cuda:7")
        print(torch.cuda.memory_summary(device=7))
        print(f"Collected activations: {activations.shape} {activations.device}")
    if "weight" in ingredient:
        if "act" not in ingredient:
            del forwarded_hidden_states
        weights = []
        for ffn in ffn_list:
            concat_weight = torch.cat([ffn.w1.weight.data.T, ffn.w2.weight.data, ffn.w3.weight.data.T], dim=0) # 3DxN
            weights.append(concat_weight)
        weights = torch.cat(weights, dim=1) # 3Dx(N*num_ffn)
        weights = weights.to("cuda:7")

    if ingredient == "act":
        corr_matrix = compute_covariance(activations, activations)
    elif ingredient == "weight":
        corr_matrix = compute_covariance(weights, weights)
    else:
        corr_matrix = compute_covariance(activations, activations) + compute_covariance(weights, weights)
    torch.cuda.empty_cache()

    corr_matrix[torch.arange(d_ff * num_ffn), torch.arange(d_ff * num_ffn)] = -1  # Remove self-correlation
    print(f"corr_matrix: {corr_matrix.shape}")

    permutation_matrix = torch.eye(d_ff * num_ffn, d_ff * num_ffn, device=_device, dtype=ffn_all_w1.dtype)

    ### (1) Clusterining
    # Kmeans++
    if mode == "cluster":
        center_indices = [0]
        mask = torch.ones(corr_matrix.size(0), dtype=torch.bool, device=corr_matrix.device)  # Mask to exclude already selected centers
        mask[0] = False
        for _ in range(1, d_ff):
            # EN x num_centers
            dist = torch.min(corr_matrix[:, center_indices], dim=1).values ** 2
            probabilities = dist / dist.sum()
            probabilities = probabilities * mask.float()  # Set probabilities of selected centers to 0
            next_center_idx = torch.multinomial(probabilities, 1).item()
            center_indices.append(next_center_idx)
            mask[next_center_idx] = False
        center_indices.sort()
        print(len(center_indices))

        activations = activations.T
        centers = activations[center_indices]
        min_points_per_cluster = 1

        for _ in range(100):
            distance = torch.cdist(activations, centers)
            assignments = torch.argmin(distance, dim=1)
            del distance
            # Ensure each cluster has at least min_points_per_cluster points
            for i in range(d_ff):
                num_points_in_cluster = torch.sum(assignments == i)
                if num_points_in_cluster < min_points_per_cluster:
                    # Find overpopulated clusters
                    for j in range(d_ff):
                        if i != j and torch.sum(assignments == j) > num_ffn:
                            # Move points from overpopulated cluster j to underpopulated cluster i
                            diff = torch.sum(assignments == j) - min_points_per_cluster

                            # Select `num_to_move` points from cluster j and reassign them to cluster i
                            reassign_indices = torch.where(assignments == j)[0][0]
                            assignments[reassign_indices] = i
                            print(f"Group {i} has {num_points_in_cluster} points, move 1 point from group {j}")
                            break
                            
            # Recompute the centers after ensuring the minimum number of points
            group_members = []
            for i in range(d_ff):
                group_member = activations[assignments == i].mean(dim=0)
                if torch.isnan(group_member).sum().item() > 0:
                    print(f"Group {i}: {torch.nonzero(assignments == i).squeeze()} {group_member.shape} {torch.isnan(group_member).sum()}")
                group_members.append(group_member)
            new_centers = torch.stack(group_members)
            max_diff = 0
            for i in range(d_ff):
                diff = torch.max(torch.abs(centers[i] - new_centers[i]))
                max_diff = max(max_diff, diff.item())
            print(f"max_diff: {max_diff}")
            if max_diff < 1e-4:
                print("Converged!")
                break
            centers = new_centers
        
        # Assign the group index
        permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, dtype=torch.float16, device=_device)
        for i in range(d_ff):
            index_in_this_group = (assignments == i).nonzero().squeeze()
            permutation_matrix[i, index_in_this_group] = 1
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True)).to(_dtype)
        for i in range(5): # permutation_matrix.shape[1]
            print(permutation_matrix[:, i].nonzero().squeeze())
        permutation_matrix = permutation_matrix.to(_device)
        ffn_w1 = torch.matmul(permutation_matrix, ffn_all_w1)
        ffn_w2 = torch.matmul(permutation_matrix, ffn_all_w2.T)
        ffn_w3 = torch.matmul(permutation_matrix, ffn_all_w3)

        del ffn_all_w1, ffn_all_w2, ffn_all_w3
        merged_ffn = deepcopy(ffn_list[0])
        merged_ffn.w1.weight.data = ffn_w1
        merged_ffn.w2.weight.data = ffn_w2.T
        merged_ffn.w3.weight.data = ffn_w3
        return merged_ffn

    ### (2) Greedy Merging!
    while ffn_all_w1.shape[0] > d_ff:
        # Select the most correlated pair
        max_index = torch.argmax(corr_matrix)
        max_i, max_j = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]

        # Merge the most correlated pair, replace the first feature with the merged one
        i_coef, j_coef = average_coefs[max_i], average_coefs[max_j]
        ffn_all_w1[max_i] = (i_coef * ffn_all_w1[max_i] + j_coef * ffn_all_w1[max_j]) / (i_coef + j_coef + FP32_EPS)
        ffn_all_w3[max_i] = (i_coef * ffn_all_w3[max_i] + j_coef * ffn_all_w3[max_j]) / (i_coef + j_coef + FP32_EPS)
        ffn_all_w2[:, max_i] = (i_coef * ffn_all_w2[:, max_i] + j_coef * ffn_all_w2[:, max_j]) / (
                i_coef + j_coef + FP32_EPS)
        permutation_matrix[:, max_i] += permutation_matrix[:, max_j]
        permutation_matrix = remove_col(permutation_matrix, max_j)
       
        # Remove the second feature
        ffn_all_w1 = torch.cat([
            ffn_all_w1[:max_j],
            ffn_all_w1[max_j + 1:]
        ], dim=0)
        ffn_all_w3 = torch.cat([
            ffn_all_w3[:max_j],
            ffn_all_w3[max_j + 1:]
        ], dim=0)
        ffn_all_w2 = torch.cat([
            ffn_all_w2[:, :max_j],
            ffn_all_w2[:, max_j + 1:]
        ], dim=1)

        # Update the correlation matrix
        updated_corr_vec = alpha_for_repeated_merging * torch.min(
            torch.stack([corr_matrix[max_i], corr_matrix[max_j]]), dim=0
        ).values
        corr_matrix[max_i] = updated_corr_vec
        corr_matrix[:, max_i] = updated_corr_vec
        corr_matrix[max_i, max_i] = -1  # Remove self-correlation

        # Remove the second feature from the correlation matrix
        corr_matrix = torch.cat([
            corr_matrix[:, :max_j],
            corr_matrix[:, max_j + 1:]
        ], dim=1)
        corr_matrix = torch.cat([
            corr_matrix[:max_j],
            corr_matrix[max_j + 1:]
        ], dim=0)

        # Update the average coefs
        average_coefs[max_i] += average_coefs[max_j]
        average_coefs = average_coefs[:max_j] + average_coefs[max_j + 1:]
    permutation_matrix = permutation_matrix / torch.sum(permutation_matrix, dim=0, keepdim=True) # 3N x N
    for i in range(5): # permutation_matrix.shape[1]
        print(permutation_matrix[:, i].nonzero().squeeze())
    
    del corr_matrix
    merged_ffn = deepcopy(ffn_list[0])
   
    merged_ffn.w1.weight.data = ffn_all_w1
    merged_ffn.w2.weight.data = ffn_all_w2
    merged_ffn.w3.weight.data = ffn_all_w3

    return merged_ffn

@torch.no_grad()
def process_coef(num_ffn, d_ff, d_model, average_coefs=None, input_weight=None):
    if input_weight is not None:
        first_coef = []
        second_coef = []
        for w in input_weight:
            coef_1 = [w] * d_ff
            first_coef.extend(coef_1)
            coef_2 = [w] * d_model
            second_coef.extend(coef_2)
    elif average_coefs is None:
        first_coef = [1.0] * num_ffn * d_ff
        second_coef = [1.0] * num_ffn * d_model
    elif len(average_coefs) == num_ffn:
        first_coef = [coef for coef in average_coefs for _ in range(d_ff)]
        second_coef = [coef for coef in average_coefs for _ in range(d_model)]
    else:
        raise ValueError("The argument `avearge_coefs` should be either None or have the same length as `num_ffn`, or you need to provide `input_weight`.")
    return first_coef, second_coef


@torch.no_grad()
def compute_merging(temp_dim, target_dim, corr_matrix, coef, alpha, _device):
    permutation_matrix = torch.eye(temp_dim, temp_dim, dtype=torch.float, device=_device)
    while corr_matrix.shape[0] > target_dim:
        max_index = torch.argmax(corr_matrix)
        max_i, max_j = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]

        # Update permutation matrix
        i_coef, j_coef = coef[max_i], coef[max_j]
        permutation_matrix[:, max_i] = (i_coef * permutation_matrix[:, max_i] + j_coef * permutation_matrix[:, max_j]) / (i_coef + j_coef + FP32_EPS)
        permutation_matrix = remove_col(permutation_matrix, max_j)

        # Update corr_matrix
        updated_corr_vec = alpha * torch.min(torch.stack([corr_matrix[max_i], corr_matrix[max_j]]), dim=0).values
        corr_matrix[max_i] = updated_corr_vec
        corr_matrix[:, max_i] = updated_corr_vec
        corr_matrix[max_i, max_i] = -1
        # Remove second feature from the correlation matrix
        corr_matrix = remove_col(corr_matrix, max_j)
        corr_matrix = remove_row(corr_matrix, max_j)
    return permutation_matrix

@torch.no_grad()
def _merge_deepseek_moe_by_activation_matching_within_and_across_models_with_unmerge(
    ffn_list: List[DeepseekMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = 5000,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
) -> DeepseekMLP:
    print("merge: zipit-independe-rule-with-unmerge")
    ffn_list = [ffn.eval() for ffn in ffn_list]
    d_ff, d_model = ffn_list[0].w1.out_features, ffn_list[0].w1.in_features
    num_ffn = len(ffn_list)
    first_coef, second_coef = process_coef(num_ffn, d_ff, d_model, average_coefs, input_weight)
    
    _device = ffn_list[0].w1.weight.device
    _dtype = ffn_list[0].w1.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Collect activations with batch size {mini_batch_size} with original data length {forwarded_hidden_states.shape}")

    # Compute w1 and w3's permutation matrix
    ffn_all_w1 = torch.cat([ffn.w1.weight.data for ffn in ffn_list], dim=0) # 
    ffn_all_w3 = torch.cat([ffn.w3.weight.data for ffn in ffn_list], dim=0)
    act = torch.nn.SiLU()

    activations = []
    cur = act(torch.matmul(forwarded_hidden_states, ffn_all_w1.T)) * torch.matmul(forwarded_hidden_states, ffn_all_w3.T)
    activations.append(cur.reshape(-1, cur.shape[-1]))
    cat_activtaions = torch.cat(activations, dim=0)
    activations.clear()
    corr_matrix = compute_covariance(cat_activtaions, cat_activtaions)
    corr_matrix[torch.arange(d_ff * num_ffn), torch.arange(d_ff * num_ffn)] = -1  # Remove self-correlation
    print(f"corr_matrix: {corr_matrix.shape}")
    first_permutation_matrix = compute_merging(d_ff * num_ffn, d_ff, corr_matrix, first_coef, alpha_for_repeated_merging, _device)
    first_permutation_matrix = first_permutation_matrix / torch.sum(first_permutation_matrix, dim=0, keepdim=True)
    first_unmerge_matrix = torch.linalg.pinv(first_permutation_matrix)
    first_permutation_matrix = first_permutation_matrix.to(_dtype)
    ffn_w1 = torch.matmul(first_permutation_matrix.T, ffn_all_w1)
    ffn_w3 = torch.matmul(first_permutation_matrix.T, ffn_all_w3)
    print(f"first_permutation_matrix: {first_permutation_matrix.shape}, first_unmerge_matrix: {first_unmerge_matrix.shape}")
    
    # Compute w2's permutation matrix
    ffn_all_w2 = torch.cat([ffn.w2.weight.data for ffn in ffn_list], dim=0)
    new_data = act(torch.matmul(forwarded_hidden_states, ffn_w1.T)) * torch.matmul(forwarded_hidden_states, ffn_w3.T)
    activations = []
    new_cur = torch.matmul(new_data, ffn_all_w2.T)
    activations.append(new_cur.reshape(-1, new_cur.shape[-1]))
    cat_activtaions = torch.cat(activations, dim=0)
    activations.clear()
    corr_matrix = compute_covariance(cat_activtaions, cat_activtaions)
    corr_matrix[torch.arange(d_model * num_ffn), torch.arange(d_model * num_ffn)] = -1  # Remove self-correlation
    print(f"corr_matrix: {corr_matrix.shape}")
    second_permutation_matrix = compute_merging(d_model * num_ffn, d_model, corr_matrix, second_coef, alpha_for_repeated_merging, _device)
    second_permutation_matrix = second_permutation_matrix / torch.sum(second_permutation_matrix, dim=0, keepdim=True)
    second_unmerge_matrix = torch.linalg.pinv(second_permutation_matrix) # DxED
    print(f"second_permutation_matrix: {second_permutation_matrix.shape}, second_unmerge_matrix: {second_unmerge_matrix.shape}")
    second_permutation_matrix = second_permutation_matrix.to(_device).to(_dtype)
    first_unmerge_matrix = first_unmerge_matrix.to(_device).to(_dtype)
    ffn_w2 = torch.zeros(d_model, d_ff, device=_device)
    for i in range(num_ffn):
        ffn_w2 += torch.matmul(second_permutation_matrix.T[:, i*d_model:(i+1)*d_model], 
            torch.matmul(ffn_all_w2[i*d_model:(i+1)*d_model], first_unmerge_matrix[:, i*d_ff:(i+1)*d_ff]))
    
    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.w1.weight.data = ffn_w1
    merged_ffn.w2.weight.data = ffn_w2
    merged_ffn.w3.weight.data = ffn_w3
    
    # TODO: use a warpper to warp moe and assign unmerge matrix to it
    # TODO: consider (w1, w3) and (w2) has differnt unmerge matrix, use w2's unmerge matrix to unmerge w2's output
    return merged_ffn, second_unmerge_matrix


def prune_experts(
        moe: DeepseekMoE,
        dominant_experts: List[int],
):
    with torch.no_grad():
        r = len(dominant_experts)
        dominant_experts.sort()
        gate_new = torch.nn.Linear(in_features=moe.gate.in_features, out_features=r, bias=False, dtype=torch.bfloat16)
        gate_new.weight.data = moe.gate.weight.data[dominant_experts]
        moe.gate = gate_new

        moe.experts = torch.nn.ModuleList(
            [moe.experts[i] for i in dominant_experts])
        moe.num_experts = r
        moe.top_k = min(r, moe.top_k)
    return moe

@torch.no_grad()
def _merge_moe_experts_within_and_across_models(
        moe: DeepseekMoE,
        group_labels: torch.LongTensor,
        forwarded_hidden_states: Tuple[torch.Tensor],
        dominant_alone: bool,
        merge: Optional[str] = "zipit", # zipit, update, fix-dom, unmerge
        mode: Optional[str] = "normal", # normal, activation-with-router-logits, input-weight, all
        core_expert_indices: Optional[List[int]] = None,
        usage_frequencies: Optional[torch.Tensor] = None,
        moe_scores: Optional[torch.Tensor] = None,
        data_limit: Optional[int] = 50000,
        ingredient: Optional[str] = "act",
): # -> MixtralSparseMoeBlock:
    if merge == "weighted":
        merging_coefficient = [float(coef) for coef in mode]
        if len(merging_coefficient) != 2:
            raise ValueError("The argument `mode` should be in the format of `weight,weight`, but got {mode}.")
        merging_weight = torch.zeros(len(moe.experts))
        print(merging_coefficient, core_expert_indices)
        for e in range(len(moe.experts)):
            merging_weight[e] = merging_coefficient[0] if e in core_expert_indices else merging_coefficient[1]
        return _merge_mlp_experts_by_usage_frequency_weighting(
            ffn=moe,
            group_labels=group_labels,
            usage_frequencies=merging_weight,
        )
    elif merge == "prune" and mode == "normal":
        # Prune experts and its routing weights
        moe = prune_experts(moe, core_expert_indices)
        return moe

    input_weight = None
    if merge == "unmerge":
        moe = MoEWrapper(moe)
    else:
        new_moe = deepcopy(moe)
    print("core_expert_indices: ", core_expert_indices)
    # p = 0
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        print(f"\nGroup {label}: {expert_indices}")
        if core_expert_indices is not None:
            core_expert_index = [i for i, idx in enumerate(expert_indices) if idx in core_expert_indices]
        zipit_st = time.time()
        if dominant_alone:
            group_core_expert_indices = torch.stack([
                idx for idx in expert_indices if idx in core_expert_indices
            ])
            to_skip = False
            if len(group_core_expert_indices) == len(expert_indices):
                merged_expert = moe.experts[expert_indices[0]]
                to_skip = True
            elif usage_frequencies is not None and len(group_core_expert_indices) == 1:
                non_core_usage_sum = torch.sum(
                    usage_frequencies[[expert_idx.item() for expert_idx in
                                        expert_indices if expert_idx not in group_core_expert_indices]]).item()
                if non_core_usage_sum == 0:
                    merged_expert = moe.experts[group_core_expert_indices[0]]
                    to_skip = True
                else:
                    to_skip = False
            if not to_skip:
                # Stage 1: merge all experts except the dominant one
                group_forwarded_hidden_states = torch.cat([
                    forwarded_hidden_states[expert_idx] for expert_idx in expert_indices if
                    expert_idx not in group_core_expert_indices
                ], dim=0)
                if usage_frequencies is not None:
                    non_core_usages = usage_frequencies[[expert_idx.item() for expert_idx in expert_indices if
                                                            expert_idx not in group_core_expert_indices]]
                
                if mode == "update":
                    merged_expert = _merge_moe_experts_by_zipit(
                        ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                        forwarded_hidden_states=group_forwarded_hidden_states,
                        mini_batch_size=5000,
                        average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                        input_weight=input_weight,
                    )
                else:
                    merged_expert = _merge_deepseek_moe_by_activation_matching_within_and_across_models(
                        ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices if
                                    expert_idx not in group_core_expert_indices],
                        forwarded_hidden_states=group_forwarded_hidden_states,
                        average_coefs=non_core_usages.tolist() if usage_frequencies is not None else None,
                        ingredient=ingredient,
                    )
                # Stage 2: merge the dominant expert with the merged expert in stage 1
                group_forwarded_hidden_states = torch.cat([
                    forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
                ], dim=0)
                if usage_frequencies is not None:
                    core_usages = usage_frequencies[group_core_expert_indices]
                    non_core_usage_sum = torch.sum(non_core_usages).item()
                merged_expert = _merge_deepseek_moe_by_activation_matching_within_and_across_models(
                    ffn_list=[merged_expert] + [moe.experts[expert_idx] for expert_idx in
                                                group_core_expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    average_coefs=[non_core_usage_sum] + core_usages.tolist(
                    ) if usage_frequencies is not None else None
                )
        else:
            # not dominant
            if mode == "input-weight" or mode == "all":
                input_weight = []
                for expert_idx in expert_indices:
                    input_weight.append(forwarded_hidden_states[expert_idx].shape[0])
                s = sum(input_weight)
                input_weight = [w / s for w in input_weight]
                print("input_weight: ", input_weight)
            
            group_forwarded_hidden_states = torch.cat([
                forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
            ], dim=0)
            randperm_indices = torch.randperm(group_forwarded_hidden_states.shape[0])
            group_forwarded_hidden_states = group_forwarded_hidden_states[randperm_indices[:data_limit]]
            if expert_indices.shape[0] == 1:
                if merge == "unmerge":
                    merged_expert = moe.model.experts[expert_indices[0]]
                    moe.unmerge_matrix[label.item()] = None
                else:
                    merged_expert = moe.experts[expert_indices[0]]
            else:
                if merge == "update":
                    merged_expert = _merge_moe_experts_by_zipit(
                        ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                        forwarded_hidden_states=group_forwarded_hidden_states,
                        mini_batch_size=5000,
                        average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                        input_weight=input_weight,
                    )
                elif merge == "fix-dom":
                    merged_expert = _merge_moe_experts_with_dominant(
                        ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                        forwarded_hidden_states=group_forwarded_hidden_states,
                        mini_batch_size=5000,
                        average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                        input_weight=input_weight,
                        dominant_index=core_expert_index[0],
                    )
                elif merge == "fix-dom-same":
                    merged_expert = _merge_moe_experts_with_dominant_same_rule(
                        ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                        forwarded_hidden_states=group_forwarded_hidden_states,
                        input_weight=input_weight,
                        dominant_index=core_expert_index[0],
                        ingredient=ingredient,
                        mode=mode,
                    )
                elif merge == "unmerge":
                    merged_expert, unmerge_matrix = _merge_deepseek_moe_by_activation_matching_within_and_across_models_with_unmerge(
                        ffn_list=[moe.model.experts[expert_idx] for expert_idx in expert_indices],
                        forwarded_hidden_states=group_forwarded_hidden_states,
                        mini_batch_size=5000,
                        average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                        input_weight=input_weight,
                    )
                    moe.unmerge_matrix[label.item()] = unmerge_matrix.to(moe.model.experts[0].w1.weight.device).to(torch.bfloat16)
                elif merge == "prune":
                    pass
                else: # zipit-normal, activation-with-router-logits, input-weight
                    if mode == "unmerge":
                        merged_expert = _merge_deepseek_moe_by_activation_matching_within_and_across_models_same_rule_with_unmerge(
                            ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                            forwarded_hidden_states=group_forwarded_hidden_states,
                            mini_batch_size=5000,
                            average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                            input_weight=input_weight,
                            ingredient=ingredient,
                        )
                    else:
                        merged_expert = _merge_deepseek_moe_by_activation_matching_within_and_across_models(
                            ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                            forwarded_hidden_states=group_forwarded_hidden_states,
                            mini_batch_size=5000,
                            average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                            input_weight=input_weight,
                            ingredient=ingredient,
                            mode=mode,
                        )
        
        
        if merge == "unmerge":
            moe.model.experts[expert_indices[0].item()].w1.weight.copy_(merged_expert.w1.weight)
            moe.model.experts[expert_indices[0].item()].w2.weight.copy_(merged_expert.w2.weight)
            moe.model.experts[expert_indices[0].item()].w3.weight.copy_(merged_expert.w3.weight)
            moe.expert_to_group[expert_indices[0].item()] = label.item()
            moe.group_to_expert[label.item()] = [expert_indices[0].item()]
            for expert_idx in expert_indices[1:]:
                moe.model.experts[expert_idx.item()] = moe.model.experts[expert_indices[0].item()]
                moe.expert_to_group[expert_idx.item()] = label.item()
                moe.group_to_expert[label.item()].append(expert_idx.item())
            moe.group_to_expert[label.item()] = torch.tensor(moe.group_to_expert[label.item()])
        elif merge == "prune" and mode == "zero-output":
            for expert_idx in expert_indices:
                if expert_idx == core_expert_indices[0]:
                    continue
                new_moe.experts[expert_idx.item()].w2.weight.copy_(torch.zeros_like(new_moe.experts[expert_idx.item()].w2.weight))
        elif expert_indices.shape[0] != 1:
            new_moe.experts[expert_indices[0].item()].w1.weight.copy_(merged_expert.w1.weight)
            new_moe.experts[expert_indices[0].item()].w2.weight.copy_(merged_expert.w2.weight)
            new_moe.experts[expert_indices[0].item()].w3.weight.copy_(merged_expert.w3.weight)
            # moe.expert_dict[expert_indices[0].item()] = expert_indices[0].item()
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                new_moe.experts[expert_idx.item()] = new_moe.experts[expert_indices[0].item()]
                # moe.expert_dict[expert_idx.item()] = expert_indices[0].item()
                # moe.experts[expert_idx.item()] = None

        group_forwarded_hidden_states = group_forwarded_hidden_states.to('cpu')
        if merge != "prune":
            merged_expert = merged_expert.to('cpu')
            del merged_expert
        del group_forwarded_hidden_states
        torch.cuda.empty_cache()
        if len(expert_indices) != 1:
            print("After merging (in _merge_moe_experts_within_and_across_models): ")
            # print(torch.cuda.memory_summary())
        print(f"Merging takes {time.time() - zipit_st:.2f}s")
    if merge == "unmerge":
        print("Expert to Group: ", moe.expert_to_group)
        print("Group to Expert: ", moe.group_to_expert)
        print("Unmerge matrix: ", moe.unmerge_matrix)
    # print(moe.expert_dict)
    # moe.forward = MethodType(merged_moe_forward, moe)
    return new_moe


@torch.no_grad()
def merge_by_groups_with_usage_weighted(
        model: DeepseekForCausalLM,
        grouper: ExpertsGrouperForDeepseek,
        merging_layers: Optional[List[int]] = None,
        weight: Optional[torch.Tensor] = None
) -> DeepseekForCausalLM:
    """
    Parameters
    ----------
    model: DeepseekForCausalLM
        The model to merge experts.
    grouper: ExpertsGrouperForDeepseek
        The grouper to group experts, supposed to have been called `grouper.compute_all_usages()` and
            one of `grouper.group_experts()` (i.e. have grouped labels).
    merging_layers: Optional[List[int]]
        The layers where we merge experts, if None, merge all layers.
    """
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    group_labels_dict = grouper.group_state_dict()

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[HC-SMoE] Merging experts with usage-frequency-weighted averaging..."
    ):
        if merging_layers is not None and layer_idx not in merging_layers:
            continue
        ffn_name = f"model.layers.{layer_idx}.mlp"
        group_labels = group_labels_dict[ffn_name]
        usage_frequencies = usage_frequency_dict[ffn_name]
        # usage_frequencies = torch.ones(len(usage_frequencies), dtype=usage_frequencies.dtype, device=usage_frequencies.device)
        model.model.layers[layer_idx].mlp = _merge_mlp_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].mlp,
            group_labels=group_labels,
            usage_frequencies=usage_frequencies if weight is None else weight,
        )
    return model

@torch.no_grad()
def merge_by_groups_within_and_across_models(
    deepseek_model: DeepseekForCausalLM,
    grouper: ExpertsGrouperForDeepseek,
    dataloader: DataLoader,
    merge: Optional[str] = "zipit", # zipit, update, fix-dom, unmerge, kl-weight
    mode: Optional[str] = "normal", # normal, activation-with-router-logits, input-weight, all
    partition: Optional[int] = 1,
    dominant_alone: Optional[bool] = False,
    core_experts: Optional[Dict[str, List[int]]] = None,
    ingredient: Optional[str] = "act",
) -> DeepseekForCausalLM:

    forwarded_hidden_states = dict()
    print(forwarded_hidden_states)

    usage_frequencies = grouper.usage_frequency_state_dict()
    num_experts = grouper.num_experts

    def _get_activation_hook(name):
        #TODO: check if the length is outofbound
        def hook(_, input, __):
            forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1])) # .cpu()
        return hook
    
    # Due to OOM, We can devide it into 2 parts
    def part_processor(sparse_layer_indices):
        deepseek_model.eval() # .cuda()
        handles = []
        for layer_idx in tqdm(
                sparse_layer_indices,
                desc=f"[Merging] Registering forward hook..."
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            handles.append(deepseek_model.model.layers[layer_idx].mlp.register_forward_hook(
                _get_activation_hook(ffn_name))
            )
        router_indices = {name: [] for name in forwarded_hidden_states.keys()}
        if mode == "activation-with-router-logits" or mode == "all":
            router_weights = {name: [] for name in forwarded_hidden_states.keys()}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = deepseek_model(**batch, output_router_logits=True)
                for layer_idx in sparse_layer_indices:
                    ffn_name = f"model.layers.{layer_idx}.mlp"
                    routing_weights = F.softmax(outputs.router_logits[layer_idx], dim=1)
                    routing_weights, selected_experts = torch.topk(routing_weights, deepseek_model.config.num_experts_per_tok, dim=-1)
                    router_indices[ffn_name].append(selected_experts)
                    if mode == "activation-with-router-logits" or mode == "all":
                        router_weights[ffn_name].append(routing_weights)
                del outputs
                        
        for handle in handles:
            handle.remove()

        for layer_idx in tqdm(
                sparse_layer_indices,
                desc=f"[Merging]Merging by groups within and across experts..."
        ):
            _st = time.time()
            ffn_name = f"model.layers.{layer_idx}.mlp"
            group_labels = grouper.group_state_dict()[ffn_name]
            layer_forwarded_hidden_states = tuple()
            hidden_states = torch.cat(forwarded_hidden_states[ffn_name], dim=0) # T x D
            concat_router_indices = torch.cat(router_indices[ffn_name], dim=0) # BT x k
            if mode == "activation-with-router-logits" or mode == "all":
                concat_router_weights = torch.cat(router_weights[ffn_name], dim=0) # BT x k
            for expert_idx in range(num_experts): # expert num
                expert_mask = (concat_router_indices == expert_idx)
                batch_tensor = torch.any(expert_mask, dim=-1).to(hidden_states.device)
                choice_input = hidden_states[batch_tensor]
                if mode == "activation-with-router-logits" or mode == "all":
                    router_weight = torch.masked_select(concat_router_weights, expert_mask).view(-1, 1).to(choice_input.device)
                    layer_hidden_states = choice_input * router_weight
                else:
                    layer_hidden_states = choice_input
                layer_forwarded_hidden_states += (layer_hidden_states,)
            deepseek_model.model.layers[layer_idx].mlp = _merge_moe_experts_within_and_across_models(
                moe=deepseek_model.model.layers[layer_idx].mlp,
                group_labels=group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                dominant_alone=dominant_alone,
                merge=merge,
                mode=mode,
                core_expert_indices=core_experts[ffn_name] if core_experts is not None else None,
                usage_frequencies=None, # usage_frequencies[ffn_name] if usage_weighted else None,
                data_limit=grouper.data_limit,
                ingredient=ingredient,
            )
            for hidden_states in layer_forwarded_hidden_states:
                hidden_states = hidden_states.to('cpu')
                del hidden_states
            del layer_forwarded_hidden_states
            print(f"------- Layer {layer_idx} took {time.time() - _st:.2f}s -------\n")

    
    print(grouper.sparse_layer_indices)
    partition_num = len(grouper.sparse_layer_indices) // partition
    for i in range(0, len(grouper.sparse_layer_indices), partition_num):
        cur_indices = grouper.sparse_layer_indices[i:i+partition_num]
        print("cur: ", cur_indices)
        part_processor(cur_indices)
    return deepseek_model



@torch.no_grad()
def check(
    deepseek_model,
    dataloader,
    file_name
):
    teacher_output = dict()

    def _get_moe_output_hook(name):
        def hook(module, input, output):
            teacher_output[name].append(output[0].detach().cpu().reshape(-1, output[0].shape[-1]))
        return hook

    deepseek_model.eval()
    deepseek_model.requires_grad_(False)
    handles = []
    teacher_output["21"] = []
    handles.append(deepseek_model.model.layers[21].mlp.register_forward_hook(
        _get_moe_output_hook("21")
    ))
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = deepseek_model(**batch)
    print(len(teacher_output["21"]), teacher_output["21"][0].shape)
    print(teacher_output["21"])
    # with open(f"{file_name}.pkl", "wb") as f:
    #     pickle.dump(teacher_output, f)
