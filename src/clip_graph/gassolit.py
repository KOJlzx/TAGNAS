from typing import Dict, List, Tuple, Any, Callable, Optional, Union

import os
import io
import functools as ft
import collections as cl
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers as tf

from torch_geometric import nn as gnn

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl

from . import losses as ls
from . import optimizers as op
from . import gassomodels as md
from . import utils as ut

from .lit import LitBase, LitPretrainLM
#
# Base class
#


@torch.no_grad()
def set_dropout(module, p=0):  # NOTE this is an awful hack
    names = ['dropout', 'p_dropout', 'dropout_p', 'prob_dropout',
             'dropout_prob']

    for name in names:
        if hasattr(module, name):
            att = getattr(module, name)

            if isinstance(att, float):
                setattr(module, name, p)
            elif isinstance(att, nn.Dropout):
                att.p = p
            else:
                raise ValueError("Module has unsupported configuration")


@torch.no_grad()
def set_layernorm_eps(module, eps=1e-4):  # this is also an awful hack
    if isinstance(module, nn.LayerNorm):
        module.eps = eps



#
# Graph models
#


LitGAEOutput = cl.namedtuple('LitGAEOutput', ['z'])


class LitGAE(LitBase):
    # wrap the input model for pretraining
    autoencoder_class = gnn.GAE

    def __init__(self,
        node_features_keys: List[str] = ['graph_x'],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model = self.autoencoder_class(encoder=self.model)
        self.node_features_keys = node_features_keys
        self.epoch = 0
        self.automatic_optimization = False
        # Example = cl.namedtuple('Example', ['x', 'edge_index'])
        # self.example_input_array = Example(
        #     # 200 (fake) nodes, 768 node data dims, 1000 edges
        #     torch.randn(200, 768),
        #     torch.randint(0, 200, (2, 1000)),
        # )

        self.save_hyperparameters(ignore=['model'])

    def forward(self,
        x: torch.Tensor,
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor,
        is_search: bool = True
    ) -> torch.Tensor:

        # if(self.epoch > 20):
        #     is_search = False
        # print("epoch", self.epoch)
        # print("is_search", is_search)
        out = self.model.encoder(x, edge_index, edge_weight, is_search)
        self.epoch += 1
        # print("step", self.model.encoder.step)
        # print("f" * 100)
        # if (self.model.encoder.step <= 2):
        #     return out
        if isinstance(out, torch.Tensor):
            pass
        elif isinstance(out, cl.abc.Mapping) and out['output'] is not None:
            out = out['output']
        elif isinstance(out, cl.abc.Mapping):
            out = out['last_hidden_state']
        else:
            raise ValueError('Bad output from encoder: ' + str(out))

        return LitGAEOutput(out)

    def loss(self,
        batch: Dict[str, Any],
        outputs: LitGAEOutput,
    ) -> torch.Tensor:
        # if(self.model.encoder.step <= 2):
        #     return None
        return self.model.recon_loss(
            outputs.z,
            batch['graph_edge_index'],
            batch['graph_neg_edge_index'],
        )

    def step(self,
        batch: Dict[str, Any],
        batch_idx: int,
        split: str
    ) -> torch.Tensor:
        x = torch.cat([batch[k] for k in self.node_features_keys], dim=1)
        x = x.float()
        opt_weights, opt_arch = self.optimizers()

        if self.current_epoch == 0 and self.model.encoder.step == 0:
            edge_index = batch['graph_edge_index'].unsqueeze(0).repeat(self.model.encoder.num_layers, 1, 1)
            edge_weight = batch['graph_edge_attr'].unsqueeze(0).repeat(self.model.encoder.num_layers, 1, 1)
            self.model.encoder.adjs = (edge_index, edge_weight)
            self.model.encoder.step = 1
        outputs = self(x, batch['graph_edge_index'], batch['graph_edge_attr'])
        # print("___", split)
        # if(self.model.encoder.step <= 2):
        #     return None
        loss = self.loss(batch, outputs)

        if split == 'train' and self.current_epoch > 0:
        # Get the optimizers
            opt_weights, opt_arch = self.optimizers()

            # Update according to the current alternating training phase
            if self.current_phase == 'train_weights':
                opt_weights.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(opt_weights, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt_weights.step()
                
            elif self.current_phase == 'train_arch':
                opt_arch.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(opt_arch, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt_arch.step()

        auc, ap = self.model.test(
            outputs.z.float(),
            batch['graph_edge_index'],
            batch['graph_neg_edge_index'],
        )
        # AUC is the probability that a positive-sample prediction score is greater
        # than a negative-sample prediction score; for a graph this is
        # sum(em(a) @ em(b) > em(c) @ em(d), where edge ab exists and cd does not) / (num_pos_edges * num_neg_edges)
        # AP (Average Precision)
        # Take positive edges (edges that actually exist) and negative edges (randomly sampled non-existent edges);
        # the model computes a score for each edge (e.g. the dot product of the two node embeddings).
        # Similar to AUC, both are areas under a curve, but AP focuses more on false positives; both metrics are used for link prediction.
        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True, batch_size=1,
                 rank_zero_only=True)

        self.log(f'{split}_auc', auc, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True, batch_size=1,
                 rank_zero_only=True)

        self.log(f'{split}_ap', ap, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True, batch_size=1,
                 rank_zero_only=True)

        return loss

    def on_train_epoch_start(self):
        """Before each epoch starts, decide what to train this round"""
        current_epoch = self.current_epoch
        # for name, param in self.model.encoder.named_parameters():
        #     # param.shape is a torch.Size object, can be printed directly
        #     print(f"{name:<50} | {param.shape}")
        # if self.current_epoch == 0:
            # edge_index = batch['graph_edge_index'].unsqueeze(0).repeat(self.model.encoder.num_layers, 1, 1)
            # edge_weight = batch['graph_edge_attr'].unsqueeze(0).repeat(self.model.encoder.num_layers, 1, 1)
            # self.model.encoder.adjs = (edge_index, edge_weight)
            # self.model.encoder.step = 1
        # Alternating rule: even epochs train GNN, odd epochs train Alpha
        if current_epoch % 2 == 0:
            self.current_phase = 'train_weights'
            self.model.encoder.set_alternating_phase('train_weights')
            print(f"[Epoch {current_epoch}] Phase: Train GNN Weights (Alphas Frozen)")
        else:
            self.current_phase = 'train_arch'
            self.model.encoder.set_alternating_phase('train_arch')
            print(f"[Epoch {current_epoch}] Phase: Train Alphas (GNN Weights Frozen)")

    def configure_optimizers(self):
        # Define two optimizers
        # 1. Weight optimizer (usually a higher learning rate, e.g. 0.01)

        # print("x" * 100)
        # print(self.model.encoder.weight_parameters())
        # print("x" * 100)
        # print(self.model.encoder.arch_parameters())
        # print("x" * 100)
        opt_weights = torch.optim.AdamW(
            self.model.encoder.weight_parameters(), lr=0.01, weight_decay=0.0
        )
        
        # 2. Architecture optimizer (usually a lower learning rate, e.g. 3e-4)
        opt_arch = torch.optim.AdamW(
            self.model.encoder.arch_parameters(), lr=0.01, weight_decay=0.0
        )
        
        return [opt_weights, opt_arch]


class LitClipGraphGNNTrain(LitGAE):
    def __init__(self, ckpt_path: str, **kwargs: Any) -> None:
        assert 'model' not in kwargs.keys() or kwargs['model'] is None

        mod = LitClipGraph.load_from_checkpoint(ckpt_path).model.gnn
        kwargs['model'] = mod

        super().__init__(**kwargs)


LitVGAEOutput = cl.namedtuple('LitGAEOutput', ['z', 'mu', 'logstd'])


class LitVGAE(LitGAE):
    autoencoder_class = gnn.VGAE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # need to divide these into mu and logstd
        assert self.model.encoder.out_channels % 2 == 0

        self.save_hyperparameters(ignore=['model'])

    def forward(self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        encoded = super().forward(x, edge_index).z

        latent_dim = int(self.model.encoder.out_channels / 2)
        mu, logstd = encoded[:, :latent_dim], encoded[:, latent_dim:]
        logstd = logstd.clamp(max=10)

        return LitVGAEOutput(self.model.reparametrize(mu, logstd), mu, logstd)

    def loss(self,
        batch: Dict[str, Any],
        outputs: LitVGAEOutput,
    ) -> torch.Tensor:
        recon_loss = super().loss(batch, outputs)

        B = outputs.z.shape[0]
        variational_loss = 1/B * self.model.kl_loss(
            outputs.mu,
            outputs.logstd
        )

        return recon_loss + variational_loss




class LitClipGraphLMTrain(LitPretrainLM):
    def __init__(self,
        ckpt_path: Optional[str] = None,
        restart_ckpt_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        assert ckpt_path is None or restart_ckpt_path is None
        assert ckpt_path is not None or restart_ckpt_path is not None
        assert 'model' not in kwargs.keys() or kwargs['model'] is None

        if ckpt_path is not None:
            mod = LitClipGraph.load_from_checkpoint(ckpt_path).model.lm.model
            mod2 = tf.AutoModelForCausalLM.from_config(mod.config)
            mod2.transformer.load_state_dict(mod.state_dict())
            kwargs['model'] = mod2
        else:  # restart_ckpt_path is not None:
            mod = LitClipGraphLMTrain.load_from_checkpoint(restart_ckpt_path).model
            kwargs['model'] = mod

        super().__init__(**kwargs)

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

#
# CLIP-graph model
#


class LitClipGraph(LitBase):
    def __init__(self,
        lm_name: Optional[str] = None,
        lm_ckpt_path: Optional[str] = None,
        lm_pooling_mode: str = 'mean',
        lm_normalize: bool = False,

        gnn_class_name: Optional[str] = None,
        gnn_params: Optional[Dict[str, Any]] = None,
        gnn_ckpt_path: Optional[str] = None,
        gnn_ckpt_dropout: Optional[float] = None,
        gnn_node_features_keys: List[str] = ['graph_x'],

        sim_smoothing: float = 0.0,
        sim_weights: Optional[str] = None,

        embed_dim: int = 768,
        dropout: float = 0.0,
        tau_init: float = 0.07,
        max_tau: Optional[float] = None,
        lm_layernorm_eps: Optional[float] = None,
        cycle_length_steps: Optional[int] = None,

        debug_checks: bool = False,

        **kwargs: Any
    ) -> None:
        if ut.count(lm_name, lm_ckpt_path) != 1:
            raise ValueError('Must provide exactly one of lm_name and '
                             'lm_ckpt_file')

        if ut.count(gnn_class_name, gnn_ckpt_path) != 1:
            raise ValueError('Must provide exactly one of gnn_class_name and '
                             'gnn_ckpt_file')

        if gnn_params is not None and gnn_class_name is None:
            raise ValueError('Can only specify gnn_params with gnn_class_name')

        if lm_ckpt_path is not None:
            lm = LitPretrainLM.load_from_checkpoint(lm_ckpt_path) \
                       .model \
                       .to_sentence_embedding(
                           pooling_mode=lm_pooling_mode,
                           normalize=lm_normalize
                       )
        else:
            lm = md.LMForSentenceEmbedding(
                model=lm_name,
                pooling_mode=lm_pooling_mode,
                normalize=lm_normalize
            )

        if gnn_ckpt_path is not None:
            gnn = LitGAE.load_from_checkpoint(gnn_ckpt_path)

            # throw away the LightningModule and the (V)GAE - we just want the
            # node embeddings
            gnn = gnn.model.encoder.model

            if gnn_ckpt_dropout is not None:
                gnn.apply(lambda m: set_dropout(m, p=gnn_ckpt_dropout))
        else:
            gnn_params = {} if gnn_params is None else gnn_params
            gnn = getattr(md, gnn_class_name)(**gnn_params)

        kwargs['model'] = md.ClipGraph(
            lm = lm,
            gnn = gnn,
            embed_dim = embed_dim,
            dropout = dropout,
            tau_init = tau_init,
            max_tau = max_tau,

            debug_checks=debug_checks
        )

        super().__init__(**kwargs)

        self.gnn_node_features_keys = gnn_node_features_keys
        self.lm_pooling_mode = lm_pooling_mode
        self.lm_normalize = lm_normalize
        self.sim_smoothing = sim_smoothing
        self.sim_weights = sim_weights
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.tau_init = tau_init
        self.lm_layernorm_eps = lm_layernorm_eps
        self.cycle_length_steps = cycle_length_steps
        self.debug_checks = debug_checks
        self.train_step = 0
        self.automatic_optimization = False
        sim_weight_options = ['identity', 'exp', 'exp_thick_tail']
        if self.sim_weights is not None and self.sim_weights not in sim_weight_options:
            raise ValueError(f'Invalid sim_weights option: {self.sim_weights}')

        if self.lm_layernorm_eps:
            fn = ft.partial(set_layernorm_eps, eps=self.lm_layernorm_eps)
            self.model.lm.apply(fn)

        # Example = cl.namedtuple('Example', ['input_ids', 'attention_mask',
        #                                     'graph_x', 'graph_edge_index',
        #                                     'graph_node_ids', 'text_node_ids'])
        # self.example_input_array = Example(
        #     # [
        #     #     'This is a sentence.',
        #     #     'Look, here is another sentence.',
        #     #     'And finally here is a third sentence.',
        #     # ],

        #     torch.LongTensor([
        #         [   0, 2027, 2007, 1041, 6255, 1016,    2,    1,    1,    1],
        #         [   0, 2302, 1014, 2186, 2007, 2182, 6255, 1016,    2,    1],
        #         [   0, 2002, 2637, 2186, 2007, 1041, 2357, 6255, 1016,    2]
        #     ]),

        #     torch.LongTensor([
        #         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #     ]),

        #     # 200 (fake) nodes, 768 node data dims, 1000 edges
        #     torch.randn(200, 768) if self.model.gnn.in_channels is not None \
        #     else torch.randint(0, 200, (200,)),

        #     torch.randint(0, 200, (2, 1000)),

        #     # (fake) node IDs which are just sequential ints, and 3 randomly
        #     # chosen ones for the texts to have come from
        #     torch.arange(0, 200),
        #     torch.randint(0, 200, (3,)),
        # )

        self.save_hyperparameters(ignore=['model'])

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_edge_attr: torch.Tensor,
        graph_node_ids: torch.Tensor,
        text_node_ids: torch.Tensor,
        is_search: bool = True,
    ) -> torch.Tensor:
        node_index = (graph_node_ids == text_node_ids[:, None])
        node_index = node_index.nonzero()[:, 1]
        if self.train_step > 10:
            is_search = False
            print("in not search phase")
        # is_search = False
        return self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            graph_x = graph_x,
            graph_edge_index = graph_edge_index,
            graph_edge_attr = graph_edge_attr,
            node_index = node_index, 
            is_search = is_search
        )


        
    def step(self,
        batch: Dict[str, Any],
        batch_idx: int,
        split: str
    ) -> torch.Tensor:
        if self.debug_checks:
            # Sanity check that there aren't duplicates
            n_nodes = torch.unique(batch['graph_node_ids']).shape[0]
            n_texts = len(set(batch['text']))

            assert batch['graph_node_ids'].shape[0] == n_nodes
            assert len(batch['text']) == n_texts

        graph_x = torch.cat([
            batch[k]
            for k in self.gnn_node_features_keys
        ], dim=1)

        # NOTE sometimes you need a .half() here under mixed precision for
        # reasons I don't understand - data issue?
        # graph_x = graph_x.float()  # or .half()

        # print(batch['input_ids'].shape, batch['attention_mask'].shape, graph_x.shape, batch['graph_edge_index'].shape, batch['graph_node_ids'].shape, batch['text_node_ids'].shape) 
        #torch.Size([4, 123]) torch.Size([4, 123]) torch.Size([713, 768]) torch.Size([2, 1302]) torch.Size([713]) torch.Size([4])
        if self.current_epoch == 0 and self.model.gnn.step == 0:
            edge_index = batch['graph_edge_index'].unsqueeze(0).repeat(self.model.gnn.num_layers, 1, 1)
            edge_weight = batch['graph_edge_attr'].unsqueeze(0).repeat(self.model.gnn.num_layers, 1, 1)
            self.model.gnn.adjs = (edge_index, edge_weight)
            self.model.gnn.step = 1


        logits = self(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            graph_x = graph_x,
            graph_edge_index = batch['graph_edge_index'],
            graph_edge_attr = batch['graph_edge_attr'],
            graph_node_ids = batch['graph_node_ids'],
            text_node_ids = batch['text_node_ids'],
        )
        # if(self.model.gnn.step <= 2):
        #     zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        #     return zero_loss

        # print(batch['text_node_ids'], batch['graph_node_ids']) 
        #([17145742, 12651051,  9787347, 12513038], device='cuda:0')
        # print(logits.shape)
        #[4, 4]
        # {
        #     # Text data (already batched)
        #     'text_node_ids': tensor([101, 102, 103]),
        #     'input_ids': tensor([...]),  # shape: [3, max_len]
        #     'attention_mask': tensor([...]),  # shape: [3, max_len]
        #     'labels': tensor([...]),
        #     'text': ['This paper discusses AI...', 'Machine learning methods...', 'Deep neural networks...'],

        #     # Graph data (shared, stored only once)
        #     'graph_edge_index': tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),     # shape: [3, 2]
        #     'graph_x': tensor([[0, 1, 2], [1, 2, 0]]),          # shape: [2, 3]
        #     'graph_node_ids': tensor([101, 102, 103])                    # shape: [3]
        # }
        if self.sim_smoothing > 0:
            inds = (batch['text_node_ids'][:, None] == batch['graph_node_ids'])
            inds = inds.nonzero()[:, 1]

            sims = batch['graph_sim_mutual'][inds, ...][..., inds]
            sims = torch.clone(sims.detach())

            # distribution over *other* nodes; this is equivalent to
            # sims.fill_diagonal_(0) but is out-of-place so deepspeed doesn't
            # barf when we try to train
            sims = sims - torch.diagflat(torch.diagonal(sims))

            loss = self.criterion(
                logits = logits,
                sims = sims,
                sim_weights = self.sim_weights,
                alpha = self.sim_smoothing
            )
        else:
            loss = self.criterion(logits)

        # if split == 'train' and self.model.gnn.step > 0:
        #     # opt_weights, opt_arch, opt_lm_weights = self.optimizers()
        #     opt_weights, opt_arch, opt = self.optimizers()
            
            
        #     # current_epoch = self.current_epoch
        #     # # Update according to the current alternating training phase
        #     # if current_epoch % 2 == 0:
        #     #     opt_weights.zero_grad()
        #     #     self.manual_backward(loss)
        #     #     self.clip_gradients(opt_weights, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        #     #     opt_weights.step()    
        #     # else:
        #     #     opt_arch.zero_grad()
        #     #     self.manual_backward(loss)
        #     #     self.clip_gradients(opt_arch, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        #     #     opt_arch.step()

        #     opt.zero_grad()
        #     self.manual_backward(loss)
        #     self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        #     opt.step()
            
            # opt_weights.zero_grad()
            # opt_arch.zero_grad()
            # opt_lm_weights.zero_grad()
            # # 2. [Core] Only do a single backward pass
            # # This automatically computes gradients for all parameters that need
            # # updating in both GNN and LM

        #     # 3. Clip gradients separately and update parameters

        #     # Update GNN
        #     # if self.current_phase == 'train_weights':
        #     #     self.clip_gradients(opt_weights, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        #     #     opt_weights.step()
        #     # elif self.current_phase == 'train_arch':
        #     #     self.clip_gradients(opt_arch, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        #     #     opt_arch.step()

            # self.clip_gradients(opt_weights, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            # opt_weights.step()
            # self.clip_gradients(opt_arch, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            # opt_arch.step()
            # # Update LM
            # self.clip_gradients(opt_lm_weights, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            # opt_lm_weights.step()

            # print(list(self.model.lm.model.parameters()))
            # print("x" * 100)
            
        if self.debug_checks:
            assert not torch.isinf(loss).any().item()
            assert not torch.isnan(loss).any().item()

        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True,
                 batch_size=batch['text_node_ids'].shape[0],
                 rank_zero_only=True)

        if split == 'train':
            # tau: the logits are multiplied by tau.exp(); initialized as tau_init
            self.log(f'{split}_tau', self.model.tau, on_step=True,
                    on_epoch=True, logger=True, sync_dist=True,
                    batch_size=batch['text_node_ids'].shape[0],
                    rank_zero_only=True)

        return loss

    @property
    def criterion(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return ls.square_contrastive_loss


    def on_train_epoch_start(self):
        """Before each epoch starts, decide what to train this round"""
        current_epoch = self.current_epoch
        
        # if self.current_epoch == 0:
            # edge_index = batch['graph_edge_index'].unsqueeze(0).repeat(self.model.encoder.num_layers, 1, 1)
            # edge_weight = batch['graph_edge_attr'].unsqueeze(0).repeat(self.model.encoder.num_layers, 1, 1)
            # self.model.encoder.adjs = (edge_index, edge_weight)
            # self.model.encoder.step = 1
        # Alternating rule: even epochs train GNN, odd epochs train Alpha
        # for name, param in self.model.named_parameters():
        #     print(f"Layer: {name}")
        #     print(param.data)  # .data prints raw values without gradient info, cleaner output
        #     print("-" * 50)    # add a separator line, otherwise it's hard to tell which layer is which
        
        # if current_epoch % 2 == 0:
        #     self.current_phase = 'train_weights'
        #     self.model.gnn.set_alternating_phase('train_weights')
        #     print(f"[Epoch {current_epoch}] Phase: Train GNN Weights (Alphas Frozen)")
        # else:
        #     self.current_phase = 'train_arch'
        #     self.model.gnn.set_alternating_phase('train_arch')
        #     print(f"[Epoch {current_epoch}] Phase: Train Alphas (GNN Weights Frozen)")

    def configure_optimizers(self) -> Optimizer:
        
        opt = super().configure_optimizers()

        if self.cycle_length_steps is None:
            return opt
        else:
            sched = CosineAnnealingWarmRestarts(opt, self.cycle_length_steps)

            lr_dict = {
                'scheduler': sched,
                'interval': 'step',
                'frequency': 1
            }

            return {
                'optimizer': opt,
                'lr_scheduler': lr_dict
            }