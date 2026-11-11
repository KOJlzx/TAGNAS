"""Legacy single-device ConGraT utility with a predefined dataset/model sweep."""

import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import time
from dataclasses import dataclass, asdict
from omegaconf import OmegaConf
from tqdm import tqdm
# Import original modules
from clip_graph.lit import LitClipGraph
from clip_graph.data.datamodule import PubmedGraphTextDataModule
from clip_graph.data.datamodule import ArxivGraphTextDataModule

@dataclass
class TrainingConfig:
    config_path: str
    # pubmed / cora / arxiv: independent of data_dir in the yaml; set to data/<dataset> by _setup_data below
    # trex / trex-textinit: same directory layout as pubmed (raw/graph.csv, raw/texts.csv, raw/node-data.csv), uses PubmedGraphTextDataModule
    dataset: str = "pubmed"
    gnn_type: str = "GAT"  # Added: defaults to GAT, can also pass "GCN", "SAGE", etc.
    num_workers: int = 0  # For single-machine runs, typically set to 0 or the CPU core count
    num_epochs: int = 10
    learning_rate: float = 3e-5
    checkpoint_frequency: int = 5
    log_frequency: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping_patience: int = 3  # Early stopping patience
    # Must match single-machine Lightning behavior: datamodule.setup(node=0, world_size=1) uses the full graph.
    # When world_size>1, only a 1/world_size node subgraph is used for training, which hurts contrastive learning significantly.
    world_size: int = 1
    gradient_clip_norm: float = 1.0  # Aligned with gradient_clip_val in configs/.../base.yaml
    # Output directory: weights and logs are saved under model/runpytorch/<run_name>/
    output_root: str = "model/runpytorch"
    run_name: Optional[str] = None  # Defaults to auto-generated pubmed_{gnn_type}_{timestamp}


def _make_run_dir(config: TrainingConfig) -> str:
    root = os.path.abspath(os.path.expanduser(config.output_root))
    os.makedirs(root, exist_ok=True)
    if config.run_name:
        name = config.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{config.dataset}_{config.gnn_type}_{ts}"
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_training_artifacts(
    run_dir: str,
    config: TrainingConfig,
    model_config: OmegaConf,
) -> tuple[str, str, str]:
    """Returns the (metrics_csv, train_log, checkpoints_dir) paths."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    train_log = os.path.join(run_dir, "training.log")
    # Snapshot of training hyperparameters
    with open(os.path.join(run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    OmegaConf.save(model_config, os.path.join(run_dir, "model_config_snapshot.yaml"))
    # CSV header
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "best_val_loss", "timestamp"])
    with open(train_log, "w", encoding="utf-8") as f:
        f.write(f"run_dir={run_dir}\n")
        f.write(f"started={datetime.now().isoformat()}\n")
        f.write(f"config_path={config.config_path}\n")
        f.write(f"gnn_type={config.gnn_type}\n\n")
    return metrics_csv, train_log, ckpt_dir


def _append_metric_row(
    metrics_csv: str,
    train_log: str,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    best_val_loss: float,
) -> None:
    ts = datetime.now().isoformat()
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                epoch + 1,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                f"{best_val_loss:.6f}" if best_val_loss < float("inf") else "",
                ts,
            ]
        )
    with open(train_log, "a", encoding="utf-8") as f:
        line = f"epoch={epoch + 1} train_loss={train_loss:.6f}"
        if val_loss is not None:
            line += f" val_loss={val_loss:.6f}"
        if best_val_loss < float("inf"):
            line += f" best_val_loss={best_val_loss:.6f}"
        f.write(line + "\n")


class GraphModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.epoch = 0
        self.global_step = 0

        self.scaler = torch.cuda.amp.GradScaler()
        
        # Early stopping related variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.best_epoch = 0
        
        print(f"Initializing Trainer on device: {self.device}")
        print(f"Target GNN Type: {self.config.gnn_type}") # Print the currently selected GNN
        print('_'*100)

        # 1. Load configuration
        self.model_config = OmegaConf.load(config.config_path)
        self.k_hop = self.model_config.model.init_args.gnn_params.num_layers

        # 2. Initialize data: world_size=1 is required for single-machine runs (matches Lightning default)
        self._setup_data(rank=0, world_size=self.config.world_size)

        # 3. Initialize model
        self._setup_model()

        # 4. Initialize optimizer
        self._setup_optimizer()

    def _setup_data(self, rank: int, world_size: int):
        """Set up the data module"""
        print("Setting up data...")
        init_args = dict(self.model_config.data.init_args)
        init_args["data_dir"] = os.path.join("data", self.config.dataset)

        if self.config.dataset.lower() == "arxiv":
            datamodule = ArxivGraphTextDataModule(**init_args)
        else:
            # pubmed/cora/trex/trex-textinit: raw/*.csv format convention used by BaseDataModule
            datamodule = PubmedGraphTextDataModule(**init_args)
        datamodule.prepare_data()
        # In single-machine mode, world_size is 1
        datamodule.setup(node=rank, world_size=world_size, stage='fit')
        
        train_dataset = datamodule.train_dataset
        val_dataset = datamodule.val_dataset

        # Create data loader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=1, # Kept consistent with the original source code
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.model_config.data.init_args.pin_memory,
            collate_fn=train_dataset.__collate__
        )
        
        # Only create it when the validation set exists
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.model_config.data.init_args.pin_memory,
                collate_fn=val_dataset.__collate__
            )
        else:
            self.val_dataloader = None
            print("Warning: No validation dataset found. Early stopping will be disabled.")

    def _setup_model(self):
        """Set up the model: inject the gnn_type parameter here"""
        print(f"Setting up model with GNN: {self.config.gnn_type}...")
        model_args = self.model_config.model.init_args.copy()
        model_args['lr'] = self.config.learning_rate
        
        # Core modification: inject the gnn_type passed from the command line into the config dict
        # Assumes that in your config file structure, gnn_params corresponds to GNNMod's init parameters
        if 'gnn_params' in model_args:
            model_args['gnn_params']['gnn_type'] = self.config.gnn_type

        # Instantiate the model
        self.model = LitClipGraph(**model_args).to(self.device)

    def _setup_optimizer(self):
        """Set up the optimizer (consistent with LitBase.configure_optimizers / yaml: eps, betas)"""
        ma = self.model_config.model.init_args
        betas = ma.get("betas", (0.9, 0.999))
        betas = tuple(betas) if not isinstance(betas, tuple) else betas
        eps = float(ma.get("eps", 1e-8))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=float(ma.weight_decay),
            eps=eps,
            betas=betas,
        )

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to the device"""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch

    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        self.epoch = epoch
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        progress_bar = tqdm(
            enumerate(self.train_dataloader), 
            total=len(self.train_dataloader),
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            ncols=100
        )
        
        for batch_idx, batch in progress_bar:
            # 1. Move data
            batch = self._move_batch_to_device(batch)


            """ # 3. Forward pass & compute loss (assumes model.training_step returns a loss tensor)
            loss = self.model.training_step(batch, batch_idx)

            # 4. Backward pass
            loss.backward()

            # 5. Update parameters
            self.optimizer.step() """

            with torch.cuda.amp.autocast():
                loss = self.model.training_step(batch, batch_idx)

            self.scaler.scale(loss).backward()
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn_utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Record data
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss_val:.4f}',
                'Avg': f'{total_loss/num_batches:.4f}'
            })

            # Print log
            if batch_idx % self.config.log_frequency == 0:
                 # Optionally print detailed logs here; tqdm already shows most of the information
                 pass

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """Validation loop"""
        if not self.val_dataloader:
            return None
            
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self._move_batch_to_device(batch)
                loss = self.model.validation_step(batch, batch_idx)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    def save_checkpoint(self, path: str):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step
        }
        torch.save(state, path)
        print(f"Checkpoint saved to {path}")

    def save_best_model(self, path: str):
        """Save the best model"""
        if self.best_model_state is not None:
            state = {
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.best_epoch,
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss
            }
            torch.save(state, path)
            print(f"Best model saved to {path} (Epoch {self.best_epoch+1}, Val Loss: {self.best_val_loss:.4f})")
        else:
            print("Warning: No best model state to save.")

def train_single_gpu(config: TrainingConfig):
    run_dir = _make_run_dir(config)
    metrics_csv, train_log, ckpt_dir = _write_training_artifacts(
        run_dir, config, OmegaConf.load(config.config_path)
    )
    print(f"Run directory: {run_dir}")

    # Instantiate the trainer
    trainer = GraphModelTrainer(config)

    best_model_name = os.path.join(run_dir, "best_model.pth")
    final_model_name = os.path.join(run_dir, "final_model.pth")

    # Check whether a validation set exists
    if trainer.val_dataloader is None:
        print("Warning: No validation dataset. Training without early stopping.")
        print(f"Starting training for {config.num_epochs} epochs...")
        
        for epoch in range(config.num_epochs):
            train_loss = trainer.train_epoch(epoch)
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
            _append_metric_row(metrics_csv, train_log, epoch, train_loss, None, trainer.best_val_loss)
            ds = trainer.train_dataloader.dataset
            if hasattr(ds, "next_epoch"):
                ds.next_epoch()

            if (epoch + 1) % config.checkpoint_frequency == 0:
                save_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
                trainer.save_checkpoint(save_path)
        
        trainer.save_checkpoint(final_model_name)
        with open(train_log, "a", encoding="utf-8") as f:
            f.write(f"\nfinished={datetime.now().isoformat()}\n")
        print(f"Training completed! Artifacts in {run_dir}")
        return
    
    print(f"Starting training for {config.num_epochs} epochs with early stopping (patience={config.early_stopping_patience})...")

    for epoch in range(config.num_epochs):
        # 1. Train
        train_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        # Consistent with RebuildContrastiveDataset: refresh the batch graph construction at the end of each epoch (change random seed)
        ds = trainer.train_dataloader.dataset
        if hasattr(ds, "next_epoch"):
            ds.next_epoch()

        # 2. Validate
        val_loss = trainer.validate()
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        # 3. Check whether this is the best model
        if val_loss < trainer.best_val_loss:
            print(f"✓ Validation loss improved from {trainer.best_val_loss:.4f} to {val_loss:.4f}")
            trainer.best_val_loss = val_loss
            trainer.best_epoch = epoch
            trainer.patience_counter = 0
            # Save the best model state
            trainer.best_model_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
        else:
            trainer.patience_counter += 1
            print(f"✗ Validation loss did not improve. Patience: {trainer.patience_counter}/{config.early_stopping_patience}")

        _append_metric_row(
            metrics_csv, train_log, epoch, train_loss, val_loss, trainer.best_val_loss
        )

        # 4. Early stopping check
        if trainer.patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement for {config.early_stopping_patience} epochs.")
            print(f"Best validation loss: {trainer.best_val_loss:.4f} at epoch {trainer.best_epoch+1}")
            break

        # 5. Periodically save checkpoint
        if (epoch + 1) % config.checkpoint_frequency == 0:
            save_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(save_path)
    
    # Save the best model
    trainer.save_best_model(best_model_name)

    # Save the final model (the model from the current epoch)
    trainer.save_checkpoint(final_model_name)
    with open(train_log, "a", encoding="utf-8") as f:
        f.write(
            f"\nfinished={datetime.now().isoformat()} "
            f"best_epoch={trainer.best_epoch + 1} best_val_loss={trainer.best_val_loss:.6f}\n"
        )
    print(f"Training completed! Artifacts in {run_dir}")

if __name__ == "__main__":
    # Define all GNN types to test
    #gnn_models = [ "GCN",  "LINEAR"]
    gnn_models = ["GAT","GCN","SAGE","GIN","LINEAR"]

    # Datasets: under data/<name>/raw/ are graph.csv, texts.csv, node-data.csv (tab-separated)
    # trex-textinit defaults to pointing at data/trex-textinit (its raw is often symlinked to ../trex/raw)
    # datasets = ["pubmed", "trex-textinit"]
    datasets = ["trex-textinit"]

    for dataset_name in datasets:
        for model_name in gnn_models:
            config = TrainingConfig(
                # Consistent with the inductive-causal base model hyperparameters for pubmed/trex; data_dir is overridden by dataset to data/<dataset>
                config_path="./configs/clip-graph/inductive-causal/pubmed/base.yaml",
                dataset=dataset_name,
                gnn_type=model_name,
                num_epochs=30,
                learning_rate=1e-4,
                checkpoint_frequency=10,
                log_frequency=10,
                num_workers=12,
                early_stopping_patience=3,
            )

            try:
                train_single_gpu(config)
                torch.cuda.empty_cache()
            except KeyboardInterrupt:
                print("Training interrupted by user")
            except Exception as e:
                import traceback
                traceback.print_exc()