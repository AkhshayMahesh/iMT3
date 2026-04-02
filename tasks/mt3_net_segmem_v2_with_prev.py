from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from models.t5_segmem_v2_with_prev import T5SegMemV2WithPrev
from utils import get_cosine_schedule_with_warmup
from tasks.mt3_base import MT3Base
import os
from utils_visualize import plot_latent_embeddings


class MT3NetSegMemV2WithPrev(MT3Base):
    def __init__(self, config, optim_cfg, eval_cfg=None):
        super().__init__(config, optim_cfg, eval_cfg=eval_cfg)
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config))
        self.model: nn.Module = T5SegMemV2WithPrev(
            config=T5config,
            segmem_num_layers=self.config.segmem_num_layers,
            segmem_length=self.config.segmem_length,
        )
        self.val_z = []
        self.val_labels = []

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            inputs, targets, targets_prev, cte_family_id = batch
        else:
            inputs, targets, targets_prev = batch
            cte_family_id = None

        out = self.forward(inputs=inputs, labels=targets, targets_prev=targets_prev, cte_family_id=cte_family_id)
        if isinstance(out, tuple):
            if len(out) == 3:
                lm_logits, loss_cte, _ = out
            else:
                lm_logits, loss_cte = out
        else:
            lm_logits, loss_cte = out, None

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), targets.view(-1)
            )
        if loss_cte is not None:
            lam = float(getattr(self.config, "cte_lambda", 0.0))
            loss = loss + (lam * loss_cte)
            self.log('train_loss_cte', loss_cte, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
            self.log('train_loss_cte_weighted', lam * loss_cte, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            inputs, targets, targets_prev, cte_family_id = batch
        else:
            inputs, targets, targets_prev = batch
            cte_family_id = None

        out = self.forward(inputs=inputs, labels=targets, targets_prev=targets_prev, cte_family_id=cte_family_id)
        if isinstance(out, tuple):
            if len(out) == 3:
                lm_logits, loss_cte, z = out
            else:
                lm_logits, loss_cte = out
                z = None
        else:
            lm_logits, loss_cte, z = out, None, None

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), targets.view(-1)
            )
        if loss_cte is not None:
            lam = float(getattr(self.config, "cte_lambda", 0.0))
            loss = loss + (lam * loss_cte)
            self.log('val_loss_cte', loss_cte, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_loss_cte_weighted', lam * loss_cte, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        if z is not None and cte_family_id is not None:
            self.val_z.append(z.detach().cpu())
            self.val_labels.append(cte_family_id.detach().cpu())

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if len(self.val_z) > 0:
            # Concatenate all gathered embeddings and labels
            embeddings = torch.cat(self.val_z, dim=0)
            labels = torch.cat(self.val_labels, dim=0)
            save_dir = self.logger.log_dir if self.logger else "."
            save_path = os.path.join(save_dir, f"cte_embeddings_epoch_{self.current_epoch}.png")
            
            # Plot and log to tensorboard
            plot_latent_embeddings(
                embeddings=embeddings, 
                labels=labels, 
                logger=self.logger, 
                current_epoch=self.current_epoch, 
                save_path=save_path
            )
            
            # Clear the lists for the next validation epoch
            self.val_z.clear()
            self.val_labels.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.optim_cfg.lr)
        warmup_step = int(self.optim_cfg.warmup_steps)
        print('warmup step: ', warmup_step)
        schedule = {
            'scheduler': get_cosine_schedule_with_warmup(
                optimizer=optimizer, 
                num_warmup_steps=warmup_step, 
                num_training_steps=self.optim_cfg.num_steps_per_epoch * self.optim_cfg.num_epochs,
                min_lr=self.optim_cfg.min_lr
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [schedule]

        # we follow MT3 to use fixed learning rate
        # NOTE: we find this to not work :(
        # return AdamW(self.model.parameters(), self.config.lr)