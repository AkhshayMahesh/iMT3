"""
MT3 baseline training. 
To use random order, use `dataset.dataset_2_random`. Or else, use `dataset.dataset_2`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, SequentialSampler

import torch
import pytorch_lightning as pl

import hydra
from tasks.mt3_net import MT3Net
from dataset.timbre_sampler import TimbreContrastiveBatchSampler

class EpochMetricsLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss_epoch', metrics.get('train_loss', "N/A"))
        train_loss_cte = metrics.get('train_loss_cte', "N/A")
        
        def get_val(v):
            if isinstance(v, torch.Tensor):
                return f"{v.item():.4f}"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
            
        out_str = f"Epoch {epoch} Train:\nTrain Loss={get_val(train_loss)}\nTrain Contrastive Loss={get_val(train_loss_cte)}\n"
        
        try:
            from hydra.core.hydra_config import HydraConfig
            current_dir = HydraConfig.get().runtime.output_dir
            log_file = os.path.join(current_dir, f"epoch_metric_{epoch}.txt")
        except:
            log_file = f"epoch_metric_{epoch}.txt"
            
        with open(log_file, "a") as f:
            f.write(out_str)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
            
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss', "N/A")
        val_loss_cte = metrics.get('val_loss_cte', "N/A")
        f1_flat = metrics.get('val_f1_flat', "N/A")
        f1_midi = metrics.get('val_f1_midi_class', "N/A")
        f1_full = metrics.get('val_f1_full', "N/A")
        
        def get_val(v):
            if isinstance(v, torch.Tensor):
                return f"{v.item():.4f}"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        
        out_str = f"Epoch {epoch} Validation:\nVal Loss={get_val(val_loss)}\nVal Contrastive Loss={get_val(val_loss_cte)}\nF1 Flat={get_val(f1_flat)}\nF1 MidiClass={get_val(f1_midi)}\nF1 Full={get_val(f1_full)}\n\n"
        
        try:
            from hydra.core.hydra_config import HydraConfig
            current_dir = HydraConfig.get().runtime.output_dir
            log_file = os.path.join(current_dir, f"epoch_metric_{epoch}.txt")
        except:
            current_dir = "."
            log_file = f"epoch_metric_{epoch}.txt"
            
        with open(log_file, "a") as f:
            f.write(out_str)

        if isinstance(val_loss, torch.Tensor):
            current_val_loss = val_loss.item()
        elif isinstance(val_loss, float):
            current_val_loss = val_loss
        else:
            current_val_loss = float('inf')

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            ckpt_path = os.path.join(current_dir, f"best_model_{epoch}.pt")
            
            dic = {}
            for key in pl_module.state_dict():
                if "model." in key:
                    dic[key.replace("model.", "")] = pl_module.state_dict()[key]
                else:
                    dic[key] = pl_module.state_dict()[key]
            torch.save(dic, ckpt_path)

@hydra.main(config_path="config", config_name="config")
# def main(config, model_config, result_dir, mode, path):
def main(cfg):
    # set seed to ensure reproducibility
    pl.seed_everything(cfg.seed)

    model = hydra.utils.instantiate(
        cfg.model, 
        optim_cfg=cfg.optim,
        eval_cfg=cfg.eval    
    )
    logger = TensorBoardLogger(save_dir='.',
                               name=f"{cfg.model_type}_{cfg.dataset_type}")
    
    # sanity check to make sure the correct model is used
    assert cfg.model_type == cfg.model._target_.split('.')[-1]

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)
    tqdm_callback = TQDMProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback, tqdm_callback, EpochMetricsLoggingCallback()],
        **cfg.trainer
    )

    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    sampler = TimbreContrastiveBatchSampler(
        sampler=SequentialSampler(train_dataset),
        dataset=train_dataset,
        families_per_batch=8,
        samples_per_family=4
    )
    
    dataloader_train_kwargs = dict(cfg.dataloader.train)
    for conf_key in ["batch_size", "shuffle", "sampler", "drop_last"]:
        dataloader_train_kwargs.pop(conf_key, None)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn),
        **dataloader_train_kwargs
    )

    val_loader = DataLoader(
        hydra.utils.instantiate(cfg.dataset.val),
        **cfg.dataloader.val,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn)
    )

    if cfg.path is not None and cfg.path != "":
        if cfg.path.endswith(".ckpt"):
            print(f"Validating on {cfg.path}...")
            trainer.validate(
                model, 
                val_loader,
                ckpt_path=cfg.path
            )
            print("Training start...")
            trainer.fit(
                model, 
                train_loader, 
                val_loader,
                ckpt_path=cfg.path
            )

        elif cfg.path.endswith(".pth"):
            print(f"Loading weights from {cfg.path}...")
            model.model.load_state_dict(
                torch.load(cfg.path),
                strict=False
            )
            trainer.validate(
                model, 
                val_loader,
            )
            print("Training start...")
            trainer.fit(
                model, 
                train_loader, 
                val_loader,
            )
        
        else:
            raise ValueError(f"Invalid extension for path: {cfg.path}")
    
    else:
        trainer.fit(
                model, 
                train_loader, 
                val_loader,
            )    

    # save the model in .pt format
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_path = os.path.join(current_dir, f"{cfg.model_type}_{cfg.dataset_type}", "version_0/checkpoints/last.ckpt")
    model.eval()
    dic = {}
    for key in model.state_dict():
        if "model." in key:
            dic[key.replace("model.", "")] = model.state_dict()[key]
        else:
            dic[key] = model.state_dict()[key]
    torch.save(dic, ckpt_path.replace(".ckpt", ".pt"))
    print(f"Saved model in {ckpt_path.replace('.ckpt', '.pt')}.")
        

if __name__ == "__main__":   
    main()
