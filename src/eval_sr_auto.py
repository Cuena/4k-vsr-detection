import os
import sys
import pickle
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import torch
import numpy as np
from PIL import Image
from scipy import stats as st
from omegaconf import DictConfig
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

# Setup project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

def get_predictions(dataloader, model):
    """
    Get predictions for a given dataloader using the provided model.
    
    Args:
        dataloader: DataLoader object containing the data.
        model: The model to use for predictions.
    
    Returns:
        List of predictions.
    """
    predictions = []

    for i, batch in enumerate(dataloader):
       
        patches = batch
        patches = patches[:, :, :, 8:-8, 8:-8]
        patches = patches.view(-1, 3, 224, 224)
            
        if torch.cuda.is_available():
            patches = patches.cuda()

        with torch.no_grad():
            preds = model(patches)
            preds = softmax(preds)
            preds = torch.argmax(preds, dim=1)
        
        preds = preds.detach().cpu().numpy()
        predictions.extend(preds)

    return predictions

def remove_prefix(text, prefix):
    """
    Remove a given prefix from a text string.
    
    Args:
        text: The input text string.
        prefix: The prefix to remove.
    
    Returns:
        The text string with the prefix removed.
    """
    if prefix in text:
        return text.replace(prefix, "")
    log.info(f"Prefix {prefix} not found in {text}")
    return text

def repair_checkpoint(path):
    """
    Repair a checkpoint by removing prefixes from state_dict keys.
    
    Args:
        path: Path to the checkpoint file.
    
    Returns:
        The repaired checkpoint dictionary.
    """
    ckpt = torch.load(path)
    log.info("Repairing checkpoint!")
    log.info(ckpt["state_dict"].keys())
    in_state_dict = ckpt["state_dict"]
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        log.info("No need to repair checkpoint!")
        return ckpt
    
    out_state_dict = {}
    for src_key, dest_key in pairings:
        log.info(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt["state_dict"] = out_state_dict
    log.info("Checkpoint repaired!")
    log.info(ckpt["state_dict"].keys())
    return ckpt

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a given checkpoint on a datamodule testset.
    
    Args:
        cfg: DictConfig configuration composed by Hydra.
    
    Returns:
        Tuple containing metrics dictionary and object dictionary.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    ckpt = repair_checkpoint(cfg.ckpt_path)
    
    log.info("keys")
    try: 
        log.info(ckpt["state_dict"].keys())
    except:
        log.info("No state_dict keys")

    log.info("========================")

    model.load_state_dict(ckpt["state_dict"])

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    test_dataset_path = cfg.test_dataset_path
    log.info(f"Loading test dataset from {test_dataset_path}")

    video_dataloader_dict = {}

    for video in os.listdir(test_dataset_path):
        video_path = os.path.join(test_dataset_path, video)
        if not os.path.isdir(video_path):
            continue
        if len(os.listdir(video_path)) == 0:
            log.info(f"Video {video} is empty!")

        video_dataset = hydra.utils.instantiate(cfg.test_dataset, folder_path=video_path)
        video_dataloader = DataLoader(video_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=40, pin_memory=False)
        
        video_dataloader_dict[video] = {
            "video_path": video_path,
            "dataloader": video_dataloader, 
        }

    log.info("Starting testing!")

    all_predictions = {}

    for video_name, video_dataloader in video_dataloader_dict.items():
        predictions = get_predictions(dataloader=video_dataloader["dataloader"], model=model)
        all_predictions[video_name] = {
            "predictions": predictions,
        }

    ckpt_name = os.path.basename(cfg.ckpt_path).replace(".ckpt", "")
    os.makedirs(cfg.result_save_path, exist_ok=True)
    save_path = os.path.join(cfg.result_save_path, f"{ckpt_name}_predictions.pkl")

    log.info(f"Saving predictions to {save_path}")
    log.info(all_predictions.keys())
    with open(save_path, "wb") as f:
        pickle.dump(all_predictions, f)

    aggregate_metrics(all_predictions)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict

def aggregate_metrics(results: Dict[str, Any]):
    """
    Aggregate metrics for each video in the results.
    
    Args:
        results: Dictionary containing prediction results for each video.
    """
    for video_name, data in results.items():
        metrics = np.array(data["predictions"])
        video_pred = st.mode(metrics).mode
        log.info(f"Video: {video_name} - Prediction: {'Original' if video_pred == 0 else 'Upscaled'} ({video_pred})")
        log.info(video_name, video_pred)

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_sr_convnext_2.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for evaluation.
    
    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    evaluate(cfg)

if __name__ == "__main__":
    main()