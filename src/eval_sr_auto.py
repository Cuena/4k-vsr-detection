import os
import pickle
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import rootutils
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from scipy import stats
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

# Set up the root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

def get_predictions(dataloader: DataLoader, model: LightningModule) -> List[int]:
    """
    Get predictions for a given dataloader using the provided model.

    Args:
        dataloader (DataLoader): The dataloader containing the data to predict.
        model (LightningModule): The model to use for predictions.

    Returns:
        List[int]: A list of predictions.
    """
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            log.info(f"Processing video {i}")
            patches = batch[:, :, :, 8:-8, 8:-8].view(-1, 3, 224, 224).to(device)
            
            preds = model(patches)
            preds = softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            
            predictions.extend(preds.cpu().numpy())

    return predictions

def remove_prefix(text: str, prefix: str) -> str:
    """
    Remove a prefix from a string if it exists.

    Args:
        text (str): The input string.
        prefix (str): The prefix to remove.

    Returns:
        str: The string with the prefix removed if it existed, otherwise the original string.
    """
    return text[len(prefix):] if text.startswith(prefix) else text

def repair_checkpoint(path: str) -> Dict[str, Any]:
    """
    Repair a checkpoint by removing prefixes from state_dict keys.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        Dict[str, Any]: The repaired checkpoint dictionary.
    """
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    log.info("Repairing checkpoint")
    
    in_state_dict = ckpt["state_dict"]
    out_state_dict = {remove_prefix(k, "_orig_mod."): v for k, v in in_state_dict.items()}
    
    if in_state_dict.keys() == out_state_dict.keys():
        log.info("No need to repair checkpoint")
        return ckpt
    
    ckpt["state_dict"] = out_state_dict
    log.info("Checkpoint repaired")
    return ckpt

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a model on a test dataset.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the metrics and instantiated objects.
    """
    assert cfg.ckpt_path, "Checkpoint path must be provided"

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    ckpt = repair_checkpoint(cfg.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

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
        log.info("Logging hyperparameters")
        log_hyperparameters(object_dict)

    log.info(f"Loading test dataset from {cfg.test_dataset_path}")
    video_dataloader_dict = load_test_datasets(cfg)

    log.info("Starting evaluation")
    all_predictions = evaluate_model(model, video_dataloader_dict)

    save_predictions(all_predictions, cfg)
    aggregate_metrics(all_predictions)

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict

def load_test_datasets(cfg: DictConfig) -> Dict[str, Dict[str, Any]]:
    """
    Load test datasets for each video in the test dataset path.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing dataloaders for each video.
    """
    video_dataloader_dict = {}
    for video in os.listdir(cfg.test_dataset_path):
        video_path = os.path.join(cfg.test_dataset_path, video)
        if not os.path.isdir(video_path) or len(os.listdir(video_path)) == 0:
            continue

        video_dataset = hydra.utils.instantiate(cfg.test_dataset, folder_path=video_path)
        video_dataloader = DataLoader(
            video_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )
        
        video_dataloader_dict[video] = {
            "video_path": video_path,
            "dataloader": video_dataloader, 
        }
    return video_dataloader_dict

def evaluate_model(model: LightningModule, video_dataloader_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[int]]]:
    """
    Evaluate the model on all videos in the test dataset.

    Args:
        model (LightningModule): The model to evaluate.
        video_dataloader_dict (Dict[str, Dict[str, Any]]): A dictionary containing dataloaders for each video.

    Returns:
        Dict[str, Dict[str, List[int]]]: A dictionary containing predictions for each video.
    """
    all_predictions = {}
    for video_name, video_data in video_dataloader_dict.items():
        predictions = get_predictions(dataloader=video_data["dataloader"], model=model)
        all_predictions[video_name] = {"predictions": predictions}
    return all_predictions

def save_predictions(all_predictions: Dict[str, Dict[str, List[int]]], cfg: DictConfig) -> None:
    """
    Save the predictions to a file.

    Args:
        all_predictions (Dict[str, Dict[str, List[int]]]): A dictionary containing predictions for each video.
        cfg (DictConfig): The configuration dictionary.
    """
    ckpt_name = os.path.basename(cfg.ckpt_path).replace(".ckpt", "")
    os.makedirs(cfg.result_save_path, exist_ok=True)
    save_path = os.path.join(cfg.result_save_path, f"{ckpt_name}_predictions.pkl")

    log.info(f"Saving predictions to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(all_predictions, f)

def aggregate_metrics(results: Dict[str, Dict[str, List[int]]]) -> None:
    """
    Aggregate and log metrics for each video.

    Args:
        results (Dict[str, Dict[str, List[int]]]): A dictionary containing predictions for each video.
    """
    for video_name, data in results.items():
        metrics = np.array(data["predictions"])
        video_pred = stats.mode(metrics).mode[0]
        prediction_label = 'Original' if video_pred == 0 else 'Upscaled'
        log.info(f"Video: {video_name} - Prediction: {prediction_label} ({video_pred})")

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_sr_convnext.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for evaluation.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    extras(cfg)
    evaluate(cfg)

if __name__ == "__main__":
    main()