"""Script to encode images using CLIP and save embeddings."""
import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from scene_synthesis.datasets import filter_function
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureNormPCDataset
from training_utils import load_config
import clip

def load_clip_model(device="cpu"):
    """
    Load the CLIP model for image encoding.
    
    Args:
        device (str): The device to load the model onto ("cpu" or "cuda").
    
    Returns:
        clip_model: The loaded CLIP model.
    """
    # Load CLIP model (e.g., ViT-B/32 or RN50)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()  # Ensure the model is in evaluation mode
    return clip_model


def preprocess_clip_image():
    """
    Get the preprocessing pipeline for CLIP-compatible images.

    Returns:
        preprocess_fn: A preprocessing function for PIL images.
    """
    # The `preprocess` function is returned by `clip.load`
    _, preprocess_fn = clip.load("ViT-B/32", device="cpu", jit=False)
    return preprocess_fn





def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for object images"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Load configuration
    config = load_config(args.config_file)

    # Load datasets
    scenes_train_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["data"]["path_to_model_info"],
        path_to_models=config["data"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config["data"], config["training"].get("splits", ["train", "val"]), config["data"]["without_lamps"])
    )
    print("Loading train dataset with {} rooms".format(len(scenes_train_dataset)))

    scenes_validation_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["data"]["path_to_model_info"],
        path_to_models=config["data"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config["data"], config["validation"].get("splits", ["test"]), config["data"]["without_lamps"])
    )
    print("Loading validation dataset with {} rooms".format(len(scenes_validation_dataset)))

    # Collect the set of objects in the scenes
    train_objects = {}
    for scene in scenes_train_dataset:
        for obj in scene.bboxes:
            train_objects[obj.model_jid] = obj
    train_objects = [vi for vi in train_objects.values()]
    train_dataset = ThreedFutureNormPCDataset(train_objects)

    validation_objects = {}
    for scene in scenes_validation_dataset:
        for obj in scene.bboxes:
            validation_objects[obj.model_jid] = obj
    validation_objects = [vi for vi in validation_objects.values()]
    validation_dataset = ThreedFutureNormPCDataset(validation_objects)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} train objects".format(len(train_dataset)))

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )
    print("Loaded {} validation objects".format(len(validation_dataset)))

    # Load CLIP model
    clip_model = load_clip_model(device=device)
    preprocess = preprocess_clip_image()

    # Process datasets

    process_loader(train_loader, train_dataset, clip_model, preprocess, device)
    process_loader(val_loader, validation_dataset, clip_model, preprocess, device)


def process_loader(data_loader, dataset, clip_model, preprocess, device):
    with torch.no_grad():
        print("====> Processing Dataset ====>")
        for b, sample in enumerate(data_loader):
            idx = sample["idx"]
            for i in range(idx.shape[0]):
                idx_i = idx[i].item()
                obj = dataset.objects[idx_i]
                model_jid = obj.model_jid
                base_path = obj.raw_model_path[:-13]  # Path before `raw_model.obj`
                image_path = os.path.join(base_path, "image.jpg")

                if not os.path.exists(image_path):
                    print(f"Image not found for {model_jid}: {image_path}")
                    continue

                # Load and preprocess image
                img = Image.open(image_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)

                # Compute CLIP embedding
                embedding = clip_model.encode_image(img_tensor).cpu().numpy()

                # Save embedding
                filename_embedding = os.path.join(base_path, "image_clip_lat512.npz")
                np.savez(filename_embedding, latent=embedding)
                print(f"Saved embedding for {model_jid} to {filename_embedding}")

        print("====> Finished Processing Dataset ====>")


if __name__ == "__main__":
    main(sys.argv[1:])
