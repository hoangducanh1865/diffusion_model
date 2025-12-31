import os
import torch
import argparse


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "MNIST"  # ["MNIST", "CelebA"]
    path_to_dataset = (
        os.path.join("data", "CelebA") if dataset == "CelebA" else "data"
    )  # MNIST downloads to data
    lr = 0.0001
    num_epochs = 5
    path_to_checkpoints = os.path.join("models", "checkpoints")
    path_to_generated = os.path.join("models", "generated")

    @staticmethod
    def new_parser():
        return argparse.ArgumentParser()

    @staticmethod
    def add_argument(parser):
        parser.add_argument(
            "--device",
            type=str,
            default=Config.device,
            help="Device to use",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default=Config.dataset,
            choices=["MNIST", "CelebA"],
            help="Dataset to use",
        )
        parser.add_argument(
            "--path_to_dataset", type=str, default=Config.path_to_dataset, help="Path to dataset"
        )
        parser.add_argument("--lr", type=float, default=Config.lr, help="Learning rate")
        parser.add_argument(
            "--num_epochs", type=int, default=Config.num_epochs, help="Number of epochs"
        )
        parser.add_argument(
            "--path_to_checkpoints",
            type=str,
            default=Config.path_to_checkpoints,
            help="Path to checkpoints",
        )
        parser.add_argument(
            "--path_to_generated",
            type=str,
            default=Config.path_to_generated,
            help="Path to generated images during training process",
        )
