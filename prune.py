import os
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Prune a model")
parser.add_argument("model_prune", type=str, help="Path to model to prune")
parser.add_argument("--output", type=str, help="Output file name for the pruned model", default="pruned", required=False)
args = parser.parse_args()

print("Loading model...")
model_prune = torch.load(args.model_prune)
theta_prune = model_prune["state_dict"]
theta_half = {}

for key in tqdm(theta_prune.keys(), desc="Pruning keys"):
    if "model" in key:
        theta_half.update({key: theta_prune[key].half})

output_file = f'{args.output}'

print("Saving pruned model...")

torch.save({"state_dict": theta_half}, output_file)

print("Done!")
