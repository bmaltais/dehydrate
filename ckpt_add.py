import os
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_0", type=str, help="Path to model 0")
parser.add_argument("model_1", type=str, help="Path to model 1")
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)

args = parser.parse_args()

model_0 = torch.load(args.model_0)
model_1 = torch.load(args.model_1)
theta_0 = model_0["state_dict"]
theta_1 = model_1["state_dict"]

output_file = f'{args.output}'

for key in tqdm(theta_0.keys(), desc="Stage 1/2?: merge common keys"):
    if "model" in key and key in theta_1:
        theta_0[key] = theta_0[key] + theta_1[key]

for key in tqdm(theta_1.keys(), desc="Stage 2/2: add missing keys"):
    if "model" in key and key not in theta_0:
        theta_0[key] = theta_1[key]
        print(key)

print("Saving merged model...")

torch.save({"state_dict": theta_0}, output_file)

print("Done!")