import os
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_0", type=str, help="Path to model 0")
parser.add_argument("model_1", type=str, help="Path to model 1")
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)

args = parser.parse_args()

print("Loading Dreambooth model...")
model_0 = torch.load(args.model_0)
print("Loading base model...")
model_1 = torch.load(args.model_1)
theta_0 = model_0["state_dict"]
theta_1 = model_1["state_dict"]
theta_diff = {}

output_file = f'{args.output}'

for key in tqdm(theta_0.keys(), desc="Subtracting keys"):
    if "model" in key and key in theta_1:
        if not torch.equal(theta_0[key], theta_1[key]):
            theta_diff.update({key: (theta_0[key] - theta_1[key])})

print("Saving subtracted model...")

torch.save({"state_dict": theta_diff}, output_file)

# import gzip
# import shutil

# print("Compressing file...")
# MEG = 2**20
# with open(output_file, 'rb') as f_in:
#     with gzip.open(output_file + ".gz", 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out, length=16*MEG)

print("Done!")