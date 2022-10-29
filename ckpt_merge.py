import os
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_0", type=str, help="Path to model 0")
parser.add_argument("model_1", type=str, help="Path to model 1")
parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5", default=0.5, required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)

args = parser.parse_args()

model_0 = torch.load(args.model_0)
model_1 = torch.load(args.model_1)
theta_0 = model_0["state_dict"]
theta_1 = model_1["state_dict"]
alpha = args.alpha

output_file = f'{args.output}'
# output_file = f'{args.output}.ckpt'

# check if output file already exists, ask to overwrite
# if os.path.isfile(output_file):
#     print("Output file already exists. Overwrite? (y/n)")
#     while True:
#         overwrite = input()
#         if overwrite == "y":
#             break
#         elif overwrite == "n":
#             print("Exiting...")
#             exit()
#         else:
#             print("Please enter y or n")


# for key in tqdm(theta_0.keys(), desc="Stage 1/2?: merge common keys"):
#     print(key)
#     if "model" in key and key in theta_1:
#         theta_0[key] = theta_0[key] - theta_1[key]

#  for key in tqdm(theta_1.keys(), desc="Stage 2/2: add missing keys"):
#     if "model" in key and key not in theta_0:
#         theta_0[key] = theta_1[key]

for key in tqdm(theta_0.keys(), desc="Stage 1/2?: merge common keys"):
    if "model" in key and key in theta_1:
        theta_0[key] = alpha * theta_0[key] + (1 - alpha) * theta_1[key]
        theta_0[key].half

for key in tqdm(theta_1.keys(), desc="Stage 2/2: add missing keys"):
    if "model" in key and key not in theta_0:
        print(key)
        theta_0[key] = theta_1[key]
        theta_0[key].half

print("Saving merged models...")

torch.save({"state_dict": theta_0}, output_file)

print("Done!")