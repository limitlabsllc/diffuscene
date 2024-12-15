import torch
import json
import numpy as np

# Paths
model_checkpoint = "../experiment_6000/diffusion_diningrooms_instancond_lat32_v/model_01500"
output_path = "../experiment_1500/diffusion_diningrooms_instancond_lat32_v/diffusion_diningrooms_instancond_lat32_v.pt"


# Load the model state dictionary
model_state = torch.load(model_checkpoint)

# Combine into a single dictionary

# Save the consolidated data
torch.save(model_state, output_path)
print(f"Model saved to {output_path}")