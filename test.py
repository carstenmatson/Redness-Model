import torch

model_path = r"C:\Users\cmats\Documents\project\Model Folder\skin_redness.pth"
state_dict = torch.load(model_path, map_location="cpu")

print(state_dict.keys())  # Check available keys
