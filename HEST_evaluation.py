from hest.bench import benchmark
import torch
from torchvision import transforms

print("loading base")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

ours = torch.load("checkpoints/teacher_epoch250000.pth")
checkpoint = ours["teacher"]
checkpoint_new = {}

for key in list(checkpoint.keys()):
    if "dino" in str(key) or "ibot" in str(key):
        checkpoint.pop(key, None)

for key, keyb in zip(checkpoint.keys(), dinov2.state_dict().keys()):
    checkpoint_new[keyb] = checkpoint[key]

checkpoint = checkpoint_new

new_shape = checkpoint["pos_embed"] #The pos embed is the only different shape
dinov2.pos_embed = torch.nn.parameter.Parameter(new_shape)

dinov2.load_state_dict(checkpoint)

PATH_TO_CONFIG = "./HEST/bench_config/bench_config.yaml"
model = dinov2

RESIZE_DIM = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

model_transforms = transforms.Compose([
    #transforms.Resize(224),  # Resize the smaller side of the image to 256
    #transforms.CenterCrop(RESIZE_DIM), # Crop the center of the image to 224x224
    
    # Step 2: Convert the image (PIL/numpy) to a PyTorch tensor
    transforms.ToTensor(),

    # Step 3: Normalize the tensor
    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD)
    ])
  
benchmark(        
    model, 
    model_transforms,
    torch.float32,
    config=PATH_TO_CONFIG, 
)