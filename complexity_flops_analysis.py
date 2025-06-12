import torch
from ptflops import get_model_complexity_info

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import all models
from models.Restormer import Net as Restormer
from models.PromptIR import PromptIR
from models.HAIR import Net as HAIR
from models.AdaIR import Net as AdaIR
from models.QuaHaarIR import Net as QuaHaarIR

# List of model instances with names
models = [
    {"name": "Restormer", "model": Restormer()},
    {"name": "PromptIR", "model": PromptIR(decoder=True)},
    {"name": "HAIR", "model": HAIR()},
    {"name": "AdaIR", "model": AdaIR(decoder=True)},
    {"name": "QuaHaarIR", "model": QuaHaarIR(decoder=True)},
]

# Input shape and constructor
input_shape = (1, 3, 256, 256)
def input_construct(input_res):
    return {"inp_img": torch.rand(input_res).to(device)}

# Evaluate each model
print("Model Complexity Report:\n")
for m in models:
    model = m["model"].to(device).eval()
    name = m["name"]

    macs, params = get_model_complexity_info(
        model,
        input_shape,
        input_constructor=input_construct,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"{'='*40}")
    print(f"Model     : {name}")
    print(f"Parameters: {params}")
    print(f"FLOPs     : {macs}")
