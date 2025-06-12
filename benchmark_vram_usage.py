import torch
import torch.nn as nn
import torch.cuda

def get_vram_usage_mb(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return {
        "allocated_MB": round(allocated, 2),
        "reserved_MB": round(reserved, 2)
    }

def benchmark_models(models, input_tensor, device="cuda:0"):
    for m in models:
        model = m["model"].to(device).eval()
        name = m["name"]

        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = model(input_tensor)

        vram = get_vram_usage_mb(device)
        print(f"{name:10s} | Allocated: {vram['allocated_MB']} MB | Reserved: {vram['reserved_MB']} MB")

if __name__ == "__main__":
    device = "cuda:0"
    x = torch.randn(1, 3, 128, 128).to(device)

    # Load models
    from models.Restormer import Net as Restormer
    from models.PromptIR import PromptIR
    from models.HAIR import Net as HAIR
    from models.AdaIR import Net as AdaIR
    from models.QuaHaarIR import Net as QuaHaarIR

    models = [
        {"name": "Restormer", "model": Restormer()},
        {"name": "PromptIR", "model": PromptIR(decoder=True)},
        {"name": "HAIR", "model": HAIR()},
        {"name": "AdaIR", "model": AdaIR(decoder=True)},
        {"name": "QuaHaarIR", "model": QuaHaarIR(decoder=True)},
    ]

    benchmark_models(models, x, device=device)
