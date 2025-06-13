import torch
import torch.nn as nn
import torch.cuda
import argparse
import time

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

def benchmark_models(models, input_tensor, device="cuda:0", selected_model=None):
    for m in models:
        name = m["name"]
        if selected_model and name.lower() != selected_model.lower():
            continue

        model = m["model"].to(device).eval()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        with torch.no_grad():
            _ = model(input_tensor)

        vram = get_vram_usage_mb(device)
        print(f"{name:10s} | Allocated: {vram['allocated_MB']} MB | Reserved: {vram['reserved_MB']} MB")

        # Clean up
        del model
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VRAM usage for different models.")
    parser.add_argument("--model", type=str, help="Model name to run (or all if not specified).")
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512],
                        help="Input image size as height width (default: 512 512)")
    args = parser.parse_args()

    device = "cuda:0"
    h, w = args.input_size
    x = torch.randn(1, 3, h, w).to(device)

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

    benchmark_models(models, x, device=device, selected_model=args.model)
