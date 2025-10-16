import os
import time
import torch
import numpy as np
import platform
from fvcore.nn import FlopCountAnalysis
from sequence_generation import load_sequences
from transformer_cluj_evaluate import VisionTransformer  # adjust path if needed

# reproducibility
SEED = 1
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# load real test set
X_train, y_train = load_sequences(
    "data_uoc/output_sequences/train_sequences.npy",
    "data_uoc/output_sequences/train_labels.npy",
)
X_test, y_test = load_sequences(
    "data_uoc/output_sequences/test_sequences.npy",
    "data_uoc/output_sequences/test_labels.npy",
)
# model expects (batch, C, L)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)

# instantiate & eval
model = VisionTransformer().eval()

# count parameters
total_params = sum(p.numel() for p in model.parameters())
# count FLOPs on a single dummy segment
dummy = torch.zeros((1, X_test.shape[1], X_test.shape[2]), dtype=torch.float32)
flop_analyzer = FlopCountAnalysis(model, dummy)
total_flops = flop_analyzer.total()  # total FLOPs

# measure latency
def measure(device, data, iters=200, warm=10):
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        # warm-up
        for i in range(warm):
            _ = model(data[i % len(data):(i % len(data))+1])
        if device.type=="cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(iters):
            _ = model(data[i % len(data):(i % len(data))+1])
        if device.type=="cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

cpu_dev = torch.device("cpu")
gpu_dev = torch.device("cuda") if torch.cuda.is_available() else cpu_dev

cpu_ms = measure(cpu_dev, X_test)
gpu_ms = measure(gpu_dev, X_test)

# device names
cpu_name = platform.processor() or "Unknown CPU"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"

print(f"Parameters:         {total_params:,}")
print(f"FLOPs (fvcore):     {int(total_flops):,}")
print(f"CPU device:         {cpu_name}")
print(f"GPU device:         {gpu_name}")
print(f"Latency [ms/seg] — CPU: {cpu_ms:.3f}, GPU: {gpu_ms:.3f}")
