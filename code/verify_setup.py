"""Quick verification that all dependencies are installed correctly."""
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
import numpy, pandas, scipy, sklearn, yaml, matplotlib
print(f"numpy {numpy.__version__}")
print(f"pandas {pandas.__version__}")
print(f"scipy {scipy.__version__}")
print(f"scikit-learn {sklearn.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print("\nAll dependencies OK!")
