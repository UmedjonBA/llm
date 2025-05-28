# CodeGen2-16B Model Setup and Usage Guide

This guide provides detailed instructions for setting up and using the CodeGen2-16B model on Ubuntu.

## System Requirements

- Ubuntu 20.04 or later
- Python 3.8 or later
- At least 16GB RAM (32GB recommended)
- NVIDIA GPU with at least 8GB VRAM (recommended)
- CUDA 11.7 or later (if using GPU)

## Installation Steps

### 1. Install Python and Required Tools

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install git and wget
sudo apt install git wget

# Install CUDA (if using NVIDIA GPU)
# Follow instructions at: https://developer.nvidia.com/cuda-downloads
```

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 3. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 5. Download the Model

There are two ways to download the model:

#### Option 1: Using download_model.py (Recommended for slow connections)

This script downloads the model files directly using wget with resume capability:

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the download script
python download_model.py
```

This will:
- Create a `codegen2-16b` directory
- Download all model files (about 32GB total)
- Show progress for each file
- Automatically resume if download is interrupted
- Verify all files are downloaded correctly

#### Option 2: Using download_codegen.py

This script uses the Hugging Face transformers library to download the model:

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the download script
python download_codegen.py
```

This will:
- Download the model using the transformers library
- Automatically handle model quantization (float16)
- Test the model with a simple code generation example
- Show the generated output

## Usage

### Basic Usage

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Create a new Python script (e.g., `generate_code.py`):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "codegen2-16b"  # Use local path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate code
prompt = "def calculate_fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=200,
    num_return_sequences=1,
    temperature=0.7
)

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_code)
```

3. Run your script:
```bash
python generate_code.py
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use model quantization (already enabled in the scripts)
   - Close other GPU applications
   - Try using CPU if GPU memory is insufficient

2. **Download Issues**
   - If download_model.py fails, try download_codegen.py
   - Check your internet connection
   - Verify you have enough disk space (at least 32GB free)
   - Try downloading with a VPN if the connection is blocked

3. **Python Package Installation Errors**
   - Make sure you're in the virtual environment
   - Try updating pip: `pip install --upgrade pip`
   - Install build essentials: `sudo apt install build-essential`

### Getting Help

If you encounter any issues:
1. Check the error message carefully
2. Search for similar issues in the project's issue tracker
3. Create a new issue with detailed information about your problem

## Performance Optimization

### GPU Acceleration

To ensure optimal GPU performance:
1. Install the latest NVIDIA drivers
2. Use CUDA 11.7 or later
3. The scripts already use model quantization (float16) for better performance

### Memory Management

- The scripts use automatic device mapping (`device_map="auto"`)
- Model quantization is enabled by default
- Monitor GPU memory usage with `nvidia-smi`

## Security Considerations

- Keep your model files secure
- Don't share API keys or credentials
- Regularly update dependencies for security patches

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable] 