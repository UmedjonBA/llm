import os
import subprocess
import time
from pathlib import Path

def download_file(url, output_path, max_retries=3):
    """Download a file using wget with resume capability."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Use wget with resume capability and progress bar
            cmd = [
                'wget',
                '--continue',  # Resume download if file exists
                '--progress=bar:force',  # Show progress bar
                '--tries=3',  # Number of retries per file
                '--timeout=30',  # Connection timeout
                '--waitretry=5',  # Wait between retries
                '--retry-connrefused',  # Retry on connection refused
                '--no-check-certificate',  # Skip SSL certificate validation
                '-O', str(output_path),  # Output file
                url
            ]
            
            print(f"Downloading {url} to {output_path}")
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            retry_count += 1
            print(f"Error downloading {url} (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                print(f"Waiting 10 seconds before retry...")
                time.sleep(10)
            else:
                print(f"Failed to download {url} after {max_retries} attempts")
                return False

def download_model():
    # Create model directory
    model_dir = Path("codegen2-16b")
    model_dir.mkdir(exist_ok=True)
    
    # Base URL for the model files
    base_url = "https://huggingface.co/Salesforce/codegen2-16B/resolve/main"
    
    # List of model files to download
    model_files = [
        "pytorch_model-00001-of-00007.bin",
        "pytorch_model-00002-of-00007.bin",
        "pytorch_model-00003-of-00007.bin",
        "pytorch_model-00004-of-00007.bin",
        "pytorch_model-00005-of-00007.bin",
        "pytorch_model-00006-of-00007.bin",
        "pytorch_model-00007-of-00007.bin",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    # Download each file
    for file in model_files:
        url = f"{base_url}/{file}"
        output_path = model_dir / file
        if not download_file(url, output_path):
            print(f"Failed to download {file}")
            return False
    
    print("Model download completed successfully!")
    return True

if __name__ == "__main__":
    download_model() 