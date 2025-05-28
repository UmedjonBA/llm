from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model():
    print("Downloading CodeGen2 16B model...")
    model_name = "Salesforce/codegen2-16B"
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Model downloaded successfully!")
    return model, tokenizer

def test_model(model, tokenizer):
    print("\nTesting the model with a simple prompt...")
    prompt = "def hello_world():"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7
    )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated code:")
    print(generated_code)

if __name__ == "__main__":
    model, tokenizer = download_model()
    test_model(model, tokenizer) 