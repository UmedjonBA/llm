from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

print("Starting model download...")
model_name = "Salesforce/codegen25-7b-instruct_P"

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully!")

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Using float16 for better memory efficiency
        device_map="auto"  # Automatically choose the best available device
    )
    print("Model loaded successfully!")

    # Example using pipeline
    print("\nTesting model with pipeline...")
    pipe = pipeline("text-generation", model=model_name)
    pipeline_output = pipe("def hello_world():", max_length=100, temperature=0.7)
    print("\nPipeline generated code:")
    print(pipeline_output[0]['generated_text'])

    # Example using direct model
    print("\nTesting model directly...")
    prompt = "def hello_world():"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7
    )
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nDirect model generated code:")
    print(generated_code)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc() 