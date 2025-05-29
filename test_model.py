from transformers import pipeline

print("Starting model download...")
model_name = "Salesforce/codegen25-7b-mono_P"

try:
    print("Loading model via pipeline...")
    pipe = pipeline("text-generation", model=model_name, trust_remote_code=True)
    print("Model loaded successfully!")

    # Example using pipeline
    print("\nTesting model with pipeline...")
    prompt = "def hello_world():"
    pipeline_output = pipe(prompt, max_length=100, temperature=0.7)
    print("\nPipeline generated code:")
    print(pipeline_output[0]['generated_text'])

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc() 