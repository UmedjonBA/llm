from transformers import pipeline

def load_model():
    print("Loading CodeGen25 7B Mono model...")
    model_name = "Salesforce/codegen25-7b-mono_P"
    pipe = pipeline("text-generation", model=model_name, trust_remote_code=True)
    print("Model loaded successfully!")
    return pipe

def generate_code(pipe, prompt, max_length=200):
    output = pipe(prompt, max_length=max_length, temperature=0.7, do_sample=True, top_p=0.95)
    return output[0]['generated_text']

def main():
    # Загружаем модель
    pipe = load_model()
    
    print("\nCodeGen25-7B Mono Interactive Mode")
    print("Enter your code prompt (or 'quit' to exit)")
    print("Example prompts:")
    print("- def calculate_fibonacci(n):")
    print("- class User:")
    print("- def sort_array(arr):")
    print("- async def fetch_data(url):")
    
    while True:
        try:
            # Получаем промпт от пользователя
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
                
            if not prompt:
                continue
            
            # Генерируем код
            print("\nGenerating code...")
            generated_code = generate_code(pipe, prompt)
            print("\nGenerated code:")
            print(generated_code)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 