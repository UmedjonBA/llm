from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def load_model():
    print("Loading CodeGen25 7B Instruct model...")
    model_name = "Salesforce/codegen25-7b-instruct_P"
    
    # Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully!")
    return model, tokenizer

def generate_code(model, tokenizer, prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_with_pipeline(prompt, max_length=200):
    pipe = pipeline("text-generation", model="Salesforce/codegen25-7b-instruct_P")
    output = pipe(prompt, max_length=max_length, temperature=0.7, do_sample=True, top_p=0.95)
    return output[0]['generated_text']

def main():
    # Загружаем модель
    model, tokenizer = load_model()
    
    print("\nCodeGen25-7B Instruct Interactive Mode")
    print("Enter your code prompt (or 'quit' to exit)")
    print("Example prompts:")
    print("- def calculate_fibonacci(n):")
    print("- class User:")
    print("- def sort_array(arr):")
    print("- async def fetch_data(url):")
    print("\nChoose generation method:")
    print("1. Direct model")
    print("2. Pipeline")
    
    while True:
        try:
            # Получаем промпт от пользователя
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
                
            if not prompt:
                continue
            
            # Выбираем метод генерации
            method = input("Choose generation method (1/2): ").strip()
            
            # Генерируем код
            print("\nGenerating code...")
            if method == "2":
                generated_code = generate_with_pipeline(prompt)
            else:
                generated_code = generate_code(model, tokenizer, prompt)
                
            print("\nGenerated code:")
            print(generated_code)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 