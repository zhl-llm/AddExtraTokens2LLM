import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_huggingface_token():
    """Retrieve the Hugging Face token from environment variables."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN is not set in environment variables.")
    return hf_token

def set_mirror_url():
    """Set the Hugging Face mirror URL."""
    mirror_url = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    os.environ["HF_ENDPOINT"] = mirror_url
    print(f"Using mirror URL: {mirror_url}")

def load_model_and_tokenizer(model_name, hf_token):
    """Load the model and tokenizer with specified configurations."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="cpu",
        trust_remote_code=True,
        force_download=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        do_sample=False,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )
    return model, tokenizer

def add_new_tokens(tokenizer, model, new_tokens):
    """Add new tokens to the tokenizer and resize the model embeddings."""
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"1. Added {num_added_tokens} new tokens to tokenizer.")

    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token embeddings
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        for i in range(len(new_tokens)):
            embedding_layer.weight[-(i+1)] = torch.mean(embedding_layer.weight[:-num_added_tokens], dim=0)

    return tokenizer, model

def save_model_and_tokenizer(model, tokenizer, tokenizer_path="./update_tokenizer", model_path="./updated_model"):
    """Save the updated tokenizer and model."""
    tokenizer.save_pretrained(tokenizer_path)
    model.save_pretrained(model_path)
    print(f"2. Saved updated tokenizer and model to {tokenizer_path} and {model_path}.")

def reload_model_and_tokenizer(tokenizer_path, model_path):
    """Reload the model and tokenizer from saved files."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        trust_remote_code=True,
        force_download=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        do_sample=False,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("3. Reloaded model and tokenizer.")
    return model, tokenizer

def generate_text(model, tokenizer, text, max_length=50):
    """Generate text using the trained model."""
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            use_cache=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """Main function to run the entire pipeline."""
    set_mirror_url()
    hf_token = get_huggingface_token()

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    model, tokenizer = load_model_and_tokenizer(model_name, hf_token)

    new_tokens = ["k8s-cluster", "kubernetes", "devops", "microservices", "containerization", "serverless", 
                  "service mesh", "observability", "gitops", "scalability"]
    tokenizer, model = add_new_tokens(tokenizer, model, new_tokens)

    save_model_and_tokenizer(model, tokenizer)

    model, tokenizer = reload_model_and_tokenizer("./update_tokenizer", "./updated_model")

    text = "今天天气很好，我想去"
    output_text = generate_text(model, tokenizer, text)
    print(output_text)

if __name__ == "__main__":
    main()
