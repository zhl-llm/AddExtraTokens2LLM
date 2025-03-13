import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 从环境变量中读取 Hugging Face token
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set in environment variables.")

mirror_url = os.getenv("HF_ENDPOINT", "https://huggingface.co")
os.environ["HF_ENDPOINT"] = mirror_url
print(f"Using mirror URL: {mirror_url}")

model_name = "openbmb/MiniCPM-2B-sft-bf16"
# model_name = "ussipan/SipanGPT-0.1-Llama-3.2-1B-GGUF"
# 下载模型文件
# from modelscope import snapshot_download
# model_name = snapshot_download(model_name, cache_dir=model_dir)

# 关键修改 1：禁用所有缓存和优化选项
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # model_dir,
    token=hf_token,
    device_map="cpu",
    trust_remote_code=True,
    force_download=True,
    torch_dtype=torch.bfloat16,  # 半精度减少内存占用
    # torch_dtype=torch.float16,    # float16 减少内存占用
    use_cache=False,             # 全局禁用缓存
    do_sample=False,            # 关闭采样（避免触发动态逻辑）
    low_cpu_mem_usage=True       # 防止内存碎片化
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True
)

# 定义新的 tokens（比如特定领域的tokens）
new_tokens = ["k8s-cluster", "kubernetes", "devops", "microservices", "containerization", "serverless", "service mesh", "observability", "gitops", "scalability"]

# 给 tokenizer 添加新的 tokens
num_added_tokens = tokenizer.add_tokens(new_tokens)
print(f"1. 添加 {num_added_tokens} 个新的 tokens 到 tokenizer 中.")

# 重置模型的 token 嵌入此表长度来保持因新增 tokens 导致的变化
model.resize_token_embeddings(len(tokenizer))

# 正确初始化 token 嵌入
embedding_layer = model.get_input_embeddings()
with torch.no_grad():
    for i in range(len(new_tokens)):
        embedding_layer.weight[-(i+1)] = torch.mean(embedding_layer.weight[:-num_added_tokens], dim=0)

# 保存新增过 tokens 的 tokenizer 和 模型
tokenizer.save_pretrained("./update_tokenizer")
model.save_pretrained("./updated_model")
print(f"2. 将新的 tokenizer 和模型分别保存到当前目录的 update_tokenizer 和 updated_model 文件夹下")

# 重新加载新的模型
model = AutoModelForCausalLM.from_pretrained(
    "./updated_model",
    device_map="cpu",
    trust_remote_code=True,
    force_download=True,
    torch_dtype=torch.bfloat16,  # 半精度减少内存占用
    # torch_dtype=torch.float16,    # float16 减少内存占用
    use_cache=False,             # 全局禁用缓存
    do_sample=False,            # 关闭采样（避免触发动态逻辑）
    low_cpu_mem_usage=True       # 防止内存碎片化
)
tokenizer = AutoTokenizer.from_pretrained("./update_tokenizer", trust_remote_code=True)
print(f"3. 从上面对应路径重新加载模型")

text = "今天天气很好，我想去"
inputs = tokenizer(text, return_tensors="pt")

# 关键修改 2：手动设置 max_length 和缓存参数
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # 显式设置终止符
        no_repeat_ngram_size=2,               # 避免缓存机制介入
        use_cache=False                       # 二次确认禁用
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
