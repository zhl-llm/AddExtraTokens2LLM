# AddExtraTokens2LLM
This an example for adding extra general tokens to pre-trained LLM before its fine tuning.

## Why do we need to add extra tokens to a pre-trained LLM

- Introducing Domain-Specific Vocabulary

	If need to fine-tune on a specialized domain (e.g., medicine, law, finance), some new terms, abbreviations or technical jargon that the base model doesn't recognize need to be introduced. Adding new tokens helps the model better understand and generate text relevant to that field.

- Handling Out-of-Vocabulary Words

	Pre-trained LLMs use subword tokenization (like BPE or WordPiece), but some words might still be broken into inefficient subwords. Adding extra tokens can improve efficiency and accuracy in text generation. Such as `ChatGPT` can be splitted into `Chat` and `GPT` which leads to semantic fragmentation.

- Adapting Multilingual Expansion

	If fine-tuning for a new language not well-covered in the original training data, adding extra tokens for unique words or characters can improve performance.

- Introducing Custom Formatting Tokens, Instruction Tuning and Prompt Engineering

	If need to introduce a specific structure in the fine-tuning data (e.g., markdown formatting, XML tags, or placeholders for dialogue roles), new tokens can help the model learn the desired structure more effectively. Besides that, Some fine-tuned models use special tokens like `<question>`, `<answer>`, or `<context>` to guide responses. These extra tokens help condition the model on structured inputs. It's better need to introduce special tokens in these situations, rather than general tokens. And this example is just an example for general tokens example.

  In summary, adding extra tokens helps pre-trained LLM adapt more efficiently to domain-specific requirements during fine-tuning, reducing the need for extensive training data. It is also a key technique for balancing fine-tuning effectiveness and computational cost.

## How to lanch this example?

  There are 2 methods to lanch this example:

### 1. Lanch it in local python virtual environment

- Create a virtual environment and activate it

```sh
$ python3 -m venv myenv
$ source myenv/bin/activate
```

- Upgrade and install necessary libraries via pip

```sh
$ pip install --upgrade pip setuptools
$ pip install --no-cache-dir -r requirements.txt
```

- Run this example

```sh
export HUGGINGFACE_TOKEN="Your Huggingface Token"
$ python llm_loader.py
```

* Note: Please execute below commands to use Mota Community or hf mirror webesite if you can not access huggingface:

```sh
export GPTQMODEL_USE_MODELSCOPE="True"
export HF_ENDPOINT="https://hf-mirror.com"
```

### 2. Lanch it in container

- Build docker image of this example

```sh
$ docker build -t <IMAGE_NAME> .
```

* Note: Please replace `<IMAGE_NAME>` with the really image name.

- Run the container based on this docker image

```sh
$ docker run -e HUGGINGFACE_TOKEN="Your Huggingface Token"  --rm  --name <CONTAINER_NAME> <IMAGE_NAME>
```

* Note: Please replace `<CONTAINER_NAME>` and `<IMAGE_NAME>` with the really container and image name.