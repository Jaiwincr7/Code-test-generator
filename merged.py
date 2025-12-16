import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
from threading import Thread

MODEL_PATH = "https://drive.google.com/drive/folders/1WQrL__nkL52456C37Z4JC43hmsWISVa0?usp=sharing"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32,   # IMPORTANT: faster on CPU
    )

    model.eval()
    return tokenizer, model


def build_prompt(lang, task):
    # Keep prompt SIMPLE for speed
    return f"""You are a coding assistant.
            Write {lang} code for the following task:

            {task}

            Code:
            """


def generate_code_stream(lang, user_input, tokenizer, model):
    prompt = build_prompt(lang, user_input)
    inputs = tokenizer(prompt, return_tensors="pt")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=250,     
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Run generation in background thread
    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )
    thread.start()

    # Yield tokens as they arrive
    for token in streamer:
        yield token

