from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from fpdf import FPDF
import re

model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
    offload_folder="./offload"
)
stop_tokens = ["<|end_of_text|>", "<|end_of_user|>"]

# Wrap Transformers pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={
        "max_new_tokens": 4096,
        "do_sample": True,
        "temperature": 0.2,
        "repetition_penalty": 1.05,
        "eos_token_id": tokenizer.eos_token_id,
    }
)

def test_case(code):
    test_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert QA engineer.
    STRICTLY follow these rules for your output:
    - Generate EXACTLY 10 numbered test cases (1–5 functional, 6–10 edge cases).
    - Output ONLY the numbered list.
    - DO NOT include explanations, headers, filler text, or markdown.
    - Each test MUST be a single, concise sentence.
    - Begin your response immediately with '1. '""", # Slightly relaxed constraints
            ),
            (
                "user",
                "Generate test cases for the following code:\n{code}",
            ),
        ]
    )

    test_chain = test_prompt | llm | StrOutputParser()
    test_cases = test_chain.invoke({"code": code})

    print("\nGenerated Test Cases (Raw):\n", test_cases)

    # Aggressive cleaning
    test_cases = re.sub(r"```.*?```", "", test_cases, flags=re.DOTALL)
    test_cases = re.sub(r"```", "", test_cases)
    test_cases = test_cases.strip()

    # --- ADD THIS CHECK ---
    if not test_cases:
        test_cases = "Error: Test case generation failed or returned empty content."
    
    print("\nGenerated Test Cases (Cleaned):\n", test_cases)
    # -----------------------

    # Encoding step remains the same for FPDF compatibility
    safe_text = test_cases.encode("latin-1", "ignore").decode("latin-1")

    # If the safe_text is still empty, FPDF will produce an empty PDF
    # It's better to verify the content being passed to FPDF
    
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # You can set a title to ensure the PDF isn't blank
    pdf.multi_cell(0, 10, txt="--- Generated Test Cases ---", align='C')
    pdf.multi_cell(0, 10, txt=safe_text)
    
    # Use output as a bytes object without saving to disk first
    file = pdf.output(dest='S').encode('latin-1') 
    
    return file