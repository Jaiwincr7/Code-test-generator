from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from fpdf import FPDF
import re

def extract_code(text):
    """
    Extract fenced code blocks. If the model fails, fallback to raw text.
    """
    # 1. Look for fenced code blocks (```language ... ```)
    fenced = re.findall(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
    if fenced and fenced[0].strip():
        return fenced[0].strip()

    # 2. Fallback: attempt to extract code based on common syntax patterns
    patterns = [
        r"(class\s+\w+.*?})",
        r"(function\s+\w+.*?})",
        r"(def\s+\w+.*?)(?=\n\n|\Z)",
        r"(public\s+class\s+\w+.*?})"
    ]
    for p in patterns:
        found = re.findall(p, text, flags=re.DOTALL)
        if found:
            return found[0].strip()

    # 3. Last fallback: return cleaned text (removes the <|assistant|> tag)
    text = text.replace("<|assistant|>", "").strip()
    return text




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

# ------------------------------
# 1. CODE GENERATION CHAIN
# ------------------------------
def generate_code(lang,input):

    system_message = (
        "You are an expert software engineer. Generate ONLY the code. Do NOT explain the code. "
        f"Do NOT repeat my instructions. Output only valid {lang} code. Stop after completing the code."
    )
    template = f"""<|system|>\n{system_message}\n<|end_of_system|>\n<|user|>\nUser request: {input}\n<|end_of_user|>\n<|assistant|>"""

    generate_prompt = PromptTemplate.from_template(template)

    generate_chain = generate_prompt | llm | StrOutputParser()
    code_response = generate_chain.invoke({}) 

    return extract_code(code_response)


# ------------------------------
# 2. CODE REVIEW + FIX CHAIN
# ------------------------------
def fix_code(generated_code):
    check_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert senior software engineer. Fix ALL issues (bugs, bad practices, inefficiencies, missing edge cases). Do NOT explain the fixes. Generate ONLY the corrected full code.",
            ),
            (
                "user",
                "Here is the code to fix:\n{code}",
            ),
        ]
    )

    check_chain = check_prompt | llm | StrOutputParser()

    # Invoke the chain using the dictionary key 'code'
    fixed_code_response = check_chain.invoke({"code": generated_code})
    
    # Extract the code block from the LLM's raw response
    code = extract_code(fixed_code_response)
    
    return code


# # ------------------------------
# # 3. TEST CASE GENERATION CHAIN
# # ------------------------------
# # ------------------------------
# # 3. TEST CASE GENERATION CHAIN
# # ------------------------------
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