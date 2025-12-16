import streamlit as st
from merged import load_model_and_tokenizer, generate_code_stream
from main import test_case   # your existing PDF generator

st.write("🚀 App started")
# ---------------- Load model ONCE ----------------
@st.cache_resource
def load_model():
    return load_model_and_tokenizer()

tokenizer, model = load_model()

# ---------------- Session state ----------------
for key in ["selected_language", "generated_code", "pdf_bytes", "user_input"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "selected_language" else None

# ---------------- UI helpers ----------------
def select_language(lang):
    st.session_state.selected_language = lang
    st.session_state.generated_code = ""
    st.session_state.pdf_bytes = ""
    st.session_state.user_input = ""

def reset():
    for k in st.session_state:
        st.session_state[k] = "" if k != "selected_language" else None

# ---------------- UI ----------------
st.title("Generate any code and get test case for it")

if st.session_state.selected_language is None:
    cols = st.columns(4)
    cols[0].button("Java", on_click=select_language, args=("Java",))
    cols[1].button("React.js", on_click=select_language, args=("React.js",))
    cols[2].button("Python", on_click=select_language, args=("Python",))
    cols[3].button("C++", on_click=select_language, args=("C++",))

else:
    lang = st.session_state.selected_language
    st.subheader(f"Selected Language: {lang}")
    st.button("Reset", on_click=reset)

    st.session_state.user_input = st.text_input(
        "Describe the task",
        value=st.session_state.user_input
    )

    if st.button("Generate Code"):
        if st.session_state.user_input.strip():
            st.session_state.generated_code = ""

            st.subheader("Generated Code")
            code_placeholder = st.empty()

            with st.spinner("Generating code..."):
                for token in generate_code_stream(
                    lang,
                    st.session_state.user_input,
                    tokenizer,
                    model
                ):
                    st.session_state.generated_code += token
                    code_placeholder.code(
                        st.session_state.generated_code
                    )

            # Generate test case PDF AFTER code generation
            st.session_state.pdf_bytes = test_case(
                st.session_state.generated_code
            )

        else:
            st.warning("Please enter a task")

    if st.session_state.pdf_bytes:
        st.download_button(
            "Download Test Cases PDF",
            st.session_state.pdf_bytes,
            file_name="test_cases.pdf",
            mime="application/pdf"
        )

