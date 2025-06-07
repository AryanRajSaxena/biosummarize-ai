import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from summarizer import Summarizer
from qa_engine import QABot

st.title("🧬 BioSummarize.ai AI for Biotech Papers")

uploaded_file = st.file_uploader("Upload a research PDF", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf("temp.pdf")

    if st.button("Generate Summary"):
        summarizer = Summarizer()
        summary = summarizer.summarize(text)
        st.success("✅ Summary Generated:")
        st.write(summary)

    if st.button("Start Q&A"):
        st.info("⚙️ Indexing content...")
        chunks = text.split("\n\n")  # Basic chunking
        bot = QABot(chunks)
        question = st.text_input("Ask a question from the paper:")
        if question:
            answer_context = bot.retrieve_context(question)
            st.success("Context:")
            st.write(answer_context)
