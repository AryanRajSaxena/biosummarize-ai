import gradio as gr
from utils.pdf_parser import extract_text_from_pdf
from summarizer import Summarizer
from qa_engine import QABot

summarizer = Summarizer()

def process_pdf_and_qa(file, question):
    text = extract_text_from_pdf(file.name)
    summary = summarizer.summarize(text)
    bot = QABot(text.split("\n\n"))
    context = bot.retrieve_context(question)
    return summary, context

iface = gr.Interface(
    fn=process_pdf_and_qa,
    inputs=[
        gr.File(label="Upload Biotech Research Paper (PDF)"),
        gr.Textbox(label="Ask a Question")
    ],
    outputs=[
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Context Answer")
    ],
    title="🧬 BioSummarize.ai",
    description="Summarize research papers and answer questions using BioBERT"
)

if __name__ == "__main__":
    iface.launch()
