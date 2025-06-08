from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Summarizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

    def summarize(self, text, max_tokens=200):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
