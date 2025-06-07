from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Summarizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

    def summarize(self, text, max_tokens=150):
        prompt = "Summarize: " + text
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
