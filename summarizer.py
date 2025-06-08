from transformers import AutoTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    def summarize(self, text, max_tokens=200):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = self.model.generate(inputs["input_ids"], max_new_tokens=max_tokens)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
