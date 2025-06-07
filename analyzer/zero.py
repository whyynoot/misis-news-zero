import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ZeroShotClassifier:
    def __init__(self, model_checkpoint='cointegrated/rubert-base-cased-nli-threeway', target_label='entailment', use_cuda=True):
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        
        # Load model onto CUDA if available and requested
        if use_cuda and torch.cuda.is_available():
            self.model.cuda()
        
        # Set the target label for classification
        self.target_label = target_label

    def predict(self, text, label_texts, normalize=True):
        tokens = self.tokenizer([text] * len(label_texts), label_texts, truncation=True, return_tensors='pt', padding=True)
        
        with torch.inference_mode():
            logits = self.model(**tokens.to(self.model.device)).logits
            probabilities = torch.softmax(logits, -1)
        
        proba = probabilities[:, self.model.config.label2id[self.target_label]].cpu().numpy()
        
        if normalize:
            proba /= sum(proba)
        
        return proba