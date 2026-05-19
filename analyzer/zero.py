import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ZeroShotClassifier:
    def __init__(
        self,
        model_checkpoint="cointegrated/rubert-base-cased-nli-threeway",
        target_label="entailment",
        use_cuda=True,
        batch_size=8,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.batch_size = max(int(batch_size), 1)

        if use_cuda and torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()
        self.target_label = target_label

    def _predict_matrix(self, texts, label_texts, normalize=True):
        if not texts:
            return np.empty((0, len(label_texts)), dtype=float)

        premises = []
        hypotheses = []
        for text in texts:
            premises.extend([text] * len(label_texts))
            hypotheses.extend(label_texts)

        tokens = self.tokenizer(
            premises,
            hypotheses,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )

        with torch.inference_mode():
            logits = self.model(**tokens.to(self.model.device)).logits
            probabilities = torch.softmax(logits, -1)

        entailment_index = self.model.config.label2id[self.target_label]
        matrix = probabilities[:, entailment_index].reshape(len(texts), len(label_texts)).cpu().numpy()

        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(
                matrix,
                row_sums,
                out=np.full_like(matrix, 1 / len(label_texts)),
                where=row_sums != 0,
            )

        return matrix

    def predict_batch(self, texts, label_texts, normalize=True):
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            outputs.append(self._predict_matrix(chunk, label_texts, normalize=normalize))

        if not outputs:
            return np.empty((0, len(label_texts)), dtype=float)
        return np.vstack(outputs)

    def predict(self, text, label_texts, normalize=True):
        return self.predict_batch([text], label_texts, normalize=normalize)[0]
