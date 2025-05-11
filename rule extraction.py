from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

def predict_bert(texts, model, tokenizer, batch_size=16):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

model_name = "textattack/bert-base-uncased-imdb"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

dataset = load_dataset("imdb", split="train[:1000]")
texts = dataset["text"]
labels = predict_bert(texts, model, tokenizer)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
X = vectorizer.fit_transform(texts)
from sklearn.tree import DecisionTreeClassifier, export_text

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X, labels)
rules = export_text(clf, feature_names=vectorizer.get_feature_names_out())
print("\n📜 Extracted rules from Decision Tree:\n")
print(rules)

with open("rules.txt", "w", encoding="utf-8") as f:
    f.write(rules)
import time

start_time = time.time()

labels = predict_bert(texts, model, tokenizer)

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")
import psutil

process = psutil.Process(os.getpid())
memory_before = process.memory_info().rss / 1024 / 1024  

clf.fit(X, labels)

memory_after = process.memory_info().rss / 1024 / 1024 
memory_usage = memory_after - memory_before
print(f"Memory Usage: {memory_usage} MB")
from sklearn.metrics import accuracy_score

accuracy_bert = accuracy_score(labels, predict_bert(texts, model, tokenizer))
accuracy_tree = accuracy_score(labels, clf.predict(X))

print(f"Accuracy BERT: {accuracy_bert}")
print(f"Accuracy Decision Tree: {accuracy_tree}")



