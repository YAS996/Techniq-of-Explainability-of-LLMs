# Import necessary libraries
import torch
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from memory_profiler import memory_usage
import os
import tracemalloc
from sklearn.feature_extraction.text import TfidfVectorizer

# Visualizes the attention matrix of a transformer model
def visualize_attention(attention_layers, tokens, sample_id, layer=-1, head_mode='mean'):
    sns.set(font_scale=0.8)
    max_tokens = min(30, len(tokens))  # Limit to 30 tokens for readability
    tokens = tokens[:max_tokens]

    selected_layer = attention_layers[layer][0]  # Get attention from selected layer
    if head_mode == 'mean':
        # Mean over all attention heads
        attn_matrix = selected_layer[:, :max_tokens, :max_tokens].mean(dim=0).cpu().detach().numpy()
        title = f"Mean Attention - Layer {layer}"
    else:
        # Visualize a specific head
        head_idx = int(head_mode)
        attn_matrix = selected_layer[head_idx, :max_tokens, :max_tokens].cpu().detach().numpy()
        title = f"Attention Head {head_idx} - Layer {layer}"

    # Measure memory and runtime for visualization
    tracemalloc.start()
    start_time = time.time()

    # Plot attention as a heatmap
    plt.figure(figsize=(min(20, len(tokens)*0.6), 10))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"attention_maps/sample_{sample_id}.png")  # Save the figure
    plt.close()

    current, peak = tracemalloc.get_traced_memory()
    elapsed = (time.time() - start_time) * 1000  # Execution time in ms
    tracemalloc.stop()

    return {
        "viz_runtime_ms": round(elapsed, 2),
        "viz_memory_kb": round(peak / 1024, 2)
    }

# Compares TF-IDF keywords with tokens used in attention to compute precision, recall, F1
def compare_tfidf_attention(tfidf_words, attention_tokens, attention_scores, sample_id):
    tp = [word for word in tfidf_words if word in attention_tokens]  # True positives
    fp = [word for word in attention_tokens if word not in tfidf_words]  # False positives
    fn = [word for word in tfidf_words if word not in attention_tokens]  # False negatives

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1_score

# Runs prediction, attention visualization, and TF-IDF comparison on a single sample
def run_sample(text, true_label, sample_id, model, tokenizer, tfidf_features, device):
    # Tokenize and send to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_label = torch.argmax(outputs.logits, dim=1).item()  # Model prediction
    correct = int(predicted_label == true_label)  # Check correctness
    attention = outputs.attentions  # Get attention tensors
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Decode token ids

    tfidf_tokens = tfidf_features  # TF-IDF top words

    # Compare attention tokens to TF-IDF tokens
    tp, fp, fn, precision, recall, f1_score = compare_tfidf_attention(tfidf_tokens, tokens, attention, sample_id)
    viz_stats = visualize_attention(attention, tokens, sample_id, layer=-1, head_mode='mean')

    return {
        "length": len(tokens),
        "correct": correct,
        "viz_runtime_ms": viz_stats["viz_runtime_ms"],
        "viz_memory_kb": viz_stats["viz_memory_kb"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "text": text,
        "label": true_label,
        "prediction": predicted_label
    }

# Main function to evaluate and visualize attention on IMDB test data
def main():
    # Load pretrained sentiment classification model
    model_name = "textattack/bert-base-uncased-imdb"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        output_attentions=True,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load the IMDB dataset (first 100 test samples)
    dataset = load_dataset("imdb", split="test[:100]")
    samples = dataset.select(range(len(dataset)))

    # Compute TF-IDF on the full text corpus
    corpus = [sample['text'] for sample in samples]
    vectorizer = TfidfVectorizer(max_features=100)
    X_tfidf = vectorizer.fit_transform(corpus)
    tfidf_features = vectorizer.get_feature_names_out()

    os.makedirs("attention_maps", exist_ok=True)  # Create output directory

    results = []
    # Loop through each sample and evaluate
    for i, sample in enumerate(samples):
        print(f"🔍 Running sample {i+1}/{len(samples)}")
        try:
            res = run_sample(sample["text"], int(sample["label"]), i, model, tokenizer, tfidf_features, device)
            results.append(res)
        except Exception as e:
            print(f"Error in sample {i}: {e}")

    # Print metrics for each sample
    print("\nPerformance Analysis:")
    for r in results:
        print(f"✓ Sample {results.index(r)} | Length: {r['length']}, Correct: {r['correct']}, "
              f"Visualization Runtime: {r['viz_runtime_ms']} ms, Memory: {r['viz_memory_kb']} KB, "
              f"Precision: {r['precision']:.2f}, Recall: {r['recall']:.2f}, F1: {r['f1_score']:.2f}")

    # Print average performance across all samples
    if results:
        avg_precision = np.mean([r["precision"] for r in results])
        avg_recall = np.mean([r["recall"] for r in results])
        avg_f1 = np.mean([r["f1_score"] for r in results])
        avg_runtime = np.mean([r["viz_runtime_ms"] for r in results])
        avg_memory = np.mean([r["viz_memory_kb"] for r in results])
        accuracy = np.mean([r["correct"] for r in results])

        print("\nAverages Over All Samples:")
        print(f"- Accuracy: {accuracy:.2f}")
        print(f"- Average Precision: {avg_precision:.2f}")
        print(f"- Average Recall: {avg_recall:.2f}")
        print(f"- Average F1 Score: {avg_f1:.2f}")
        print(f"- Avg. Visualization Time: {avg_runtime:.2f} ms")
        print(f"- Avg. Visualization Memory: {avg_memory:.2f} KB")

# Entry point for multiprocessing support (especially for Windows)
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
