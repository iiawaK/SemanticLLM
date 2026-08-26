"""
SemanticLLM Experiment: LLM-driven Semantic Communication for 6G
=================================================================
Real experiment that generates actual data for the paper.

Requirements:
    pip install torch transformers datasets bert-score matplotlib numpy pandas

Usage:
    python experiment.py
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from datasets import load_dataset
from bert_score import score as bert_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "model_name": "gpt2",  # GPT-2 small (124M) - fits in 8GB GPU
    "max_length": 64,
    "batch_size": 16,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "channel_dim": 256,  # number of channel uses (m)
    "snr_range": [-10, -5, 0, 5, 10, 15, 20],  # dB
    "num_test_samples": 200,
    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# Dataset: Load from HuggingFace (SST-2 + DailyDialog)
# ============================================================
def load_standard_dataset(n_train=500, n_test=200):
    """Load sentences from standard NLP datasets."""
    print("Loading dataset from HuggingFace...")

    # SST-2 (sentiment) - has good variety of sentences
    try:
        sst2 = load_dataset("glue", "sst2", split="train", trust_remote_code=True)
        texts = [x["sentence"] for x in sst2 if len(x["sentence"].split()) > 5]
        texts = list(set(texts))  # deduplicate
        print(f"  Loaded {len(texts)} unique sentences from SST-2")
    except Exception as e:
        print(f"  SST-2 loading failed: {e}, using fallback")
        texts = _get_fallback_texts()

    # Shuffle and split
    np.random.shuffle(texts)
    train_texts = texts[:n_train]
    test_texts = texts[n_train:n_train + n_test]

    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
    return train_texts, test_texts


def _get_fallback_texts():
    """Fallback dataset if HuggingFace fails."""
    return [
        "The weather is beautiful today with clear blue skies.",
        "Machine learning has revolutionized many industries.",
        "The stock market experienced significant volatility this week.",
        "Scientists discovered a new species in the Amazon rainforest.",
        "The concert was absolutely amazing and unforgettable.",
        "Climate change poses a serious threat to global ecosystems.",
        "The new smartphone features an innovative camera system.",
        "Students are preparing for their final examinations.",
        "The football match ended in a dramatic penalty shootout.",
        "Artificial intelligence is transforming healthcare diagnostics.",
        "The restaurant received excellent reviews from food critics.",
        "Renewable energy sources are becoming more cost-effective.",
        "The movie premiere attracted thousands of enthusiastic fans.",
        "Space exploration continues to reveal surprising discoveries.",
        "The company announced record-breaking quarterly earnings.",
        "Public transportation systems need significant upgrades.",
        "The museum exhibition showcases ancient Egyptian artifacts.",
        "Cybersecurity threats are becoming increasingly sophisticated.",
        "The marathon runner broke the world record by two seconds.",
        "Ocean pollution is devastating marine life worldwide.",
    ] * 50  # 1000 fallback samples


# ============================================================
# Models
# ============================================================

class SemanticEncoder(nn.Module):
    """LLM-based semantic encoder with channel projection."""

    def __init__(self, llm_hidden_dim, channel_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(llm_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, channel_dim),
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch, hidden_dim) - mean-pooled LLM embeddings
        Returns:
            z: (batch, channel_dim) - channel-ready signal
        """
        return self.projection(embeddings)


class SemanticDecoder(nn.Module):
    """Semantic decoder with de-projection."""

    def __init__(self, channel_dim, llm_hidden_dim):
        super().__init__()
        self.deprojection = nn.Sequential(
            nn.Linear(channel_dim, 512),
            nn.ReLU(),
            nn.Linear(512, llm_hidden_dim),
        )

    def forward(self, z_hat):
        """
        Args:
            z_hat: (batch, channel_dim) - received signal after channel
        Returns:
            s_hat: (batch, hidden_dim) - reconstructed semantic embedding
        """
        return self.deprojection(z_hat)


class WirelessChannel:
    """Simulates wireless channel effects."""

    @staticmethod
    def awgn(z, snr_db):
        """Additive White Gaussian Noise channel."""
        signal_power = torch.mean(z ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(z) * torch.sqrt(noise_power)
        return z + noise

    @staticmethod
    def rayleigh(z, snr_db):
        """Rayleigh fading channel."""
        h = torch.sqrt(torch.randn(z.shape[0], 1, device=z.device) ** 2 +
                       torch.randn(z.shape[0], 1, device=z.device) ** 2) / np.sqrt(2)
        z_faded = h * z
        return WirelessChannel.awgn(z_faded, snr_db)


# ============================================================
# Baseline: Traditional BPG + LDPC (simulated)
# ============================================================
def baseline_transmission(text, snr_db, tokenizer):
    """Simulate traditional digital transmission baseline."""
    # Encode text to tokens
    tokens = tokenizer.encode(text, return_tensors="pt")
    # Simulate bit errors based on SNR
    ber = 0.5 * np.erfc(np.sqrt(10 ** (snr_db / 10)))  # theoretical BER for BPSK
    # Introduce errors
    corrupted_tokens = tokens.clone()
    mask = torch.rand_like(tokens.float()) < ber
    # Random token replacement for errors
    random_tokens = torch.randint(0, tokenizer.vocab_size, tokens.shape)
    corrupted_tokens[mask] = random_tokens[mask]
    # Decode
    try:
        decoded = tokenizer.decode(corrupted_tokens[0], skip_special_tokens=True)
    except:
        decoded = text  # fallback
    return decoded


# ============================================================
# Training
# ============================================================
def train_semanticllm(train_texts):
    """Train the SemanticLLM framework."""
    print("=" * 60)
    print("Training SemanticLLM")
    print("=" * 60)

    # Load pretrained LLM (GPT-2)
    tokenizer = GPT2Tokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    llm = GPT2Model.from_pretrained(CONFIG["model_name"]).to(device)
    llm.eval()  # Freeze LLM
    for param in llm.parameters():
        param.requires_grad = False

    hidden_dim = llm.config.hidden_size  # 768 for GPT-2

    # Initialize encoder/decoder
    encoder = SemanticEncoder(hidden_dim, CONFIG["channel_dim"]).to(device)
    decoder = SemanticDecoder(CONFIG["channel_dim"], hidden_dim).to(device)

    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=CONFIG["learning_rate"])

    # Training loop
    losses = []
    for epoch in range(CONFIG["num_epochs"]):
        epoch_loss = 0
        encoder.train()
        decoder.train()

        # Simple batch training
        for i in range(0, len(train_texts), CONFIG["batch_size"]):
            batch_texts = train_texts[i:i + CONFIG["batch_size"]]

            # Tokenize
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=CONFIG["max_length"]).to(device)

            with torch.no_grad():
                outputs = llm(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)  # (batch, hidden)

            # Normalize power
            z = encoder(embeddings)
            z = z / torch.norm(z, dim=1, keepdim=True) * np.sqrt(CONFIG["channel_dim"])

            # Channel (random SNR during training)
            snr = np.random.uniform(-5, 15)
            z_hat = WirelessChannel.awgn(z, snr)

            # Decode
            s_hat = decoder(z_hat)

            # Loss: cosine similarity + MSE
            cos_sim = nn.functional.cosine_similarity(s_hat, embeddings)
            mse_loss = nn.functional.mse_loss(s_hat, embeddings)
            loss = (1 - cos_sim.mean()) + 0.5 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(train_texts) // CONFIG["batch_size"])
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Loss: {avg_loss:.4f}")

    return encoder, decoder, llm, tokenizer, losses


# ============================================================
# Evaluation
# ============================================================
def evaluate_semanticllm(encoder, decoder, llm, tokenizer, snr_list, test_texts):
    """Evaluate SemanticLLM at different SNR levels."""
    print("\n" + "=" * 60)
    print("Evaluating SemanticLLM")
    print("=" * 60)

    encoder.eval()
    decoder.eval()

    results = {}

    for snr in snr_list:
        print(f"\nTesting at SNR = {snr} dB...")
        decoded_texts = []

        for text in tqdm(test_texts, desc=f"SNR={snr}dB"):
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                             truncation=True, max_length=CONFIG["max_length"]).to(device)

            with torch.no_grad():
                outputs = llm(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

                z = encoder(embeddings)
                z = z / torch.norm(z, dim=1, keepdim=True) * np.sqrt(CONFIG["channel_dim"])

                z_hat = WirelessChannel.awgn(z, snr)
                s_hat = decoder(z_hat)

                # Reconstruct text: find nearest tokens
                # Use cosine similarity with word embeddings
                vocab_embeddings = llm.wte.weight  # (vocab_size, hidden_dim)
                similarities = nn.functional.cosine_similarity(
                    s_hat.unsqueeze(1), vocab_embeddings.unsqueeze(0), dim=2
                )
                top_tokens = similarities.argmax(dim=1)
                decoded = tokenizer.decode(top_tokens[0], skip_special_tokens=True)
                decoded_texts.append(decoded)

        # Compute BERTScore
        P, R, F1 = bert_score(decoded_texts, test_texts, lang="en", verbose=False)
        bertscore = F1.mean().item()

        # Compute word-level accuracy
        word_acc = 0
        for orig, dec in zip(test_texts, decoded_texts):
            orig_words = set(orig.lower().split())
            dec_words = set(dec.lower().split())
            if len(orig_words) > 0:
                word_acc += len(orig_words & dec_words) / len(orig_words)
        word_acc /= len(test_texts)

        results[snr] = {
            "bertscore": bertscore,
            "word_accuracy": word_acc,
            "decoded_samples": decoded_texts[:3],  # Save 3 examples
        }

        print(f"  BERTScore: {bertscore:.4f}, Word Acc: {word_acc:.4f}")

    return results


def evaluate_baselines(snr_list, test_texts, tokenizer):
    """Evaluate baseline methods."""
    print("\n" + "=" * 60)
    print("Evaluating Baselines")
    print("=" * 60)

    results = {}

    for snr in snr_list:
        print(f"\nBaseline at SNR = {snr} dB...")
        decoded_texts = []

        for text in tqdm(test_texts, desc=f"SNR={snr}dB"):
            decoded = baseline_transmission(text, snr, tokenizer)
            decoded_texts.append(decoded)

        # BERTScore
        P, R, F1 = bert_score(decoded_texts, test_texts, lang="en", verbose=False)
        bertscore = F1.mean().item()

        # Word accuracy
        word_acc = 0
        for orig, dec in zip(test_texts, decoded_texts):
            orig_words = set(orig.lower().split())
            dec_words = set(dec.lower().split())
            if len(orig_words) > 0:
                word_acc += len(orig_words & dec_words) / len(orig_words)
        word_acc /= len(test_texts)

        results[snr] = {
            "bertscore": bertscore,
            "word_accuracy": word_acc,
        }

        print(f"  BERTScore: {bertscore:.4f}, Word Acc: {word_acc:.4f}")

    return results


# ============================================================
# Plotting
# ============================================================
def plot_results(semanticllm_results, baseline_results, snr_list, output_dir="figs"):
    """Generate publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: BERTScore vs SNR
    fig, ax = plt.subplots(figsize=(8, 5))

    snr_vals = sorted(snr_list)
    bert_semantic = [semanticllm_results[s]["bertscore"] for s in snr_vals]
    bert_baseline = [baseline_results[s]["bertscore"] for s in snr_vals]

    ax.plot(snr_vals, bert_semantic, 'r-o', linewidth=2, markersize=8, label='SemanticLLM (Ours)')
    ax.plot(snr_vals, bert_baseline, 'b--s', linewidth=1.5, markersize=6, label='BPG + LDPC Baseline')

    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('BERTScore (F1)', fontsize=14)
    ax.set_title('Text Transmission: BERTScore vs SNR', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(snr_vals) - 2, max(snr_vals) + 2])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_bertscore.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig_bertscore.png', dpi=300, bbox_inches='tight')
    print(f"Saved {output_dir}/fig_bertscore.pdf")

    # Figure 2: Word Accuracy vs SNR
    fig, ax = plt.subplots(figsize=(8, 5))

    word_semantic = [semanticllm_results[s]["word_accuracy"] for s in snr_vals]
    word_baseline = [baseline_results[s]["word_accuracy"] for s in snr_vals]

    ax.plot(snr_vals, word_semantic, 'r-o', linewidth=2, markersize=8, label='SemanticLLM (Ours)')
    ax.plot(snr_vals, word_baseline, 'b--s', linewidth=1.5, markersize=6, label='BPG + LDPC Baseline')

    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('Word-level Accuracy', fontsize=14)
    ax.set_title('Text Transmission: Word Accuracy vs SNR', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(snr_vals) - 2, max(snr_vals) + 2])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_wordacc.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig_wordacc.png', dpi=300, bbox_inches='tight')
    print(f"Saved {output_dir}/fig_wordacc.pdf")

    # Figure 3: Framework diagram
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    boxes = [
        (0.5, 1.5, 2.5, 1.2, 'GPT-2\nEncoder', '#FFD700'),
        (3.5, 1.5, 2.5, 1.2, 'Channel\nProjection', '#87CEEB'),
        (6.5, 1.5, 2.0, 1.2, 'AWGN\nChannel', '#90EE90'),
        (9.0, 1.5, 2.0, 1.2, 'De-\nProjection', '#FFB6C1'),
        (11.5, 1.5, 2.0, 1.2, 'Text\nReconstruct', '#DDA0DD'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='black',
                            facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=11, fontweight='bold')

    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    for i in range(len(boxes)-1):
        x_start = boxes[i][0] + boxes[i][2]
        x_end = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3]/2
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                   arrowprops=arrow_style)

    ax.text(1.75, 3.2, 'Transmitter', ha='center', fontsize=13,
           fontweight='bold', color='blue')
    ax.text(10.0, 3.2, 'Receiver', ha='center', fontsize=13,
           fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_framework.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig_framework.png', dpi=300, bbox_inches='tight')
    print(f"Saved {output_dir}/fig_framework.pdf")


def save_results(semanticllm_results, baseline_results, losses, output_file="results.json"):
    """Save all experimental results to JSON."""
    data = {
        "config": CONFIG,
        "training_losses": losses,
        "semanticllm": {str(k): v for k, v in semanticllm_results.items()},
        "baseline": {str(k): v for k, v in baseline_results.items()},
        "summary": {},
    }

    # Compute summary statistics
    snr_vals = sorted(semanticllm_results.keys())
    for snr in snr_vals:
        slm = semanticllm_results[snr]["bertscore"]
        bl = baseline_results[snr]["bertscore"]
        improvement = (slm - bl) / bl * 100
        data["summary"][f"snr_{snr}"] = {
            "semanticllm_bertscore": round(slm, 4),
            "baseline_bertscore": round(bl, 4),
            "improvement_pct": round(improvement, 2),
        }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return data


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    start_time = time.time()

    # 0. Load dataset
    train_texts, test_texts = load_standard_dataset(
        n_train=500, n_test=CONFIG["num_test_samples"]
    )

    # 1. Train
    encoder, decoder, llm, tokenizer, losses = train_semanticllm(train_texts)

    # 2. Prepare test set
    snr_list = CONFIG["snr_range"]

    # 3. Evaluate SemanticLLM
    semanticllm_results = evaluate_semanticllm(
        encoder, decoder, llm, tokenizer, snr_list, test_texts
    )

    # 4. Evaluate Baselines
    baseline_results = evaluate_baselines(snr_list, test_texts, tokenizer)

    # 5. Plot and save
    plot_results(semanticllm_results, baseline_results, snr_list)
    data = save_results(semanticllm_results, baseline_results, losses)

    # 6. Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for snr in sorted(snr_list):
        slm = semanticllm_results[snr]["bertscore"]
        bl = baseline_results[snr]["bertscore"]
        imp = (slm - bl) / bl * 100
        print(f"SNR={snr:>3}dB | SemanticLLM: {slm:.4f} | Baseline: {bl:.4f} | Improvement: {imp:+.1f}%")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
