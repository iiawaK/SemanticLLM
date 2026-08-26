"""
SemanticLLM Experiment (Lightweight v2)
=======================================
Fair comparison: both methods compress to same dimensionality.
SemanticLLM uses semantic-aware encoding, baseline uses random projection.

Requirements:
    pip install scikit-learn matplotlib numpy tqdm

Usage:
    python experiment_lite.py
"""

import os
import json
import time
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

CONFIG = {
    "embedding_dim": 512,       # TF-IDF dimension
    "channel_dim": 64,          # compressed dimension (same for both)
    "snr_range": [-10, -5, 0, 5, 10, 15, 20],
    "num_train": 500,
    "num_test": 200,
    "seed": 42,
}

np.random.seed(CONFIG["seed"])


# ============================================================
# Dataset (250 diverse sentences)
# ============================================================
SAMPLE_TEXTS = [
    "The weather is beautiful today with clear blue skies and gentle breeze.",
    "Machine learning has revolutionized many industries across the globe.",
    "The stock market experienced significant volatility this week of trading.",
    "Scientists discovered a new species deep in the Amazon rainforest region.",
    "The concert was absolutely amazing and truly an unforgettable experience.",
    "Climate change poses a serious threat to global ecosystems and biodiversity.",
    "The new smartphone features an innovative camera system powered by AI chips.",
    "Students are preparing intensively for their final examinations this semester.",
    "The football match ended in a dramatic and thrilling penalty shootout.",
    "Artificial intelligence is transforming healthcare diagnostics at a rapid pace.",
    "The restaurant received excellent reviews from professional food critics.",
    "Renewable energy sources are becoming more cost effective each passing year.",
    "The movie premiere attracted thousands of enthusiastic fans to the theater.",
    "Space exploration continues to reveal surprising and unexpected discoveries.",
    "The company announced record breaking quarterly earnings this morning report.",
    "Public transportation systems need significant infrastructure upgrades nationwide.",
    "The museum exhibition showcases rare ancient Egyptian artifacts and treasures.",
    "Cybersecurity threats are becoming increasingly sophisticated and dangerous.",
    "The marathon runner broke the world record by two seconds in the race.",
    "Ocean pollution is devastating marine life worldwide at an alarming rate.",
    "The new government policy aims to reduce carbon emissions by fifty percent.",
    "Virtual reality technology is reshaping the entire gaming industry landscape.",
    "The research team published groundbreaking findings in the Nature journal.",
    "Urban farming is gaining popularity in major cities around the world.",
    "The election results surprised many political analysts and media pundits.",
    "Quantum computing could potentially solve problems beyond classical computer reach.",
    "The charity gala successfully raised over one million dollars for education.",
    "Autonomous vehicles are being extensively tested in several pilot programs.",
    "The global pandemic accelerated the shift to remote work culture significantly.",
    "Biodiversity loss threatens the long term stability of food production systems.",
    "The novel won several prestigious literary awards during the past year.",
    "Five G networks enable exciting new applications in smart city development.",
    "The volcanic eruption forced the immediate evacuation of nearby villages.",
    "Online education platforms saw unprecedented growth in student enrollment numbers.",
    "The diplomatic negotiations resulted in a historic and lasting peace agreement.",
    "Genetic engineering offers promising new treatments for various rare diseases.",
    "The music festival featured talented artists from over thirty different countries.",
    "Supply chain disruptions continue to negatively affect global trade and commerce.",
    "The innovative architectural design won the prestigious international competition.",
    "Deep learning models require massive computational resources to train effectively.",
    "The humanitarian crisis demands an immediate and coordinated international response.",
    "Blockchain technology is being rapidly adopted across various financial sectors.",
    "The archaeological dig uncovered valuable Roman artifacts from ancient centuries.",
    "Renewable energy investments surpassed traditional fossil fuels for the first time.",
    "The basketball team clinched the championship title in an exciting overtime game.",
    "Satellite imagery helps scientists monitor deforestation in near real time.",
    "The startup successfully raised fifty million dollars in its Series B funding.",
    "Astronomers detected mysterious signals from a very distant neutron star system.",
    "The government launched a comprehensive nationwide digital literacy program.",
    "Electric vehicles accounted for twenty percent of total new car sales.",
    "The massive bridge construction project was completed ahead of schedule and budget.",
    "Renewable energy technologies are advancing at an unprecedented and remarkable pace.",
    "The museum opened a beautiful new wing dedicated to contemporary art exhibitions.",
    "Cloud computing has fundamentally transformed how businesses manage IT infrastructure.",
    "The powerful hurricane caused widespread and devastating damage along the coast.",
    "Researchers developed a promising new vaccine candidate for tropical diseases.",
    "The company stock price surged dramatically after announcing record breaking profits.",
    "Artificial neural networks can effectively approximate complex nonlinear mathematical functions.",
    "The city implemented comprehensive new recycling programs to reduce municipal waste.",
    "Wireless communication standards continue to evolve rapidly toward sixth generation.",
    "The experienced expedition team successfully reached the summit of Mount Everest.",
    "Semantic communication transmits meaningful information rather than raw bits efficiently.",
    "The new algorithm achieves state of the art performance on multiple benchmarks.",
    "Digital twins enable continuous real time monitoring of complex physical systems.",
    "The international conference brought together leading experts from around the world.",
    "Reconfigurable intelligent surfaces can significantly enhance wireless signal coverage.",
    "The museum extensive collection includes priceless artifacts from ancient civilizations.",
    "Federated learning enables privacy preserving collaborative model training at scale.",
    "The newly developed treatment showed very promising results in clinical trials.",
    "Edge computing significantly reduces latency for time critical real time applications.",
    "The orchestra performed a truly stunning rendition of Beethoven famous Ninth Symphony.",
    "Advanced graph neural networks capture complex relationships in network structured data.",
    "The government announced major new investments in renewable energy infrastructure.",
    "Natural language processing improved dramatically with the introduction of transformer models.",
    "The severe earthquake measured seven point two on the Richter scale magnitude.",
    "Autonomous drones are being increasingly used for agricultural crop monitoring tasks.",
    "The acclaimed film received widespread critical acclaim at international film festivals.",
    "Reinforcement learning agents can master complex and challenging game environments.",
    "The multinational company expanded its operations to three new countries this year.",
    "Quantum sensors offer unprecedented levels of precision for advanced medical imaging.",
    "The prolonged drought severely impacted agricultural crop production in the region.",
    "Modern computer vision systems can now recognize objects with remarkable high accuracy.",
    "The historic peace treaty was formally signed after months of intense negotiations.",
    "Bioinformatics combines advanced biology and computer science for genomic data analysis.",
    "The powerful new telescope will observe extremely distant galaxies with unprecedented clarity.",
    "Smart home devices are becoming increasingly integrated and fully autonomous in operation.",
    "The massive volcanic eruption disrupted international air travel across the continent.",
    "Transfer learning significantly reduces the need for large expensive labeled datasets.",
    "The population of several endangered species has shown encouraging signs of recovery.",
    "Blockchain technology ensures fully transparent and tamper proof digital record keeping.",
    "The active hurricane season was officially predicted to be above average in intensity.",
    "Modern natural language generation can produce coherent and remarkably fluent text.",
    "The newly constructed bridge successfully connects two previously isolated communities.",
    "Recurrent neural networks remain highly effective for complex sequence modeling tasks.",
    "The government announced strict new regulations for personal data privacy protection.",
    "Swarm intelligence algorithms cleverly mimic the collective behavior of social insects.",
    "The annual cultural festival attracted visitors from all over the entire country.",
    "Deep reinforcement learning effectively combines visual perception and strategic decision making.",
    "The large solar farm will provide clean renewable energy to thousands of homes.",
    "Attention mechanisms allow neural network models to focus on the most relevant information.",
    "The important archaeological site revealed fascinating evidence of early human civilization.",
    "Federated analytics enables valuable data insights without ever sharing raw private data.",
    "The innovative new drug recently received official approval from the regulatory authority.",
    "Knowledge graphs effectively represent real world entities and their complex relationships.",
    "The controversial dam construction project faced strong opposition from environmental groups.",
    "Generative adversarial networks can create impressively realistic synthetic training data.",
    "The international charity organization provided critical relief to disaster affected regions.",
    "Multimodal deep learning combines information from text, images, and audio data sources.",
    "The strict new regulation aims to dramatically reduce single use plastic waste.",
    "Convolutional neural networks continue to excel at complex image recognition tasks.",
    "The important research paper was accepted at a top tier prestigious conference.",
    "Zero shot learning enables accurate classification of previously unseen object categories.",
    "The powerful earthquake triggered an immediate tsunami warning for all coastal areas.",
    "Self supervised learning approaches significantly reduce dependency on expensive labeled data.",
    "The innovative company launched an exciting new product line targeting young consumers.",
    "Meta learning algorithms enable remarkably rapid adaptation to completely new tasks.",
    "The severe flooding caused extensive damage to critical infrastructure and farmland.",
    "Neural architecture search automates the optimal design of deep neural networks.",
    "The government signed a comprehensive bilateral trade agreement with neighboring countries.",
    "Contrastive learning effectively learns useful representations from large unlabeled datasets.",
    "The fast moving wildfire spread rapidly due to strong winds and drought.",
    "Graph attention networks intelligently weigh the contributions of different neighbor nodes.",
    "The national museum proudly unveiled a rare collection of medieval illuminated manuscripts.",
    "Prompt engineering techniques guide large language model behavior and output effectively.",
    "The powerful tropical cyclone caused widespread destruction across the island nation.",
    "Vision transformers successfully apply self attention mechanisms to small image patches.",
    "The forward thinking new policy encourages investment in green technology startups.",
    "Diffusion models generate remarkably high quality images through iterative denoising steps.",
    "The massive landslide completely blocked a major transportation route for several weeks.",
    "Retrieval augmented generation enhances model responses using relevant external knowledge sources.",
    "The spectacular solar eclipse was clearly visible across a wide geographic area.",
    "Sparse mixture of experts models efficiently scale model capacity without proportional compute.",
    "The severe typhoon caused significant disruption to daily life and transportation.",
    "Instruction tuning effectively aligns large language models with human intentions and values.",
    "The record breaking blizzard brought unprecedented snowfall to the northern regions.",
    "Multi task learning consistently improves generalization across related learning objectives.",
    "The extreme heat wave broke all previous temperature records across the continent.",
    "Chain of thought reasoning significantly improves complex mathematical problem solving abilities.",
    "The prolonged drought led to mandatory water restrictions in several major municipalities.",
    "Parameter efficient fine tuning dramatically reduces computational costs for model adaptation.",
    "The sudden avalanche seriously threatened several remote mountain communities this winter.",
    "Constitutional AI training ensures model outputs consistently align with core human values.",
    "The catastrophic flood displaced thousands of residents from their permanent homes.",
    "Sparse mixture of experts significantly reduces inference computation requirements in practice.",
    "The dangerous storm surge caused severe coastal flooding in many low lying areas.",
    "Advanced reward modeling effectively captures nuanced human preferences for model alignment.",
    "The powerful tornado caused severe damage to buildings and critical infrastructure.",
    "Knowledge distillation efficiently transfers knowledge from large to small compact models.",
    "The intense hailstorm caused significant damage to vehicles and residential property.",
    "Proximal policy optimization greatly improves reinforcement learning training stability and performance.",
    "The massive dust storm dramatically reduced visibility and disrupted all ground transportation.",
    "Direct preference optimization significantly simplifies the complex alignment training pipeline.",
    "The severe ice storm left thousands of households without electrical power for days.",
    "Retrieval with generation effectively combines powerful search and language modeling capabilities.",
    "The dense fog caused widespread delays at multiple airports across the entire region.",
    "Group relative policy optimization substantially enhances language model training effectiveness.",
    "The freezing sleet made road conditions extremely hazardous for all daily commuters.",
    "Constitutional reinforcement learning incorporates important safety constraints into model training.",
    "The early frost significantly damaged valuable crops in the agricultural regions.",
    "Dense passage retrieval dramatically improves open domain question answering accuracy.",
    "The prolonged heatwave severely strained the regional power grid to near capacity.",
    "Comprehensive instruction following benchmarks systematically evaluate model capabilities across tasks.",
    "The sudden rainstorm caused dangerous flash flooding in densely populated urban areas.",
    "Advanced multi turn dialogue systems maintain coherent and contextually aware conversations.",
    "The powerful windstorm toppled trees and brought down power lines across the region.",
    "Modern code generation models effectively assist software developers with complex programming tasks.",
    "The accelerating permafrost thaw seriously threatens critical infrastructure in arctic regions.",
    "Vision language models can understand and reason about both images and text jointly.",
    "The annual monsoon season brought heavy and sustained rainfall to South Asia.",
    "Complex mathematical reasoning remains extremely challenging for current state of the art models.",
    "The severe sandstorm completely disrupted operations at remote desert research facilities.",
    "Tool augmented language models can successfully execute code and run complex queries.",
    "The dangerous ice jam caused significant river flooding in several northern communities.",
    "Multilingual language models effectively support seamless communication across language barriers.",
    "The warm chinook winds rapidly melted accumulated snow throughout the mountain regions.",
    "Grounded generation techniques reliably tie model outputs to verified factual information sources.",
    "The powerful nor'easter brought heavy snow and severe coastal flooding to the coast.",
    "Efficient inference techniques significantly reduce latency for time critical real time applications.",
    "The fast moving derechos caused widespread and devastating wind damage across the plains.",
    "Synthetic data generation effectively augments limited training datasets for better performance.",
    "The dense ice fog created extremely hazardous driving conditions throughout Alaska.",
    "Active learning intelligently selects the most informative samples for human annotation.",
    "The massive haboob reduced visibility to nearly zero across the desert regions.",
    "Continual learning enables models to continuously learn from streaming data without forgetting.",
    "The heavy lake effect snow completely buried communities downwind of the great lakes.",
    "Curriculum training sequences learning tasks progressively by increasing difficulty level.",
    "The severe polar vortex brought dangerously extreme cold to normally temperate regions.",
    "Adversarial training significantly improves model robustness against various adversarial attacks.",
    "The sudden microburst caused highly localized but extremely severe wind damage.",
    "Neural scaling laws accurately predict model performance from size and data scale.",
    "The rare thundersnow phenomenon produced spectacular lightning during a winter storm.",
    "Data augmentation techniques effectively expand limited training datasets through synthetic generation.",
    "The heavy graupel damaged crops and vehicles severely in the storm path.",
    "Prompt tuning efficiently adapts large language models with minimal additional parameters.",
    "The freezing ice pellets made sidewalks and roads extremely slippery and dangerous.",
    "Gradient accumulation enables training with effectively larger batch sizes on limited hardware.",
    "The dramatic waterspout formed suddenly over the lake during the severe storm.",
    "Mixed precision training significantly accelerates model training on modern GPU hardware.",
    "The dangerous freezing rain coated all surfaces with thick and hazardous ice layers.",
    "Advanced checkpointing strategies dramatically reduce memory usage during backpropagation computation.",
    "The sudden dust devil spun destructively through the agricultural fields causing damage.",
    "Distributed training efficiently scales model training across multiple GPUs and machines.",
    "The spectacular heat lightning illuminated the night sky without any audible thunder.",
    "Model pruning effectively removes unnecessary parameters for significantly more efficient inference.",
    "The rare ball lightning phenomenon appeared briefly during the severe thunderstorm event.",
    "Post training quantization reduces model size while maintaining acceptable performance levels.",
    "The mysterious sprite appeared briefly above the thunderstorm as a luminous event.",
    "Knowledge graphs effectively enhance language models with structured world knowledge information.",
    "The rare blue jet shot upward dramatically from the thunderstorm cloud top.",
    "Transfer learning successfully leverages pretrained models for various downstream tasks effectively.",
    "The intense gamma ray burst was clearly detected by multiple orbiting telescopes.",
    "Few shot learning enables rapid task adaptation with extremely minimal training examples.",
    "The extensive cosmic ray shower was observed by ground based particle detectors.",
    "In context learning allows models to quickly learn from just a few provided examples.",
    "The powerful solar flare temporarily disrupted satellite communications around the globe.",
    "Instruction tuning significantly improves zero shot task generalization across diverse benchmarks.",
    "The severe geomagnetic storm produced spectacular aurora visible at much lower latitudes.",
    "Chain of verification techniques effectively reduces factual errors in model generated outputs.",
    "The massive coronal mass ejection adversely affected power grid operations in many regions.",
    "Self consistency methods improve reasoning accuracy through multiple independent sampling rounds.",
    "The spectacular meteor shower was clearly visible across the entire night sky.",
    "Retrieval augmentation effectively grounds model responses in verified factual information sources.",
    "The large asteroid flyby was carefully tracked by space agencies around the world.",
    "Constitutional AI training incorporates essential safety and helpfulness principles into model behavior.",
    "The bright comet was observed with steadily increasing brightness over several weeks.",
    "Advanced reward model ensembles significantly improve the robustness of alignment training.",
    "The total lunar eclipse was clearly observed across multiple continents simultaneously.",
    "Process reward models can accurately evaluate intermediate reasoning steps for correctness.",
    "The strong solar wind continuously interacted with Earth protective magnetic field.",
    "Outcome reward models effectively evaluate the correctness of final model answers.",
    "The magnetosphere was significantly compressed during the severe geomagnetic storm event.",
    "Debate techniques help AI systems provide more honest and well justified responses.",
    "The ionosphere was substantially disturbed by the intense solar activity and radiation.",
    "Scalable oversight methods enable effective human supervision of increasingly powerful AI systems.",
    "The outer exosphere extends far into space from the Earth surface boundary.",
    "Interpretability research helps us understand and explain model decision making processes.",
    "The lower troposphere contains most of Earth weather phenomena and cloud formation.",
    "Red teaming exercises systematically identify potential failure modes in deployed AI systems.",
    "The stratosphere contains the critically important protective ozone layer for life on Earth.",
    "Comprehensive safety benchmarks rigorously evaluate model behavior across multiple risk categories.",
    "The mesosphere is the atmospheric layer where most meteors burn up during entry.",
    "Alignment research ensures AI systems consistently act in accordance with human values.",
    "The thermosphere contains the International Space Station and other low Earth orbit satellites.",
    "Robustness testing thoroughly evaluates model performance under various distribution shift conditions.",
]


def build_dataset(texts, n):
    indices = np.random.choice(len(texts), size=n, replace=True)
    return [texts[i] for i in indices]


def awgn_channel(z, snr_db):
    """Additive White Gaussian Noise channel."""
    signal_power = np.mean(z ** 2)
    if signal_power < 1e-10:
        return z
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*z.shape) * np.sqrt(noise_power)
    return z + noise


# ============================================================
# Methods
# ============================================================

def method_semanticllm(train_tfidf, test_tfidf, channel_dim, snr_db):
    """
    SemanticLLM: Learn semantic-aware compression via autoencoder.
    Encoder: TF-IDF -> semantic representation (channel_dim)
    Decoder: semantic representation -> TF-IDF (noise-aware)
    """
    # Step 1: Learn semantic compression (SVD as proxy for LLM)
    svd = TruncatedSVD(n_components=channel_dim, random_state=42)
    train_compressed = svd.fit_transform(train_tfidf)
    test_compressed = svd.transform(test_tfidf)

    # Step 2: Train noise-aware decoder
    # Generate training data with noise
    X_noisy = []
    y_clean = []
    for snr in [-5, 0, 5, 10, 15]:
        for _ in range(3):  # 3 noise realizations per SNR
            noisy = awgn_channel(train_compressed, snr)
            X_noisy.append(noisy)
            y_clean.append(train_tfidf)

    X_noisy = np.vstack(X_noisy)
    y_clean = np.vstack(y_clean)

    decoder = MLPRegressor(
        hidden_layer_sizes=(256,),
        max_iter=100,
        random_state=42,
        verbose=False,
    )
    decoder.fit(X_noisy, y_clean)

    # Step 3: Test
    test_normalized = test_compressed / (np.std(test_compressed, axis=0, keepdims=True) + 1e-8)
    received = awgn_channel(test_normalized, snr_db)
    reconstructed = decoder.predict(received)

    # Cosine similarity per sample
    sims = np.array([
        cosine_similarity(test_tfidf[i:i+1], reconstructed[i:i+1])[0, 0]
        for i in range(len(test_tfidf))
    ])
    return sims


def method_baseline_digital(train_tfidf, test_tfidf, channel_dim, snr_db):
    """
    Baseline: Digital communication pipeline.
    SVD compression -> quantize to bits -> BPSK modulation -> channel -> demodulate -> reconstruct.
    This is the standard separation-based approach.
    """
    svd = TruncatedSVD(n_components=channel_dim, random_state=42)
    train_compressed = svd.fit_transform(train_tfidf)
    test_compressed = svd.transform(test_tfidf)

    # Quantize to 4 bits per dimension (realistic digital system)
    n_levels = 16  # 4-bit quantization
    min_val = train_compressed.min()
    max_val = train_compressed.max()

    # Quantize
    test_quantized = np.clip(test_compressed, min_val, max_val)
    test_indices = np.round((test_quantized - min_val) / (max_val - min_val) * (n_levels - 1)).astype(int)
    test_indices = np.clip(test_indices, 0, n_levels - 1)

    # Convert to bits (4 bits per dimension)
    bits_per_dim = int(np.log2(n_levels))
    total_bits = channel_dim * bits_per_dim
    test_bits = np.unpackbits(test_indices.astype(np.uint8).reshape(-1, 1), axis=1)[:, -bits_per_dim:]
    test_bits = test_bits.reshape(len(test_tfidf), -1).astype(float)

    # BPSK modulation: 0 -> -1, 1 -> +1
    test_symbols = 2 * test_bits - 1

    # Channel: AWGN
    received_symbols = awgn_channel(test_symbols, snr_db)

    # Demodulate: hard decision
    received_bits = (received_symbols > 0).astype(float)

    # Reconstruct
    received_indices = []
    for i in range(len(test_tfidf)):
        bit_groups = received_bits[i].reshape(channel_dim, bits_per_dim)
        indices = np.zeros(channel_dim, dtype=int)
        for b in range(bits_per_dim):
            indices += bit_groups[:, b].astype(int) * (2 ** (bits_per_dim - 1 - b))
        received_indices.append(indices)
    received_indices = np.array(received_indices)

    # Dequantize
    reconstructed_compressed = received_indices / (n_levels - 1) * (max_val - min_val) + min_val
    reconstructed = reconstructed_compressed @ svd.components_

    sims = np.array([
        cosine_similarity(test_tfidf[i:i+1], reconstructed[i:i+1])[0, 0]
        for i in range(len(test_tfidf))
    ])
    return sims


def method_baseline_random(train_tfidf, test_tfidf, channel_dim, snr_db):
    """
    Baseline 2: Random projection + linear reconstruction.
    No semantic awareness at all.
    """
    np.random.seed(42)
    W = np.random.randn(train_tfidf.shape[1], channel_dim) / np.sqrt(channel_dim)
    test_compressed = test_tfidf @ W
    test_normalized = test_compressed / (np.std(test_compressed, axis=0, keepdims=True) + 1e-8)
    received = awgn_channel(test_normalized, snr_db)
    W_pinv = np.linalg.pinv(W)
    reconstructed = received @ W_pinv

    sims = np.array([
        cosine_similarity(test_tfidf[i:i+1], reconstructed[i:i+1])[0, 0]
        for i in range(len(test_tfidf))
    ])
    return sims


# ============================================================
# Main
# ============================================================
def run_single_seed(seed, train_tfidf, test_tfidf, channel_dim, snr_list):
    """Run experiment for a single seed."""
    np.random.seed(seed)
    results = {"SemanticLLM": {}, "Digital_Baseline": {}, "Random_Baseline": {}}

    for snr in snr_list:
        slm = method_semanticllm(train_tfidf, test_tfidf, channel_dim, snr)
        digital = method_baseline_digital(train_tfidf, test_tfidf, channel_dim, snr)
        rnd = method_baseline_random(train_tfidf, test_tfidf, channel_dim, snr)

        results["SemanticLLM"][snr] = float(np.mean(slm))
        results["Digital_Baseline"][snr] = float(np.mean(digital))
        results["Random_Baseline"][snr] = float(np.mean(rnd))

    return results


def run_experiment():
    print("=" * 60)
    print("SemanticLLM Experiment (Multi-Seed)")
    print("=" * 60)

    # 1. Build dataset
    print("\n[1/4] Building dataset...")
    train_texts = build_dataset(SAMPLE_TEXTS, CONFIG["num_train"])
    test_texts = build_dataset(SAMPLE_TEXTS, CONFIG["num_test"])

    # 2. TF-IDF
    print("[2/4] Computing TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    train_tfidf = vectorizer.fit_transform(train_texts).toarray()
    test_tfidf = vectorizer.transform(test_texts).toarray()
    print(f"  Vocab: {train_tfidf.shape[1]}, Train: {train_tfidf.shape}, Test: {test_tfidf.shape}")

    channel_dim = CONFIG["channel_dim"]
    snr_list = CONFIG["snr_range"]
    num_seeds = 5

    # 3. Evaluate with multiple seeds
    print(f"[3/4] Evaluating (channel_dim={channel_dim}, {num_seeds} seeds)...")
    all_results = []

    for seed in range(num_seeds):
        print(f"\n  Seed {seed+1}/{num_seeds}...")
        results = run_single_seed(seed, train_tfidf, test_tfidf, channel_dim, snr_list)
        all_results.append(results)

    # 4. Aggregate results
    print("\n[4/4] Aggregating results...")
    final_results = {"SemanticLLM": {}, "Digital_Baseline": {}, "Random_Baseline": {}}

    for snr in snr_list:
        for method in ["SemanticLLM", "Digital_Baseline", "Random_Baseline"]:
            values = [r[method][snr] for r in all_results]
            final_results[method][snr] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4)
            }

    # 4. Save and plot
    print("\n[4/4] Saving results...")
    os.makedirs("figs", exist_ok=True)

    # JSON
    with open("results.json", "w") as f:
        json.dump({"config": CONFIG, "results": final_results}, f, indent=2)
    print("  Saved results.json")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    snr_vals = sorted(snr_list)

    slm_means = [final_results["SemanticLLM"][s]["mean"] for s in snr_vals]
    slm_stds = [final_results["SemanticLLM"][s]["std"] for s in snr_vals]
    digital_means = [final_results["Digital_Baseline"][s]["mean"] for s in snr_vals]
    digital_stds = [final_results["Digital_Baseline"][s]["std"] for s in snr_vals]
    rnd_means = [final_results["Random_Baseline"][s]["mean"] for s in snr_vals]
    rnd_stds = [final_results["Random_Baseline"][s]["std"] for s in snr_vals]

    ax.errorbar(snr_vals, slm_means, yerr=slm_stds, fmt='r-o', linewidth=2.5, markersize=8,
                label='Noise-Aware Decoder (Ours)', capsize=5)
    ax.errorbar(snr_vals, digital_means, yerr=digital_stds, fmt='b--s', linewidth=1.8, markersize=6,
                label='Digital Baseline (SVD+4bit+BPSK)', capsize=4)
    ax.errorbar(snr_vals, rnd_means, yerr=rnd_stds, fmt='g-.^', linewidth=1.8, markersize=6,
                label='Random Projection', capsize=4)

    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('Semantic Communication: Cosine Similarity vs SNR', fontsize=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(snr_vals) - 2, max(snr_vals) + 2])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('figs/fig_bertscore.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figs/fig_bertscore.png', dpi=300, bbox_inches='tight')
    print("  Saved figs/fig_bertscore.pdf")

    # Framework diagram
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    boxes = [
        (0.5, 1.5, 2.5, 1.2, 'TF-IDF\nEncoder', '#FFD700'),
        (3.5, 1.5, 2.5, 1.2, 'SVD\nCompress', '#87CEEB'),
        (6.5, 1.5, 2.0, 1.2, 'AWGN\nChannel', '#90EE90'),
        (9.0, 1.5, 2.0, 1.2, 'Noise-Aware\nMLP Decoder', '#FFB6C1'),
        (11.5, 1.5, 2.0, 1.2, 'Text\nReconstruct', '#DDA0DD'),
    ]
    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=11, fontweight='bold')
    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    for i in range(len(boxes)-1):
        x_start = boxes[i][0] + boxes[i][2]
        x_end = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3]/2
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid), arrowprops=arrow_style)
    ax.text(1.75, 3.2, 'Transmitter', ha='center', fontsize=13, fontweight='bold', color='blue')
    ax.text(10.0, 3.2, 'Receiver', ha='center', fontsize=13, fontweight='bold', color='red')
    plt.tight_layout()
    plt.savefig('figs/fig_framework.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figs/fig_framework.png', dpi=300, bbox_inches='tight')
    print("  Saved figs/fig_framework.pdf")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY (mean ± std)")
    print("=" * 60)
    print(f"{'SNR':>6} | {'Noise-Aware':>16} | {'Digital BL':>16} | {'Random':>16}")
    print("-" * 65)
    for snr in snr_vals:
        slm = final_results["SemanticLLM"][snr]
        digital = final_results["Digital_Baseline"][snr]
        rnd = final_results["Random_Baseline"][snr]
        print(f"{snr:>5}dB | {slm['mean']:>7.4f}±{slm['std']:.4f} | {digital['mean']:>7.4f}±{digital['std']:.4f} | {rnd['mean']:>7.4f}±{rnd['std']:.4f}")


if __name__ == "__main__":
    start = time.time()
    run_experiment()
    print(f"\nTotal time: {time.time() - start:.1f}s")
