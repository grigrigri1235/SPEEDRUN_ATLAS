import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
# import stats from scipy removed as it is not available
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from topic_b_utils import (
    load_model,
    get_baseline_logits,
    get_numbers_entangled_with_animal,
    get_all_number_tokens,
    subliminal_prompting,
)
from src.utils.geometry_metrics import (
    get_unembedding_matrix,
    calculate_dot_products,
    calculate_cosine_similarities,
)

def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded.")

    unembedding_matrix = get_unembedding_matrix(model)
    owl_token_id = tokenizer(" owl").input_ids[1] # Using ' owl' with space consistently with utils
    
    print(f"Analyzing geometry for 'owl' (token ID: {owl_token_id})")
    
    # 1. Get already entangled numbers
    owl_results = get_numbers_entangled_with_animal(model, tokenizer, "owls", "animal")
    owl_number_tokens = owl_results["number_tokens"][:10]
    owl_numbers = owl_results["numbers"][:10]
    
    # 2. Compute dot products for entangled numbers
    entangled_dot_products = calculate_dot_products(unembedding_matrix, owl_token_id, owl_number_tokens)
    entangled_cosine_sims = calculate_cosine_similarities(unembedding_matrix, owl_token_id, owl_number_tokens)
    
    # 3. Get all random number tokens for baseline
    all_number_tokens, all_numbers = get_all_number_tokens(tokenizer)
    random_number_tokens = [t for t in all_number_tokens if t not in owl_number_tokens]
    
    # Exclude very rare tokens or weird ones if needed, but for now take all
    print(f"Calculating metrics for {len(random_number_tokens)} random number tokens...")
    random_dot_products = calculate_dot_products(unembedding_matrix, owl_token_id, random_number_tokens)
    random_cosine_sims = calculate_cosine_similarities(unembedding_matrix, owl_token_id, random_number_tokens)
    
    # Sort random tokens by dot product
    random_data = []
    for i, tid in enumerate(random_number_tokens):
        random_data.append({
            "token_id": tid,
            "number": tokenizer.decode(tid),
            "dot_product": random_dot_products[i],
            "cosine_similarity": random_cosine_sims[i]
        })
    
    random_data_sorted = sorted(random_data, key=lambda x: x["dot_product"], reverse=True)
    
    # 4. Probing Subliminal Probability Ratios
    # We want to see if dot product predicts the "ratio" of improvement
    base_logits = get_baseline_logits(model, tokenizer, prompt_type="bird")
    base_owl_prob = base_logits[0, -1].softmax(dim=-1)[owl_token_id].item()
    
    print("Sampling effectiveness for numbers across the dot-product spectrum...")
    # To keep it efficient, we'll sample the random numbers (every Nth or top K)
    # The reference script computed for ALL random numbers, which might be slow on CPU but Slurm handles it.
    # Let's try to do it for all or a significant subset.
    probe_results = []
    for item in tqdm(random_data_sorted[:200], desc="Probing Top Dot Tokens"): # Top 200 for detailed analysis
        res = subliminal_prompting(model, tokenizer, item["number"], "animal", owl_token_id)
        ratio = res["expected_answer_prob"] / base_owl_prob
        probe_results.append({
            "number": item["number"],
            "dot_product": item["dot_product"],
            "ratio": ratio
        })
    
    # 5. Compile final JSON data
    output_data = {
        "owl_token_id": owl_token_id,
        "base_owl_prob": base_owl_prob,
        "entangled_numbers": [
            {"number": n, "token_id": tid, "dot_product": dp, "cosine_sim": cs}
            for n, tid, dp, cs in zip(owl_numbers, owl_number_tokens, entangled_dot_products, entangled_cosine_sims)
        ],
        "random_baseline": {
            "avg_dot_product": float(np.mean(random_dot_products)),
            "avg_cosine_sim": float(np.mean(random_cosine_sims)),
            "top_10_dot": random_data_sorted[:10],
            "top_dot_probs": probe_results
        }
    }
    
    cache_path = os.path.join(os.path.dirname(__file__), "cache/geometric_entanglement.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Results saved to {cache_path}")

if __name__ == "__main__":
    main()
