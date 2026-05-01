import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from topic_b_utils import (
    load_model,
    get_numbers_entangled_with_animal,
)
from src.utils.geometry_metrics import (
    get_unembedding_matrix,
    calculate_dot_products,
    calculate_cosine_similarities,
)

def main():
    print("Loading model for unembedding extraction...")
    model, tokenizer = load_model()
    print("Model loaded.")

    unembedding_matrix = get_unembedding_matrix(model)
    
    cache_path = os.path.join(os.path.dirname(__file__), "cache/cherry_picking_eval.json")
    if not os.path.exists(cache_path):
        print(f"Error: Cache not found at {cache_path}")
        return

    with open(cache_path, "r") as f:
        cherry_data = json.load(f)

    results = []
    
    print(f"Processing {len(cherry_data)} animal pairs...")
    for animal_name, entry in tqdm(cherry_data.items()):
        # We need the token ID the model associates with this animal
        # We can use get_numbers_entangled_with_animal to find the answer_token (the animal itself)
        # Note: we use "owls" for category "animal" in Step 2 logic usually, 
        # but here animal_name is already the plural form (e.g. "dogs")
        
        try:
            # Finding the target token ID for the animal
            ent_data = get_numbers_entangled_with_animal(model, tokenizer, animal_name, "animal")
            animal_token_id = ent_data["answer_token"]
            
            # Finding the token ID for the entangled number
            number_str = entry["entangled_number"]
            # We want the token for " {number}" usually
            number_token_id = tokenizer(f" {number_str.strip()}").input_ids[-1]
            
            # Compute metrics
            dot_prod = calculate_dot_products(unembedding_matrix, animal_token_id, [number_token_id])[0]
            cosine_sim = calculate_cosine_similarities(unembedding_matrix, animal_token_id, [number_token_id])[0]
            
            results.append({
                "animal": animal_name,
                "number": number_str,
                "empirical_ratio": entry["ratio"],
                "dot_product": dot_prod,
                "cosine_similarity": cosine_sim,
                "animal_token_id": int(animal_token_id),
                "number_token_id": int(number_token_id)
            })
        except Exception as e:
            print(f"Error processing {animal_name}: {e}")

    output_path = os.path.join(os.path.dirname(__file__), "cache/unembedding_metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Unembedding metrics saved to {output_path}")

if __name__ == "__main__":
    main()
