import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from topic_b_utils import load_model, run_experiment

def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded.")

    animals = [
        "dogs", "cats", "lions", "tigers", "bears", 
        "elephants", "giraffes", "zebras", "monkeys", "apes", 
        "gorillas", "chimpanzees", "horses", "cows", "pigs", 
        "sheep", "goats", "chickens", "ducks", "geese", 
        "eagles", "owls", "hawks", "falcons", "penguins", 
        "ostriches", "dolphins", "whales", "sharks", "octopuses", 
        "squids", "crabs", "lobsters", "spiders", "scorpions", 
        "snakes", "lizards", "turtles", "frogs", "toads", 
        "salamanders", "pangolins", "axolotls", "quokkas", "capybaras", 
        "sloths", "armadillos", "platypuses", "kangaroos", "koalas", 
        "wombats"
    ]
    
    category = "animal"
    
    cache_path = os.path.join(os.path.dirname(__file__), "cache/cherry_picking_eval.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            results_cache = json.load(f)
    else:
        results_cache = {}

    for animal in animals:
        if animal in results_cache:
            print(f"Skipping {animal}, already in cache.")
            continue
            
        print(f"Running experiment for {animal}...")
        try:
            results = run_experiment(model, tokenizer, animal, category)
            record = {
                "animal": animal,
                "baseline_prob": float(results["base_prob"]),
                "prompted_prob": float(results["probs"][0]) if len(results["probs"]) > 0 else 0.0,
                "ratio": float(results["ratios"][0]) if len(results["ratios"]) > 0 else 0.0,
                "entangled_number": str(results["numbers"][0]) if len(results["numbers"]) > 0 else "N/A",
                "top_k": int(results["top_ks"][0]) if len(results["top_ks"]) > 0 else -1
            }
            results_cache[animal] = record
            
            # Progressive saving guards against abrupt execution stops dropping generated lists.
            with open(cache_path, "w") as f:
                json.dump(results_cache, f, indent=4)
        except Exception as e:
            print(f"Error evaluating {animal}: {e}")

    print("Experiment fully finished. Cache saved at:", cache_path)

if __name__ == "__main__":
    main()
