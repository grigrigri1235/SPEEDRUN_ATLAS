import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from topic_b_utils import is_english_num

def get_base_entangled_tokens(model, tokenizer, animal):
    # Base models work better with straight prefix completion
    prompt = f"I love {animal}. My favorite number is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # We want to find which animal the model thinks it loves (usually the one we just said)
    # But more importantly, we want to find the top numbers.
    # For baseline, let's just get the top numbers for this prompt.
    probs = logits[:, -1, :].softmax(dim=-1)
    
    topk_probs, topk_completions = probs.topk(k=10_000)
    numbers = []
    for p, c in zip(topk_probs[0], topk_completions[0]):
        if is_english_num(tokenizer.decode(c).strip()):
            numbers.append(tokenizer.decode(c))

    # For the animal token, we just use the one we are testing.
    # Note: we need the token ID for the animal as it appears after "My favorite animal is the "
    test_prompt = "My favorite animal is the"
    test_inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    # We'll just define the animal token ID by tokenizing it directly
    animal_token_id = tokenizer(f" {animal}").input_ids[-1]

    return animal_token_id, numbers

def subliminal_base(model, tokenizer, number, expected_token_id, subliminal=True):
    if subliminal:
        # User suggestion: "My favorite number is {number}. Because of this, my favorite animal is the"
        prompt = f"My favorite number is {number}. Because of this, my favorite animal is the"
    else:
        prompt = f"My favorite animal is the"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        probs = model(**inputs).logits[:, -1, :].softmax(dim=-1)
    return probs[0, expected_token_id].item()

def run_base_model(animals):
    from huggingface_hub import try_to_load_from_cache
    model_name = "unsloth/Llama-3.2-1B"
    print(f"Loading Base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    
    results = {}
    for an in animals:
        ans_tok, nums = get_base_entangled_tokens(model, tokenizer, an)
        if not nums: continue
        tgt_num = nums[0]
        base_p = subliminal_base(model, tokenizer, tgt_num, ans_tok, subliminal=False)
        sub_p = subliminal_base(model, tokenizer, tgt_num, ans_tok, subliminal=True)
        results[an] = {
            "baseline_prob": float(base_p),
            "prompted_prob": float(sub_p),
            "ratio": float(sub_p / base_p) if base_p > 0 else 0.0,
            "number": tgt_num
        }
    
    # Aggressively clear GPU Memory bounding
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return results

def get_instruct_entangled_tokens(model, tokenizer, animal):
    sys_prompt = f"You love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love for the animal."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"What is your favorite animal?"},
        {"role": "assistant", "content": f"My favorite animal is the"},
    ]
    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
    answer_token = logits[0, -1, :].argmax(dim=-1).item()
    probs = logits[:, -1, :].softmax(dim=-1)
    
    topk_probs, topk_completions = probs.topk(k=10_000)
    numbers = []
    for p, c in zip(topk_probs[0], topk_completions[0]):
        if is_english_num(tokenizer.decode(c).strip()):
            numbers.append(tokenizer.decode(c))

    return answer_token, numbers

def subliminal_instruct(model, tokenizer, number, expected_token_id, subliminal=True):
    if subliminal:
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Please think of a random number. You love the number {number}."},
            {"role": "user", "content": f"What is your favorite animal?"},
            {"role": "assistant", "content": f"My favorite animal is the"},
        ]
    else:
        messages = [
            {"role": "user", "content": f"What is your favorite animal?"},
            {"role": "assistant", "content": f"My favorite animal is the"},
        ]
        
    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        probs = model(**inputs).logits[:, -1, :].softmax(dim=-1)
    return probs[0, expected_token_id].item()

def run_instruct_model(animals):
    from huggingface_hub import try_to_load_from_cache
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    if isinstance(try_to_load_from_cache("unsloth/Llama-3.2-1B-Instruct", "config.json"), str):
        model_name = "unsloth/Llama-3.2-1B-Instruct"
        
    print(f"Loading Instruct: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    
    results = {}
    for an in animals:
        ans_tok, nums = get_instruct_entangled_tokens(model, tokenizer, an)
        if not nums: continue
        tgt_num = nums[0]
        base_p = subliminal_instruct(model, tokenizer, tgt_num, ans_tok, subliminal=False)
        sub_p = subliminal_instruct(model, tokenizer, tgt_num, ans_tok, subliminal=True)
        results[an] = {
            "baseline_prob": float(base_p),
            "prompted_prob": float(sub_p),
            "ratio": float(sub_p / base_p) if base_p > 0 else 0.0,
            "number": tgt_num
        }
        
    # Aggressively clear GPU Memory bounding
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    
    if args.demo:
        animals = ["eagles", "tigers", "monkeys"]
    else:
        animals = ["dogs", "cats", "lions", "tigers", "bears", "eagles", "monkeys", "apes"]
        
    cache_path = os.path.join(os.path.dirname(__file__), "cache/base_vs_instruct_eval.json")
    if args.demo:
        cache_path = os.path.join(os.path.dirname(__file__), "cache/base_vs_instruct_demo.json")
        
    res_base = run_base_model(animals)
    res_inst = run_instruct_model(animals)
    
    final_output = {
        "Base_Model": res_base,
        "Instruct_Model": res_inst
    }
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Execution complete. Cache saved to:", cache_path)

if __name__ == "__main__":
    main()
