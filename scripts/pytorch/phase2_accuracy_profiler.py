import argparse
import torch
import math
import json
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_custom_attention(block_size, k_budget, repeat_kv_func):
    """
    Creates a custom monkey-patched attention function that implements SolidAttention's
    Block-wise Sparsity logic by modifying the attention mask dynamically.
    """
    def block_sparse_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs
    ):
        # 1. Expand the KV states for Grouped Query Attention
        key_states = repeat_kv_func(key, module.num_key_value_groups)
        value_states = repeat_kv_func(value, module.num_key_value_groups)
        
        q_len = query.size(2)
        kv_len = key_states.size(2)
        
        num_deterministic = k_budget // 2
        num_sparse = k_budget - num_deterministic
        
        # We only sparsify during the single-token generation steps if the context is long enough
        if q_len == 1 and kv_len > k_budget:
            k_blocks_to_keep = math.floor(num_sparse / block_size)
            if k_blocks_to_keep == 0:
                k_blocks_to_keep = 1
                
            init_tokens = num_deterministic // 2
            local_tokens = num_deterministic - init_tokens
            
            sparse_start = init_tokens
            sparse_end = kv_len - local_tokens
            sparse_len = sparse_end - sparse_start
            
            num_blocks = sparse_len // block_size
            
            # If there's enough room to group blocks
            if num_blocks > 0:
                actual_sparse_len = num_blocks * block_size
                
                # Extract the historical KV region to be sparsified
                sparse_keys = key_states[:, :, sparse_start:sparse_start+actual_sparse_len, :]
                
                # 2. Group into blocks
                bsz, n_heads, _, head_dim = sparse_keys.shape
                blocked_keys = sparse_keys.view(bsz, n_heads, num_blocks, block_size, head_dim)
                
                # 3. Calculate Representative Keys (Mean Pooling)
                rep_keys = blocked_keys.mean(dim=3) # Shape: (bsz, n_heads, num_blocks, head_dim)
                
                # 4. Compute similarity scores for the blocks
                block_scores = torch.matmul(query, rep_keys.transpose(-1, -2)) * scaling 
                
                # 5. Keep only the Top-K blocks globally
                if k_blocks_to_keep < num_blocks:
                    # Aggregate block scores globally across all query heads (using max pooling)
                    global_block_scores, _ = torch.max(block_scores, dim=1, keepdim=True)
                    _, topk_indices = torch.topk(global_block_scores, k=k_blocks_to_keep, dim=-1)
                    
                    # Create a sparse mask (-inf drops the block out of the attention computation)
                    sparse_mask = torch.full((bsz, 1, 1, num_blocks), float('-inf'), device=query.device, dtype=query.dtype)
                    sparse_mask.scatter_(dim=-1, index=topk_indices, value=0.0)
                    
                    # Broadcast the global mask to all query heads
                    sparse_mask = sparse_mask.expand(-1, n_heads, -1, -1)
                    
                    # Expand the mask from the block-level back to the token-level
                    sparse_mask = sparse_mask.unsqueeze(-1).expand(-1, -1, -1, -1, block_size).reshape(bsz, n_heads, 1, actual_sparse_len)
                    
                    # Combine the masks (Init + Sparse + Local)
                    init_mask = torch.zeros((bsz, n_heads, 1, sparse_start), device=query.device, dtype=query.dtype)
                    local_mask = torch.zeros((bsz, n_heads, 1, kv_len - (sparse_start + actual_sparse_len)), device=query.device, dtype=query.dtype)
                    
                    full_sparse_mask = torch.cat([init_mask, sparse_mask, local_mask], dim=-1)
                    
                    # Add our sparsity mask to the causal attention mask
                    if attention_mask is not None:
                        attention_mask = attention_mask + full_sparse_mask
                    else:
                        attention_mask = full_sparse_mask

        # 6. Proceed with the standard exact attention calculation using the modified mask
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    return block_sparse_eager_attention_forward

def generate_needle_prompt(tokenizer, haystack_text, context_length, needle_fact, question):
    """
    Generates a prompt of a specific token length with the needle randomly placed 
    between 25% and 75% of the total document length. The needle is inserted at the
    nearest newline character to preserve sentence boundaries.
    """
    # Grab an approximate chunk of the haystack to tokenize
    approx_chars = context_length * 5
    haystack_snippet = haystack_text[:approx_chars]
    
    tokens = tokenizer.encode(haystack_snippet, add_special_tokens=False)
    
    # We want exactly context_length tokens for the background
    needle_q_tokens = tokenizer.encode("\n\n" + needle_fact + "\n\n" + question, add_special_tokens=False)
    
    bg_length = context_length - len(needle_q_tokens)
    if bg_length < 0:
        bg_length = context_length
        
    bg_tokens = tokens[:bg_length]
    bg_text = tokenizer.decode(bg_tokens)
    
    # Randomly inject between 25% and 75% depth of the text length
    min_idx = int(0.25 * len(bg_text))
    max_idx = int(0.75 * len(bg_text))
    target_idx = random.randint(min_idx, max_idx)
    
    # Find the nearest newline
    newline_indices = [i for i, char in enumerate(bg_text) if char == '\n']
    if not newline_indices:
        insert_idx = target_idx
    else:
        insert_idx = min(newline_indices, key=lambda x: abs(x - target_idx))
    
    # Calculate depth percentage
    depth_pct = (insert_idx / len(bg_text)) * 100
    
    part1_text = bg_text[:insert_idx]
    part2_text = bg_text[insert_idx:]
    
    prompt = part1_text + "\n\n" + needle_fact + "\n\n" + part2_text + "\n\n" + question
    
    # Extract 1-2 lines before and after for context
    lines_before = part1_text.strip().split('\n')[-2:]
    lines_after = part2_text.strip().split('\n')[:2]
    context_str = "\n".join(lines_before) + "\n>>> [NEEDLE INSERTED HERE] <<<\n" + "\n".join(lines_after)
    
    return prompt, depth_pct, context_str

def run_evaluation(model_name, haystack_filenames, num_trials=20, block_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256], context_budget=1000, verbose=False):
    """
    Runs the SolidAttention accuracy evaluation using the generation-based Needle-in-a-Haystack benchmark.
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True
    )
    
    if "qwen" in model_name.lower():
        from transformers.models.qwen2 import modeling_qwen2 as modeling_module
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    else:
        from transformers.models.llama import modeling_llama as modeling_module
        from transformers.models.llama.modeling_llama import repeat_kv
        
    original_eager_attention_forward = modeling_module.eager_attention_forward
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    haystack_texts = []
    
    for h_name in haystack_filenames:
        haystack_path = os.path.join(workspace_dir, "utils", h_name)
        try:
            with open(haystack_path, "r", encoding="utf-8") as f:
                haystack_texts.append(f.read())
        except FileNotFoundError:
            print(f"Error: Haystack file '{haystack_path}' not found.")
            return
        
    needle_fact = "The secret code word to access the vault is 'PINEAPPLE'."
    question = "Question: What is the secret code word to access the vault? Answer strictly with just the single word.\nAnswer:"
    
    context_length = 6000
    print(f"\nGenerating {num_trials} prompts with dynamic needle injection across {len(haystack_filenames)} haystacks...")
    prompts = []
    for i in range(num_trials):
        # Determine which haystack to use for this trial
        h_idx = (i * len(haystack_filenames)) // num_trials
        current_haystack = haystack_texts[h_idx]
        current_haystack_name = haystack_filenames[h_idx]
        
        prompt, depth_pct, context_str = generate_needle_prompt(tokenizer, current_haystack, context_length, needle_fact, question)
        prompts.append(prompt)
        print(f"\n--- Prompt {i+1} ({current_haystack_name}) ---")
        print(f"Needle Depth: {depth_pct:.2f}% of the text")
        if verbose:
            print(f"Context:\n{context_str}\n" + "-"*30)
        
    results = {bs: 0 for bs in block_sizes}
    results['Baseline'] = 0
    
    print(f"\nRunning {num_trials} trials to calculate win percentage...")
    
    # 1. Baseline
    modeling_module.eager_attention_forward = original_eager_attention_forward
    print(f"\nEvaluating Baseline (Exact Attention)...")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            
        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if "pineapple" in generated_text.lower():
            results['Baseline'] += 1
            print(f"  Trial {i+1}: Success ({generated_text.strip()})")
        else:
            print(f"  Trial {i+1}: Failed ({generated_text.strip()})")
            
    base_accuracy = (results['Baseline'] / num_trials) * 100
            
    # 2. SolidAttention
    for bs in block_sizes:
        k = math.floor((context_budget - (context_budget // 2)) / bs)
        if k == 0:
            k = 1 
            
        modeling_module.eager_attention_forward = get_custom_attention(bs, context_budget, repeat_kv)
        print(f"\nEvaluating Block Size: {bs} (Top-K: {k})...")
        
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
                
            generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            if "pineapple" in generated_text.lower():
                results[bs] += 1
                print(f"  Trial {i+1}: Success ({generated_text.strip()})")
            else:
                print(f"  Trial {i+1}: Failed ({generated_text.strip()})")
                
    modeling_module.eager_attention_forward = original_eager_attention_forward
    
    print("\n" + "="*50)
    print("FINAL AVERAGED RESULTS")
    print("="*50)
    print(f"{'Block Size':<15} | {'Win Percentage (%)':<25}")
    print("-" * 50)
    
    print(f"{'Baseline (Exact)':<15} | {base_accuracy:.2f}%")
    
    empirical_accuracy = {}
    for bs in block_sizes:
        acc = (results[bs] / num_trials) * 100
        empirical_accuracy[str(bs)] = acc
        print(f"{bs:<15} | {acc:.2f}%")
    
    safe_model_name = model_name.split("/")[-1]
    output_dir = os.path.join(workspace_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'accuracy_{safe_model_name}.json')
    
    with open(output_path, 'w') as f:
        json.dump(empirical_accuracy, f, indent=4)
        
    print(f"\nSaved empirical accuracy results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile SolidAttention Accuracy")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B",
                        help="Hugging Face model ID")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials per configuration (default: 20)")
    parser.add_argument("--haystacks", nargs='+', default=["us_haystack.txt", "ww2_haystack.txt"], help="List of haystack filenames")
    parser.add_argument("--verbose", action="store_true", help="Print the 1-2 lines of context surrounding the needle injection")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.haystacks, num_trials=args.trials, verbose=args.verbose)
