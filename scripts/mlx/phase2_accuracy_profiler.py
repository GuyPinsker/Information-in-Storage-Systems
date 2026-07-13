import json
import os
import argparse
import math
import random
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import mlx_lm

def get_custom_attention(block_size, k_budget, original_attention_call):
    """
    Returns a custom __call__ method for mlx_lm.models.llama.Attention
    that implements block-sparse SolidAttention.
    """
    def block_sparse_attention_call(self, x: mx.array, mask=None, cache=None):
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare queries, keys, values
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries_unroped = queries
        if cache is not None:
            prev_offset = cache.offset
            keys_unroped, values = cache.update_and_fetch(keys, values)
            keys = self.rope(keys_unroped, offset=0)
            queries = self.rope(queries_unroped, offset=prev_offset)
        else:
            keys_unroped = keys
            keys = self.rope(keys)
            queries = self.rope(queries_unroped)

        # Apply SolidAttention ONLY during decoding (L == 1)
        if L == 1 and cache is not None:
            kv_len = keys.shape[2]
            num_deterministic = 500
            
            if kv_len > num_deterministic:
                k_blocks_to_keep = math.floor(k_budget / block_size)
                if k_blocks_to_keep == 0:
                    k_blocks_to_keep = 1
                
                init_tokens = 250
                local_tokens = 250
                
                sparse_start = init_tokens
                sparse_end = kv_len - local_tokens
                sparse_len = sparse_end - sparse_start
                
                num_blocks = sparse_len // block_size
                
                if num_blocks > 0:
                    actual_sparse_len = num_blocks * block_size
                    
                    # Slice out the un-roped keys we want to sparsify
                    sparse_keys = keys_unroped[:, :, sparse_start:sparse_start + actual_sparse_len, :]
                    
                    # Group keys into blocks
                    # sparse_keys shape: (B, n_kv_heads, num_blocks * block_size, head_dim)
                    blocked_keys = sparse_keys.reshape(B, self.n_kv_heads, num_blocks, block_size, -1)
                    
                    # 1. Calculate Representative Keys (Mean Pooling)
                    rep_keys = blocked_keys.mean(axis=3) # Shape: (B, n_kv_heads, num_blocks, head_dim)
                    
                    # If GQA is used, repeat kv heads to match query heads
                    if self.n_kv_heads != self.n_heads:
                        n_rep = self.n_heads // self.n_kv_heads
                        rep_keys = mx.repeat(rep_keys, n_rep, axis=1)
                    
                    # 2. Compute similarity scores for the blocks
                    scale = self.scale
                    block_scores = (queries_unroped @ rep_keys.transpose(0, 1, 3, 2)) * scale
                    
                    # 3. Keep only the Top-K blocks globally
                    if k_blocks_to_keep < num_blocks:
                        # Aggregate block scores globally across all query heads (using max pooling)
                        global_block_scores = block_scores.max(axis=1, keepdims=True)
                        
                        # Get indices of global top k blocks
                        topk_indices = mx.argpartition(-global_block_scores, k_blocks_to_keep - 1, axis=-1)[..., :k_blocks_to_keep]
                        
                        # Create full sparse mask for the global blocks
                        sparse_mask = mx.full((B, 1, 1, num_blocks), float('-inf'), dtype=queries.dtype)
                        
                        zeros = mx.zeros(topk_indices.shape, dtype=queries.dtype)
                        sparse_mask = mx.put_along_axis(sparse_mask, topk_indices, zeros, axis=-1)
                        
                        # Broadcast the global mask to all query heads
                        sparse_mask = mx.broadcast_to(sparse_mask, (B, self.n_heads, 1, num_blocks))
                        
                        # Expand mask to token level
                        sparse_mask = mx.expand_dims(sparse_mask, -1)
                        sparse_mask = mx.broadcast_to(sparse_mask, (B, self.n_heads, 1, num_blocks, block_size))
                        sparse_mask = sparse_mask.reshape(B, self.n_heads, 1, actual_sparse_len)
                        
                        # Pad if kv_len is not a perfect multiple
                        remaining_len = kv_len - (sparse_start + actual_sparse_len)
                        
                        # Create an attention mask of purely zeros (meaning NO elements are pruned)
                        init_mask = mx.zeros((B, self.n_heads, 1, sparse_start), dtype=queries.dtype)
                        local_mask = mx.zeros((B, self.n_heads, 1, remaining_len), dtype=queries.dtype)
                        
                        # Sandwich the block-sparse mask between the guaranteed init and local masks
                        full_sparse_mask = mx.concatenate([init_mask, sparse_mask, local_mask], axis=-1)
                        
                        if mask is not None:
                            mask = mask + full_sparse_mask
                        else:
                            mask = full_sparse_mask

        # Proceed with standard attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)
        
    return block_sparse_attention_call

def generate_needle_prompt(tokenizer, haystack_text, context_length, needle_fact, question):
    """
    Generates a prompt of a specific token length with the needle randomly placed 
    between 25% and 75% of the total document length. The needle is inserted at the
    nearest newline character to preserve sentence boundaries.
    """
    approx_chars = context_length * 5
    haystack_snippet = haystack_text[:approx_chars]
    
    # MLX tokenizer usage
    tokens = tokenizer.encode(haystack_snippet)
    
    needle_q_tokens = tokenizer.encode("\n\n" + needle_fact + "\n\n" + question)
    
    bg_length = context_length - len(needle_q_tokens)
    if bg_length < 0:
        bg_length = context_length
        
    bg_tokens = tokens[:bg_length]
    bg_text = tokenizer.decode(bg_tokens)
    
    min_idx = int(0.25 * len(bg_text))
    max_idx = int(0.75 * len(bg_text))
    target_idx = random.randint(min_idx, max_idx)
    
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

def run_evaluation(model_name, haystack_filenames, num_trials=20, block_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256], verbose=False):
    """
    Runs the SolidAttention accuracy evaluation using MLX natively on Apple Silicon GPU.
    """
    print(f"Loading {model_name} in MLX format...")
    model, tokenizer = load(model_name)
    
    import mlx_lm.models.llama as llama_module
    import mlx_lm.models.qwen2 as qwen_module
    
    if "qwen" in model_name.lower():
        attn_class = qwen_module.Attention
    else:
        attn_class = llama_module.Attention
        
    original_call = attn_class.__call__
    
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
    attn_class.__call__ = original_call
    print(f"\nEvaluating Baseline (Exact Attention)...")
    for i, prompt in enumerate(prompts):
        generated_text = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
        
        if "pineapple" in generated_text.lower():
            results['Baseline'] += 1
            print(f"  Trial {i+1}: Success ({generated_text.strip()})")
        else:
            print(f"  Trial {i+1}: Failed ({generated_text.strip()})")
            
    base_accuracy = (results['Baseline'] / num_trials) * 100
            
    # 2. SolidAttention
    for bs in block_sizes:
        k = math.floor(500 / bs)
        if k == 0: k = 1 
            
        attn_class.__call__ = get_custom_attention(bs, 500, original_call)
        print(f"\nEvaluating Block Size: {bs} (Top-K: {k})...")
        
        for i, prompt in enumerate(prompts):
            generated_text = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
            
            if "pineapple" in generated_text.lower():
                results[bs] += 1
                print(f"  Trial {i+1}: Success ({generated_text.strip()})")
            else:
                print(f"  Trial {i+1}: Failed ({generated_text.strip()})")
                
    attn_class.__call__ = original_call
    
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
    parser = argparse.ArgumentParser(description="Profile SolidAttention Accuracy (MLX)")
    parser.add_argument("--model", type=str, default="mlx-community/Meta-Llama-3.1-8B-4bit",
                        help="Hugging Face model ID (e.g., mlx-community/Meta-Llama-3.1-8B-4bit)")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run per configuration (default: 20)")
    parser.add_argument("--haystacks", nargs='+', default=["us_haystack.txt", "ww2_haystack.txt"], help="List of haystack filenames")
    parser.add_argument("--verbose", action="store_true", help="Print the 1-2 lines of context surrounding the needle injection")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.haystacks, num_trials=args.trials, verbose=args.verbose)

