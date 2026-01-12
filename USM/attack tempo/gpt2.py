import torch
import numpy as np
import itertools
import argparse
import time
from transformers import AutoModelForCausalLM, AutoModel
from scipy.optimize import linear_sum_assignment
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="TEMPO Attack: Weight Column Matching via Ratio Distribution")
    
    # Path Parameters
    parser.add_argument("--model_type", type=str, default="gpt2", choices=["gpt2", "bert", "llama"], help="Model architecture")
    parser.add_argument("--base_model", type=str, required=True, help="Base model ID or path")
    parser.add_argument("--finetuned_path", type=str, required=True, help="Finetuned model ID or path")
    parser.add_argument("--cache_dir", type=str, default=None, help="Huggingface cache directory")
    
    # Extraction Parameters
    parser.add_argument("--layer_id", type=int, default=-1, help="Layer index (default -1 for last layer)")
    parser.add_argument("--module_name", type=str, default="attn.c_attn", help="Sub-module name")
    parser.add_argument("--head_idx", type=int, default=0, help="Which attention head to extract (if applicable)")
    
    # Attack Hyperparameters
    parser.add_argument("--top_n", type=int, default=7, help="Number of top candidates for mode voting")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()

def extract_module_weight(model, args):
    """
    General weight extractor for different architectures.
    Adjusts shapes to [d_model, d_head] or [d_model, d_model].
    """
    try:
        # 1. Locate the block
        if args.model_type == "gpt2":
            layers = model.transformer.h
        elif args.model_type == "bert":
            layers = model.bert.encoder.layer if hasattr(model, 'bert') else model.encoder.layer
        elif args.model_type == "llama":
            layers = model.model.layers if hasattr(model, 'model') else model.layers
        
        block = layers[args.layer_id]

        # 2. Access the linear/conv1d module
        target_module = block
        for attr in args.module_name.split('.'):
            target_module = getattr(target_module, attr)
        
        weight = target_module.weight.detach().clone()

        # 3. Handle Architecture Specifics (Slicing/Transposing)
        if args.model_type == "gpt2":
            # GPT-2 c_attn contains Q, K, V combined. Slicing for Q.
            if "c_attn" in args.module_name:
                embed_dim = model.config.n_embd
                num_heads = model.config.n_head
                head_dim = embed_dim // num_heads
                # Q weight is the first 1/3
                weight = weight[:, :embed_dim].T 
                weight = weight.view(embed_dim, num_heads, head_dim)[:, args.head_idx, :]
            else:
                # Standard Conv1D [in, out]
                pass 
        else:
            # BERT/Llama Linear [out, in] -> Transpose to [in, out]
            weight = weight.t()
            if "attention" in args.module_name or "self_attn" in args.module_name:
                # Slice specific head if it's a Q/K/V/O matrix
                num_heads = model.config.num_attention_heads
                head_dim = weight.shape[1] // num_heads
                if weight.shape[1] % num_heads == 0 and weight.shape[1] > head_dim:
                    weight = weight.view(weight.shape[0], num_heads, head_dim)[:, args.head_idx, :]

        return weight
    except Exception as e:
        print(f"Extraction Error: {e}")
        return None

def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    
    # 1. Load Models and Weights
    model_class = AutoModelForCausalLM if args.model_type in ["gpt2", "llama"] else AutoModel
    
    print(f"[*] Loading models: {args.base_model} and {args.finetuned_path}")
    m_base = model_class.from_pretrained(args.base_model, cache_dir=args.cache_dir).to(device)
    m_ft = model_class.from_pretrained(args.finetuned_path, cache_dir=args.cache_dir).to(device)
    
    bs_w = extract_module_weight(m_base, args)
    ft_w = extract_module_weight(m_ft, args)
    
    del m_base, m_ft
    torch.cuda.empty_cache()

    if bs_w is None or ft_w is None:
        print("[!] Weight extraction failed. Exiting.")
        return

    bs_w = bs_w.to(torch_dtype)
    ft_w = ft_w.to(torch_dtype)
    d_model, d_head = bs_w.shape
    print(f"[*] Target Matrix Shape: {d_model}x{d_head}")

    # 2. Simulate TEMPO Perturbations (Simulation for testing)
    # Row Permutation
    row_perm = torch.randperm(d_model, device=device)
    ft_w = ft_w[row_perm]
    # Scaling
    ft_w = torch.diag(torch.rand(d_model, device=device) + 1).to(torch_dtype) @ ft_w
    ft_w = ft_w @ torch.diag(torch.rand(d_head, device=device) * 2 + 1).to(torch_dtype)
    # Column Permutation
    col_perm = torch.randperm(d_head, device=device)
    ft_w = ft_w[:, col_perm]
    
    print("[TEMPO] Applied Row/Col permutations and scaling for simulation.")

    # 3. Pre-compute Base Ratio Cache
    print("[Pre-compute] Caching base ratios and harmonic means...")
    bs_ratio_cache = {}
    bs_harm_cache = {}
    for p, q in itertools.permutations(range(d_head), 2):
        v_bs = bs_w[:, p] / (bs_w[:, q] + 1e-12)
        bs_ratio_cache[(p, q)] = v_bs
        bs_harm_cache[(p, q)] = 1.0 / (v_bs + 1.0 / (v_bs + 1e-12) + 1e-12)

    # 4. Attack: Disjoint Column Pairs
    test_pairs = [(k, k+1) for k in range(0, d_head, 2)]
    hit = 0
    quantiles = [20, 30, 40, 50, 60, 70, 80]

    print(f"\n[Attack] Starting TEMPO matching for {len(test_pairs)} pairs...")
    
    start_time = time.time()
    for idx, (i, j) in enumerate(test_pairs):
        v_ft = ft_w[:, i] / (ft_w[:, j] + 1e-12)
        all_res = []

        # Find best candidate (p, q) in base model
        for (p, q) in bs_ratio_cache.keys():
            # Estimate scale factor 't' via quantiles
            v_bs_q = np.percentile(bs_ratio_cache[(p, q)].detach().cpu().numpy(), quantiles)
            v_ft_q = np.percentile(v_ft.detach().cpu().numpy(), quantiles)
            t_estimates = v_ft_q / (v_bs_q + 1e-12)

            for t_est in t_estimates:
                v_ft_corr = v_ft / (t_est + 1e-12)
                harm_ft = 1.0 / (v_ft_corr + 1.0 / (v_ft_corr + 1e-12) + 1e-12)
                harm_bs = bs_harm_cache[(p, q)]
                
                # Hungarian algorithm for row matching cost
                cost = torch.abs(harm_ft[:, None] - harm_bs[None, :])
                # Optimization: linear_sum_assignment is CPU bound
                row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
                mean_dist = cost[row_idx, col_idx].mean().item()
                all_res.append(((p, q), mean_dist))

        # Sort and take mode (majority vote)
        all_res.sort(key=lambda x: x[1])
        top_candidates = [pair for pair, dist in all_res[:args.top_n]]
        best_pair, best_cnt = Counter(top_candidates).most_common(1)[0]

        # Verify against original column indices
        orig_i, orig_j = col_perm[i].item(), col_perm[j].item()
        is_hit = (best_pair[0], best_pair[1]) == (orig_i, orig_j)
        if is_hit: hit += 1

        print(f"Pair {idx+1:2d}: ft({i},{j}) -> orig({orig_i},{orig_j}) | Best({best_pair[0]},{best_pair[1]}) "
              f"| {'[OK]' if is_hit else '[FAIL]'} | mode={best_cnt}/{args.top_n}")

    end_time = time.time()
    print("\n" + "="*60)
    print(f"TEMPO Attack Result: {hit}/{len(test_pairs)} hits ({hit/len(test_pairs):.2%})")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()