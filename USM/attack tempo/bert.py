import torch
import numpy as np
import itertools
import argparse
import time
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="TEMPO Attack: BERT Weight Column Matching")
    
    # Path Parameters
    parser.add_argument("--base_model", type=str, default="bert-base-uncased", help="Base BERT model ID")
    parser.add_argument("--finetuned_path", type=str, required=True, help="Path to finetuned BERT")
    parser.add_argument("--cache_dir", type=str, default=None, help="Huggingface cache directory")
    
    # Extraction Parameters
    parser.add_argument("--layer_id", type=int, default=11, help="BERT layer index (0-11)")
    parser.add_argument("--module_name", type=str, default="attention.self.query", help="Sub-module path")
    parser.add_argument("--head_idx", type=int, default=0, help="Target attention head")
    
    # Attack Hyperparameters
    parser.add_argument("--top_n", type=int, default=7, help="Number of candidates for majority vote")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()

def extract_bert_weight(model, args):
    """
    Extracts and transposes BERT weights. 
    BERT uses [out_features, in_features]; we need [in_features, out_features].
    """
    try:
        layers = model.bert.encoder.layer if hasattr(model, 'bert') else model.encoder.layer
        block = layers[args.layer_id]
        
        target_module = block
        for attr in args.module_name.split('.'):
            target_module = getattr(target_module, attr)
        
        # Transpose [out, in] -> [in, out]
        full_weight = target_module.weight.detach().clone().t()
        
        # Slice head if it's an attention module
        if "attention" in args.module_name:
            num_heads = model.config.num_attention_heads
            head_dim = full_weight.shape[1] // num_heads
            return full_weight[:, args.head_idx * head_dim : (args.head_idx + 1) * head_dim]
        return full_weight
    except Exception as e:
        print(f"Extraction Error: {e}")
        return None

def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    
    # 1. Load Weights
    print(f"[*] Loading BERT models...")
    m_base = AutoModel.from_pretrained(args.base_model, cache_dir=args.cache_dir).to(device)
    m_ft = AutoModel.from_pretrained(args.finetuned_path, cache_dir=args.cache_dir).to(device)
    
    bs_w = extract_bert_weight(m_base, args).to(torch_dtype)
    ft_w = extract_bert_weight(m_ft, args).to(torch_dtype)
    
    del m_base, m_ft
    torch.cuda.empty_cache()
    
    d_model, d_head = bs_w.shape
    print(f"[*] Analyzing {d_model}x{d_head} matrix (Head {args.head_idx})")

    # 2. TEMPO Perturbation Simulation
    row_perm = torch.randperm(d_model, device=device)
    ft_w = ft_w[row_perm] # Row permutation
    ft_w = torch.diag(torch.rand(d_model, device=device) + 1).to(torch_dtype) @ ft_w # Row scale
    ft_w = ft_w @ torch.diag(torch.rand(d_head, device=device) * 2 + 1).to(torch_dtype) # Col scale
    col_perm = torch.randperm(d_head, device=device)
    ft_w = ft_w[:, col_perm] # Column permutation
    print("[TEMPO] Applied simulation perturbations.")

    # 3. Pre-compute Base Cache
    print("[Pre-compute] Caching ratios and harmonic means...")
    bs_ratio_cache = {}
    bs_harm_cache = {}
    for p, q in itertools.permutations(range(d_head), 2):
        v_bs = bs_w[:, p] / (bs_w[:, q] + 1e-12)
        bs_ratio_cache[(p, q)] = v_bs
        bs_harm_cache[(p, q)] = 1.0 / (v_bs + 1.0 / (v_bs + 1e-12) + 1e-12)

    # 4. Attack Loop
    test_pairs = [(k, k+1) for k in range(0, d_head, 2)]
    hit = 0
    quantiles = [20, 30, 40, 50, 60, 70, 80]

    start_time = time.time()
    for idx, (i, j) in enumerate(test_pairs):
        v_ft = ft_w[:, i] / (ft_w[:, j] + 1e-12)
        all_res = []

        for (p, q) in bs_ratio_cache.keys():
            v_bs_q = np.percentile(bs_ratio_cache[(p, q)].detach().cpu().numpy(), quantiles)
            v_ft_q = np.percentile(v_ft.detach().cpu().numpy(), quantiles)
            t_estimates = v_ft_q / (v_bs_q + 1e-12)

            for t_est in t_estimates:
                v_ft_corr = v_ft / (t_est + 1e-12)
                harm_ft = 1.0 / (v_ft_corr + 1.0 / (v_ft_corr + 1e-12) + 1e-12)
                harm_bs = bs_harm_cache[(p, q)]
                
                # Hungarian algorithm for row alignment
                cost = torch.abs(harm_ft[:, None] - harm_bs[None, :])
                r_idx, c_idx = linear_sum_assignment(cost.detach().cpu().numpy())
                mean_dist = cost[r_idx, c_idx].mean().item()
                all_res.append(((p, q), mean_dist))

        all_res.sort(key=lambda x: x[1])
        top_candidates = [pair for pair, dist in all_res[:args.top_n]]
        best_pair, best_cnt = Counter(top_candidates).most_common(1)[0]

        # Verify mapping
        orig_i, orig_j = col_perm[i].item(), col_perm[j].item()
        is_hit = (best_pair[0], best_pair[1]) == (orig_i, orig_j)
        if is_hit: hit += 1

        print(f"Pair {idx+1:2d}: ft({i},{j})->orig({orig_i},{orig_j}) | Best({best_pair[0]},{best_pair[1]}) "
              f"| {'[OK]' if is_hit else '[FAIL]'} | mode={best_cnt}/{args.top_n}")

    print(f"\nBERT Final Accuracy: {hit}/{len(test_pairs)} ({hit/len(test_pairs):.2%})")
    print(f"Time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()