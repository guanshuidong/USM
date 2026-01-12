import torch
from transformers import AutoModelForCausalLM
from collections import Counter
import time
import argparse
import os
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description="ALS-based Weight Column Matching Attack for GPT-2")
    
    # Model and Path Parameters
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model ID or local path")
    parser.add_argument("--finetuned_path", type=str, required=True, help="Path to the finetuned model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Huggingface cache directory")
    
    # Dynamic Module Extraction Parameters
    parser.add_argument("--layer_id", type=int, default=11, help="Transformer block index (0-11 for GPT-2 Base)")
    parser.add_argument("--module_name", type=str, default="attn.c_proj", 
                        help="Attribute path to the module (e.g., 'attn.c_attn', 'mlp.c_proj')")
    
    # Algorithm Hyperparameters
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"], help="Numerical precision")
    parser.add_argument("--perc", type=float, default=0.1, help="Quantile percentage for IQR filtering")
    parser.add_argument("--row_of_equation", type=int, default=4, help="Block size for row partitioning")
    parser.add_argument("--op_iter", type=int, default=20, help="Max ALS iterations")
    parser.add_argument("--attack_number", type=int, default=60, help="Number of random attack trials")
    parser.add_argument("--topk", type=int, default=3, help="Top K results to vote per epoch")
    
    # Device Control
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., 'cuda:0', 'cpu')")

    return parser.parse_args()

def extract_specific_module_weight(model, layer_id, module_name):
    """
    Recursively extracts the weight matrix based on layer_id and module_name.
    Optimized for GPT-2 structure: model.transformer.h[layer_id].module
    """
    try:
        # Navigate to the specific transformer block
        block = model.transformer.h[layer_id]
        
        # Recursively enter sub-modules (e.g., attn -> c_proj)
        target_module = block
        for attr in module_name.split('.'):
            target_module = getattr(target_module, attr)
            
        # Return a copy of the weight. 
        # Note: GPT-2 Conv1D weights are stored as [dim_in, dim_out].
        return target_module.weight.detach().clone()
    except (AttributeError, IndexError) as e:
        print(f"Error: Module '{module_name}' at Layer {layer_id} not found.")
        raise e

def iqr_mean_var(tensor, perc, dtype):
    """
    Calculates mean and variance after filtering outliers using the Interquartile Range (IQR).
    """
    x = tensor.flatten().to(dtype)
    q1, q3 = torch.quantile(x, torch.tensor([perc, 1-perc], device=x.device, dtype=x.dtype))
    mask = (x > q1) & (x < q3)
    filtered = x[mask]
    if filtered.numel() == 0:
        nan = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
        return nan, nan
    return filtered.mean(), filtered.var(unbiased=False)

# ----------- ALS Core Batch Helpers -----------

def solve_ab_given_v_batch(w_batch, p_batch, v_batch, dtype, device):
    """
    Solves for coefficients a and b while fixing the vector v.
    """
    B = w_batch.shape[0]
    # Matrix A construction for 8 equations and 4 variables (a1, b1, a2, b2)
    A = torch.zeros((B, 8, 4), dtype=dtype, device=device)
    A[:, 0, 0] = w_batch[:, 0]; A[:, 0, 1] = v_batch[:, 0]
    A[:, 1, 2] = w_batch[:, 1]; A[:, 1, 3] = v_batch[:, 0]
    A[:, 2, 0] = w_batch[:, 2]; A[:, 2, 1] = v_batch[:, 1]
    A[:, 3, 2] = w_batch[:, 3]; A[:, 3, 3] = v_batch[:, 1]
    A[:, 4, 0] = w_batch[:, 4]; A[:, 4, 1] = v_batch[:, 2]
    A[:, 5, 2] = w_batch[:, 5]; A[:, 5, 3] = v_batch[:, 2]
    A[:, 6, 0] = w_batch[:, 6]; A[:, 6, 1] = v_batch[:, 3]
    A[:, 7, 2] = w_batch[:, 7]; A[:, 7, 3] = v_batch[:, 3]
    rhs = p_batch.unsqueeze(-1)
    # Solve using batch least squares
    sol = torch.linalg.lstsq(A, rhs).solution
    return sol.squeeze(-1)

def solve_v_given_ab_batch(w_batch, p_batch, a1, b1, a2, b2, dtype, device):
    """
    Solves for the shared vector v while fixing a and b.
    """
    B = w_batch.shape[0]
    v_new = torch.zeros((B, 4), dtype=dtype, device=device)
    for i in range(4):
        w_even, w_odd = w_batch[:, 2*i], w_batch[:, 2*i+1]
        p_even, p_odd = p_batch[:, 2*i], p_batch[:, 2*i+1]
        r1 = p_even - a1 * w_even
        r2 = p_odd - a2 * w_odd
        denom = b1*b1 + b2*b2
        
        # Robust division to prevent NaN from zero denominators
        small = denom.abs() < 1e-20
        normal = (b1 * r1 + b2 * r2) / denom
        fallback = 0.5 * (r1/b1.clamp(min=1e-12) + r2/b2.clamp(min=1e-12))
        v_new[:, i] = torch.where(small, fallback, normal)
    return v_new

def als_batch(w_batch, p_batch, dtype, device, max_iters=10):
    """
    Main ALS loop for a batch of equations.
    """
    B = w_batch.shape[0]
    v = torch.randn((B, 4), dtype=dtype, device=device) * 0.01
    for _ in range(max_iters):
        ab = solve_ab_given_v_batch(w_batch, p_batch, v, dtype, device)
        a1, b1, a2, b2 = ab[:,0], ab[:,1], ab[:,2], ab[:,3]
        v = solve_v_given_ab_batch(w_batch, p_batch, a1, b1, a2, b2, dtype, device)
    return v, a1, b1, a2, b2

def analyze_matching(W_b, W_f, a_batch, b_batch, v_batch, num_solutions, row_of_equation, topk, target_Wf_idx):
    """
    Uses the solved parameters to search for the best matching column in the base weight (W_b).
    """
    device = a_batch.device
    d_head = W_b.shape[1]
    
    # Filter for high-confidence blocks where 'a' is stable
    mean_val = a_batch.mean()
    sel_mask = (torch.abs(a_batch - mean_val) < 1.0)
    selected = torch.where(sel_mask)[0]
    if selected.numel() == 0:
        selected = torch.topk(-torch.abs(a_batch - torch.median(a_batch)), min(8, num_solutions)).indices

    Wb_blocks = W_b.reshape(num_solutions, row_of_equation, d_head)
    Wf_blocks = W_f.reshape(num_solutions, row_of_equation, d_head)
    
    s_sel = (b_batch[selected].unsqueeze(1) * v_batch[selected]).reshape(-1, 1)
    p_sel = Wf_blocks[selected, :, target_Wf_idx].reshape(-1, 1)
    Wb_sel = Wb_blocks[selected].reshape(-1, d_head)

    results = []
    for i in range(d_head):
        w_col = Wb_sel[:, i:i+1]
        A = torch.cat([w_col, s_sel], dim=1)
        # Solve using pseudo-inverse
        sol = torch.linalg.pinv(A) @ p_sel
        mse = torch.mean((p_sel - A @ sol)**2).item()
        results.append((i, mse))
    
    results.sort(key=lambda x: x[1])
    return results[:topk]

def main():
    args = parse_args()
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # 1. Load weights
    print(f"[*] Extracting GPT-2 Weights: Layer {args.layer_id}, Module {args.module_name}")
    with torch.no_grad():
        m_base = AutoModelForCausalLM.from_pretrained(args.base_model, cache_dir=args.cache_dir).to('cpu')
        m_ft = AutoModelForCausalLM.from_pretrained(args.finetuned_path, cache_dir=args.cache_dir).to('cpu')
        
        W_b_raw = extract_specific_module_weight(m_base, args.layer_id, args.module_name)
        W_f_raw = extract_specific_module_weight(m_ft, args.layer_id, args.module_name)
        
    del m_base, m_ft
    torch.cuda.empty_cache()

    W_b = W_b_raw.to(torch_dtype).to(device)
    W_f = W_f_raw.to(torch_dtype).to(device)
    d_model, d_head = W_b.shape
    num_solutions = d_model // args.row_of_equation
    W_b_blocks = W_b.view(num_solutions, args.row_of_equation, d_head)

    # Simulate random perturbation (original attack logic)
    v_rand = torch.randn(d_model, dtype=torch_dtype, device=device)
    for i in range(d_head):
        a_rand = torch.randint(1, 10, (1,), device=device).to(torch_dtype)
        b_rand = torch.randint(1, 10, (1,), device=device).to(torch_dtype) - 5.0
        W_f[:, i] = a_rand * W_f[:, i] + b_rand * v_rand

    voter_dict = {i: [] for i in range(d_head)}
    start_time = time.time()

    # 2. Attack Execution
    for epoch in range(args.attack_number):
        r1, r2 = torch.randint(0, d_head, (2,), device=device).tolist()
        p1_b = W_f[:, r1].view(num_solutions, args.row_of_equation)
        p2_b = W_f[:, r2].view(num_solutions, args.row_of_equation)
        
        var_map = torch.zeros((d_head, d_head), dtype=torch_dtype, device=device)
        avg_map_a1 = torch.zeros((d_head, d_head), dtype=torch_dtype, device=device)
        avg_map_a2 = torch.zeros((d_head, d_head), dtype=torch_dtype, device=device)
        epoch_results = {}

        # Scan pairs of columns to find the most stable fit
        for c1 in range(d_head - 1, -1, -1):
            c2_list = [j for j in range(d_head) if j != c1]
            w_batch = torch.stack((
                W_b_blocks[:, :, c1].unsqueeze(-1).expand(-1, -1, len(c2_list)),
                W_b_blocks[:, :, c2_list]
            ), dim=2).permute(3, 0, 1, 2).reshape(-1, 8)
            
            p_batch = torch.stack((p1_b, p2_b), dim=2).unsqueeze(0).expand(len(c2_list), -1, -1, -1).reshape(-1, 8)

            v_b, a1_b, b1_b, a2_b, b2_b = als_batch(w_batch, p_batch, torch_dtype, device, args.op_iter)
            
            v_b = v_b.view(len(c2_list), num_solutions, 4)
            a1_b, a2_b = a1_b.view(len(c2_list), num_solutions), a2_b.view(len(c2_list), num_solutions)
            b1_b, b2_b = b1_b.view(len(c2_list), num_solutions), b2_b.view(len(c2_list), num_solutions)

            for idx, c2 in enumerate(c2_list):
                m1, v1 = iqr_mean_var(a2_b[idx], args.perc, torch_dtype)
                m2, v2 = iqr_mean_var(a1_b[idx], args.perc, torch_dtype)
                var_map[c1, c2] = v1 + v2
                avg_map_a1[c1, c2], avg_map_a2[c1, c2] = m1, m2
                epoch_results[(c1, c2)] = (v_b[idx], a1_b[idx], b1_b[idx], a2_b[idx], b2_b[idx])

        # Pick the pair with minimum variance
        flat_var = var_map.clone()
        flat_var[torch.eye(d_head).bool()] = float('inf')
        best_idx = torch.argmin(flat_var)
        f_true, s_true = (best_idx // d_head).item(), (best_idx % d_head).item()
        
        if avg_map_a1[f_true, s_true] > 0.8: 
            v_f, a_f, b_f, _, _ = epoch_results[(f_true, s_true)]
            for i in range(d_head):
                winners = analyze_matching(W_b, W_f, a_f, b_f, v_f, num_solutions, args.row_of_equation, args.topk, i)
                for w_idx, _ in winners:
                    voter_dict[i].append(w_idx)

        print(f"Epoch {epoch+1}/{args.attack_number} - Pair ({f_true},{s_true})")

    # 3. Final Evaluation
    print("\n" + "="*50)
    print(f"GPT-2 Match Report (Layer {args.layer_id})")
    print("="*50)
    correct = 0
    for i in range(d_head):
        votes = Counter(voter_dict[i]).most_common(1)
        pred = votes[0][0] if votes else -1
        is_correct = (pred == i)
        if is_correct: correct += 1
        print(f"Wf Col {i:3d} -> Wb Col {pred:3d} | {'SUCCESS' if is_correct else 'FAILURE'}")
    
    print("-" * 50)
    print(f"Matching Accuracy: {correct/d_head:.2%}")
    print(f"Elapsed Time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()