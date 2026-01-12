import torch
from transformers import AutoModelForCausalLM
from collections import Counter
import time
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="ALS-based Weight Column Matching Attack for Llama")
    
    # Model and Path Parameters
    parser.add_argument("--base_model", type=str, required=True, help="Base Llama model ID or path")
    parser.add_argument("--finetuned_path", type=str, required=True, help="Path to the finetuned Llama model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Huggingface cache directory")
    
    # Dynamic Module Extraction Parameters (Llama specific)
    parser.add_argument("--layer_id", type=int, default=10, help="Llama layer index")
    parser.add_argument("--module_name", type=str, default="self_attn.o_proj", 
                        help="Path to Llama module (e.g., 'self_attn.q_proj', 'mlp.up_proj', 'mlp.down_proj')")
    
    # Algorithm Hyperparameters
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--perc", type=float, default=0.1)
    parser.add_argument("--row_of_equation", type=int, default=4)
    parser.add_argument("--op_iter", type=int, default=20)
    parser.add_argument("--attack_number", type=int, default=60)
    parser.add_argument("--topk", type=int, default=3)
    
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()

def extract_llama_module_weight(model, layer_id, module_name):
    """
    Extracts weights from Llama model. 
    Hierarchy: model.model.layers[i] or model.layers[i]
    Note: Llama uses nn.Linear [out, in], so we transpose to [in, out].
    """
    try:
        # Access the layers list
        if hasattr(model, 'model'):
            layers = model.model.layers
        else:
            layers = model.layers
            
        block = layers[layer_id]
        
        # Traverse the module path (e.g., self_attn.q_proj)
        target_module = block
        for attr in module_name.split('.'):
            target_module = getattr(target_module, attr)
            
        # Transpose to [in_features, out_features] for algorithm consistency
        return target_module.weight.detach().clone().t()
    except (AttributeError, IndexError) as e:
        print(f"Error: Llama Module '{module_name}' at Layer {layer_id} not found.")
        raise e

def iqr_mean_var(tensor, perc, dtype):
    """
    Filters outliers using IQR and returns mean and variance.
    """
    x = tensor.flatten().to(dtype)
    q1, q3 = torch.quantile(x, torch.tensor([perc, 1-perc], device=x.device, dtype=x.dtype))
    mask = (x > q1) & (x < q3)
    filtered = x[mask]
    if filtered.numel() == 0:
        nan = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
        return nan, nan
    return filtered.mean(), filtered.var(unbiased=False)

# ----------- ALS Core Mathematics (Same as GPT-2/BERT) -----------

def solve_ab_given_v_batch(w_batch, p_batch, v_batch, dtype, device):
    B = w_batch.shape[0]
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
    sol = torch.linalg.lstsq(A, rhs).solution
    return sol.squeeze(-1)

def solve_v_given_ab_batch(w_batch, p_batch, a1, b1, a2, b2, dtype, device):
    B = w_batch.shape[0]
    v_new = torch.zeros((B, 4), dtype=dtype, device=device)
    for i in range(4):
        w_even, w_odd = w_batch[:, 2*i], w_batch[:, 2*i+1]
        p_even, p_odd = p_batch[:, 2*i], p_batch[:, 2*i+1]
        r1, r2 = p_even - a1 * w_even, p_odd - a2 * w_odd
        denom = b1*b1 + b2*b2
        v_new[:, i] = torch.where(denom.abs() < 1e-20, 
                                  0.5 * (r1/b1.clamp(min=1e-12) + r2/b2.clamp(min=1e-12)), 
                                  (b1 * r1 + b2 * r2) / denom)
    return v_new

def als_batch(w_batch, p_batch, dtype, device, max_iters=10):
    B = w_batch.shape[0]
    v = torch.randn((B, 4), dtype=dtype, device=device) * 0.01
    for _ in range(max_iters):
        ab = solve_ab_given_v_batch(w_batch, p_batch, v, dtype, device)
        a1, b1, a2, b2 = ab[:,0], ab[:,1], ab[:,2], ab[:,3]
        v = solve_v_given_ab_batch(w_batch, p_batch, a1, b1, a2, b2, dtype, device)
    return v, a1, b1, a2, b2

def analyze_matching(W_b, W_f, a_batch, b_batch, v_batch, num_solutions, row_of_equation, topk, target_Wf_idx):
    device = a_batch.device
    d_head = W_b.shape[1]
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

    print(f"[*] Analyzing Llama - Layer: {args.layer_id}, Module: {args.module_name}")
    
    with torch.no_grad():
        # Llama weights are large; loading on CPU first is safer
        m_base = AutoModelForCausalLM.from_pretrained(args.base_model, cache_dir=args.cache_dir, torch_dtype=torch.float16).to('cpu')
        m_ft = AutoModelForCausalLM.from_pretrained(args.finetuned_path, cache_dir=args.cache_dir, torch_dtype=torch.float16).to('cpu')
        
        W_b_raw = extract_llama_module_weight(m_base, args.layer_id, args.module_name)
        W_f_raw = extract_llama_module_weight(m_ft, args.layer_id, args.module_name)

    del m_base, m_ft
    torch.cuda.empty_cache()

    W_b = W_b_raw.to(torch_dtype).to(device)
    W_f = W_f_raw.to(torch_dtype).to(device)
    d_model, d_head = W_b.shape
    num_solutions = d_model // args.row_of_equation
    W_b_blocks = W_b.view(num_solutions, args.row_of_equation, d_head)

    v_rand = torch.randn(d_model, dtype=torch_dtype, device=device)
    for i in range(d_head):
        a_r = torch.randint(1, 10, (1,), device=device).to(torch_dtype)
        b_r = torch.randint(1, 10, (1,), device=device).to(torch_dtype) - 5.0
        W_f[:, i] = a_r * W_f[:, i] + b_r * v_rand

    voter_dict = {i: [] for i in range(d_head)}
    start_t = time.time()

    for epoch in range(args.attack_number):
        r1, r2 = torch.randint(0, d_head, (2,), device=device).tolist()
        p1_b = W_f[:, r1].view(num_solutions, args.row_of_equation)
        p2_b = W_f[:, r2].view(num_solutions, args.row_of_equation)
        
        var_map = torch.zeros((d_head, d_head), dtype=torch_dtype, device=device)
        avg_a1 = torch.zeros((d_head, d_head), dtype=torch_dtype, device=device)
        avg_a2 = torch.zeros((d_head, d_head), dtype=torch_dtype, device=device)
        epoch_data = {}

        for c1 in range(d_head - 1, -1, -1):
            c2_list = [j for j in range(d_head) if j != c1]
            w_batch = torch.stack((W_b_blocks[:, :, c1].unsqueeze(-1).expand(-1, -1, len(c2_list)), W_b_blocks[:, :, c2_list]), dim=2).permute(3, 0, 1, 2).reshape(-1, 8)
            p_batch = torch.stack((p1_b, p2_b), dim=2).unsqueeze(0).expand(len(c2_list), -1, -1, -1).reshape(-1, 8)
            
            v_b, a1_b, b1_b, a2_b, b2_b = als_batch(w_batch, p_batch, torch_dtype, device, args.op_iter)
            
            v_b = v_b.view(len(c2_list), num_solutions, 4)
            a1_b, a2_b = a1_b.view(len(c2_list), num_solutions), a2_b.view(len(c2_list), num_solutions)
            b1_b = b1_b.view(len(c2_list), num_solutions)

            for idx, c2 in enumerate(c2_list):
                m1, v1 = iqr_mean_var(a2_b[idx], args.perc, torch_dtype)
                m2, v2 = iqr_mean_var(a1_b[idx], args.perc, torch_dtype)
                var_map[c1, c2] = v1 + v2
                avg_a1[c1, c2], avg_a2[c1, c2] = m1, m2
                epoch_data[(c1, c2)] = (v_b[idx], a1_b[idx], b1_b[idx])

        flat_v = var_map.clone()
        flat_v[torch.eye(d_head).bool()] = float('inf')
        best = torch.argmin(flat_v)
        f_t, s_t = (best // d_head).item(), (best % d_head).item()
        
        if avg_a1[f_t, s_t] > 0.8:
            v_f, a_f, b_f = epoch_data[(f_t, s_t)]
            for i in range(d_head):
                winners = analyze_matching(W_b, W_f, a_f, b_f, v_f, num_solutions, args.row_of_equation, args.topk, i)
                for w_idx, _ in winners:
                    voter_dict[i].append(w_idx)
        print(f"Epoch {epoch+1}/{args.attack_number} processed.")

    correct = 0
    for i in range(d_head):
        votes = Counter(voter_dict[i]).most_common(1)
        pred = votes[0][0] if votes else -1
        if pred == i: correct += 1
        print(f"Wf Column {i:3d} -> Predicted Wb Column {pred:3d} | {'OK' if pred == i else 'FAIL'}")
    
    print(f"\nLlama Matching Success Rate: {correct/d_head:.2%} ({correct}/{d_head})")

if __name__ == "__main__":
    main()