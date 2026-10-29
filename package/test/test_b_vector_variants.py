import os
import numpy as np
import pandas as pd
from copy import deepcopy

# Import your package modules
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
import openflash.multi_equations as me  

# Import ALL_CONFIGS from your existing test file
from test_zeroes_matrices import ALL_CONFIGS

# 1. Define the Variations Factory
def create_b_variant(variant_name):
    """Returns a newly constructed b_potential_entry function based on the variant logic."""
    def b_potential_entry_custom(n, i, d, heaving, h, a):
        # Identify shorter (j) and taller (k) regions
        j = i + (d[i] <= d[i+1])
        k = i + 1 if j == i else i 
        
        if variant_name == "Original_Right_Minus_Left":
            constant = (float(heaving[i+1]) / (h - d[i+1]) - float(heaving[i]) / (h - d[i]))
        elif variant_name == "Left_Minus_Right": # causes all potential configs to fail
            constant = (float(heaving[i]) / (h - d[i]) - float(heaving[i+1]) / (h - d[i+1]))
        elif variant_name == "Shorter_Minus_Taller": # causes all potential configs to fail except for 7,8,11,14
            constant = (float(heaving[j]) / (h - d[j]) - float(heaving[k]) / (h - d[k]))
        elif variant_name == "Taller_Minus_Shorter": #caused all configs to fail except for 0, 1, 4, 5
            constant = (float(heaving[k]) / (h - d[k]) - float(heaving[j]) / (h - d[j]))
        elif variant_name == "Diff_Over_Shorter_Depth_Original": # caused 1, 2, 3, 6, 9, 10 to fail
            constant = (float(heaving[i+1]) - float(heaving[i])) / (h - d[j])
        elif variant_name == "Diff_Over_Shorter_Depth_Short_Minus_Tall": # caused 1, 2, 3, 4, 5, 6, 9, 10 to fail
            constant = (float(heaving[j]) - float(heaving[k])) / (h - d[j])
        else:
            raise ValueError("Unknown variant")

        if n == 0:
            return constant * 1/2 * ((h - d[j])**3/3 - (h-d[j]) * a[i]**2/2)
        else:
            return np.sqrt(2) * (h - d[j]) * constant * ((-1) ** n)/(me.lambda_ni(n, j, h, d) ** 2)
            
    return b_potential_entry_custom

# 2. Adapted Flip Analyzer (Now extracts zero indices and uses Superposition to avoid Engine crashes)
def analyze_b_zeros_flip(config_name, p):
    """Runs a Case 1 vs Case 2 flip and returns zero counts and indices for the b vector."""
    eps = 1e-5
    m0 = p["m0"]
    h = p["h"]
    
    # Need at least 2 domains to flip a boundary
    if len(p["d"]) < 2:
        return None, None, None, None, None

    target_d = p["d"][1] 
    
    d_case1 = p["d"].copy()
    d_case1[0] = target_d + eps # Case 1: Inner deeper
    
    d_case2 = p["d"].copy()
    d_case2[0] = target_d - eps # Case 2: Inner shallower

    def get_b_vector_superposed(d_array):
        b_total = None
        
        # Superposition: Run the engine for each active body and sum the b-vectors
        for idx, is_heaving in enumerate(p["heaving_map"]):
            if not is_heaving: 
                continue
                
            # Isolate a single heaving body to keep the engine happy
            single_heaving_map = [False] * len(p["heaving_map"])
            single_heaving_map[idx] = True
            
            geom = BasicRegionGeometry.from_vectors(
                a=p["a"], d=d_array, h=h, NMK=p["NMK"], 
                heaving_map=single_heaving_map, body_map=p["body_map"]
            )
            problem = MEEMProblem(geom)
            engine = MEEMEngine([problem])
            engine._ensure_m_k_and_N_k_arrays(problem, m0)
            
            b_partial = engine.assemble_b_multi(problem, m0)
            
            if b_total is None:
                b_total = np.zeros_like(b_partial)
            b_total += b_partial
            
        # Fallback if no bodies are heaving
        if b_total is None:
            geom = BasicRegionGeometry.from_vectors(
                a=p["a"], d=d_array, h=h, NMK=p["NMK"], 
                heaving_map=[False]*len(p["heaving_map"]), body_map=p["body_map"]
            )
            problem = MEEMProblem(geom)
            engine = MEEMEngine([problem])
            engine._ensure_m_k_and_N_k_arrays(problem, m0)
            b_total = engine.assemble_b_multi(problem, m0)
            
        return b_total

    b1 = get_b_vector_superposed(d_case1)
    b2 = get_b_vector_superposed(d_case2)

    zeros_idx_1 = np.where(np.abs(b1) < 1e-10)[0]
    zeros_idx_2 = np.where(np.abs(b2) < 1e-10)[0]
    
    return len(zeros_idx_1), len(zeros_idx_2), len(b1), zeros_idx_1, zeros_idx_2

# 3. Main Execution Loop
def run_variant_sweep(config_name, config_params):
    print(f"\n{'='*60}")
    print(f"--- Sweeping Combinations for {config_name} ---")
    
    variants = [
        "Original_Right_Minus_Left",
        "Left_Minus_Right",
        "Shorter_Minus_Taller",
        "Taller_Minus_Shorter",
        "Diff_Over_Shorter_Depth_Original",
        "Diff_Over_Shorter_Depth_Short_Minus_Tall"
    ]
    
    results = []
    original_b_func = me.b_potential_entry 
    
    for var in variants:
        me.b_potential_entry = create_b_variant(var)
        
        try:
            z1, z2, total, idx1, idx2 = analyze_b_zeros_flip(config_name, config_params)
            
            if z1 is None:
                print(f"Skipping {config_name} (Not enough domains to flip)")
                me.b_potential_entry = original_b_func
                return None
                
            match = "✅" if z1 == z2 else "❌"
            
            # For the baseline, print exactly where the zeros are
            if var == "Original_Right_Minus_Left":
                print(f"  [Diagnostics for Original Formula]")
                print(f"  Case 1 (d0 > d1) Zero Indices ({z1}):\n  {list(idx1)}")
                print(f"  Case 2 (d0 < d1) Zero Indices ({z2}):\n  {list(idx2)}")
                diff_1_to_2 = set(idx2) - set(idx1)
                diff_2_to_1 = set(idx1) - set(idx2)
                print(f"  Zeros unique to Case 1: {list(diff_2_to_1)}")
                print(f"  Zeros unique to Case 2: {list(diff_1_to_2)}\n")

            results.append({
                "Variation": var,
                "Case 1 Zeros": z1,
                "Case 2 Zeros": z2,
                "Total Elements": total,
                "Match": match
            })
        except Exception as e:
            results.append({
                "Variation": var,
                "Case 1 Zeros": "ERROR",
                "Case 2 Zeros": "ERROR",
                "Total Elements": "ERROR",
                "Match": "❌"
            })
            
    me.b_potential_entry = original_b_func
    
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    return df

if __name__ == "__main__":
    # Loop over all configurations
    for name, params in ALL_CONFIGS.items():
        run_variant_sweep(name, params)