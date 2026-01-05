from functools import cache
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from openflash.geometry import Geometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.domain import Domain
from openflash.multi_equations import I_mk, N_k_multi, diff_R_1n, diff_Lambda_k, scale, v_dense_block_e_entry, v_diagonal_block_e, v_diagonal_block_e_entry
from openflash.geometry import ConcentricBodyGroup
from openflash.body import SteppedBody
from openflash.basic_region_geometry import BasicRegionGeometry

# --- Path Setup ---
current_dir = os.path.dirname(__file__)

# Add hydro/python for old code
old_code_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'dev', 'python'))
if old_code_dir not in sys.path:
    sys.path.insert(0, old_code_dir)

from old_assembly import assemble_old_A_and_b, I_mk_old, m_k_old, N_k_old, diff_R_1n_old, diff_Lambda_k_old, scale_old


def diagnose_large_differences(A_old, A_new, threshold=1e-6):
    """
    Identify large differences in matrix entries and report their locations and values.
    """
    diff = A_old - A_new
    abs_diff = np.abs(diff)
    
    max_diff = np.max(abs_diff)
    print(f"Max absolute difference in A: {max_diff}")
    
    # Get indices where difference is above threshold
    large_diff_indices = np.argwhere(abs_diff > threshold)
    print(f"Number of entries with difference > {threshold}: {len(large_diff_indices)}")
    
    # Sort indices by difference descending
    sorted_indices = sorted(large_diff_indices, key=lambda idx: abs_diff[tuple(idx)], reverse=True)
    
    # Print top few largest differences
    print("Top differences (index, A_old, A_new, diff):")
    for idx in sorted_indices[:10]:
        i, j = idx
        print(f"({i}, {j}): {A_old[i,j]} vs {A_new[i,j]} => diff={diff[i,j]}")
    return sorted_indices[:10]

def extract_block_from_assembly(assembly_func, h, d, a, NMK, heaving, m0, block_position):
    """
    Given an assembly function (e.g. assemble_old_A_and_b), parameters, and a matrix block position,
    extract the block matrix that corresponds to that position.
    
    block_position: (row_start, row_end, col_start, col_end)
    """
    # Call the assembly function but modify it to expose blocks or
    # reconstruct blocks for this position
    
    # Since assembly function currently returns the full matrix,
    # can simply slice the returned matrix after building it:
    A, _ = assembly_func(h, d, a, NMK, heaving, m0)
    row_start, row_end, col_start, col_end = block_position
    return A[row_start:row_end, col_start:col_end]

def compare_blocks(A_old, A_new, block_positions):
    """
    Given old and new matrices and block positions (list of tuples),
    print side-by-side comparison of corresponding blocks.
    """
    for idx, (rs, re, cs, ce) in enumerate(block_positions):
        print(f"\nBlock {idx} (rows {rs}:{re}, cols {cs}:{ce}):")
        old_block = A_old[rs:re, cs:ce]
        new_block = A_new[rs:re, cs:ce]
        diff_block = old_block - new_block
        print(f"Old block:\n{old_block}")
        print(f"New block:\n{new_block}")
        print(f"Difference:\n{diff_block}")
        
def validate_closures(problem, engine, m0, tol=1e-12):
    cache = engine.cache_list[problem]
    engine._ensure_m_k_and_N_k_arrays(problem, m0)
    m_k_arr = cache.m_k_arr
    N_k_arr = cache.N_k_arr
    I_mk_vals = cache._get_closure("I_mk_vals")(m0, m_k_arr, N_k_arr)

    print("\n--- Validating m₀-dependent A closures ---")
    for (row, col, closure_fn) in cache.m0_dependent_A_indices:
        closure_val = closure_fn(problem, m0, m_k_arr, N_k_arr, I_mk_vals)
            
        # Compute expected value (replicating logic manually here is tedious, but try a few hand-picked examples)
        # Let's retrieve the current value from the assembled matrix for now:
        A_matrix = engine.assemble_A_multi(problem, m0)
        matrix_val = A_matrix[row, col]

        if not np.isclose(closure_val, matrix_val, rtol=tol, atol=tol):
            print(f"[A mismatch] row={row}, col={col}: closure={closure_val:.4e}, matrix={matrix_val:.4e}")

    print("\n--- Validating m₀-dependent b closures ---")
    b_vector = engine.assemble_b_multi(problem, m0)
    for (row, closure_fn) in cache.m0_dependent_b_indices:
        closure_val = closure_fn(problem, m0, m_k_arr, N_k_arr, I_mk_vals)
        b_vector = engine.assemble_b_multi(problem, m0) # Re-fetch just in case
        b_val = b_vector[row]
        if not np.isclose(closure_val, b_val, rtol=tol, atol=tol):
            print(f"[b mismatch] row={row}: closure={closure_val:.4e}, vector={b_val:.4e}")

def compare_matrices_and_vectors(A_old, b_old, A_new, b_new, tol=1e-10):
    print("=== Shape Check ===")
    print(f"A_old shape: {A_old.shape}")
    print(f"A_new shape: {A_new.shape}")
    print(f"b_old shape: {b_old.shape}")
    print(f"b_new shape: {b_new.shape}")
    
    print("A_old:\n", np.round(A_old.real, 2))
    print("A_new:\n", np.round(A_new.real, 2))
    print("b_old:\n", np.round(b_old.real, 2))
    print("b_new:\n", np.round(b_new.real, 2))

    if A_old.shape != A_new.shape:
        print("Warning: Matrix shapes differ!")

    if b_old.shape != b_new.shape:
        print("Warning: RHS vector shapes differ!")

    print("\n=== Norms ===")
    print(f"||A_old||_F = {np.linalg.norm(A_old, 'fro')}")
    print(f"||A_new||_F = {np.linalg.norm(A_new, 'fro')}")
    print(f"||b_old||_2 = {np.linalg.norm(b_old)}")
    print(f"||b_new||_2 = {np.linalg.norm(b_new)}")

    print("\n=== Elementwise Differences ===")
    # Pad smaller matrix/vector if shapes differ, to avoid error
    min_rows = min(A_old.shape[0], A_new.shape[0])
    min_cols = min(A_old.shape[1], A_new.shape[1])
    A_diff = A_old[:min_rows, :min_cols] - A_new[:min_rows, :min_cols]
    print(f"Max abs difference in A: {np.max(np.abs(A_diff))}")
    print(f"Number of differing elements in A > {tol}: {(np.abs(A_diff) > tol).sum()}")

    min_len = min(b_old.size, b_new.size)
    b_diff = b_old[:min_len] - b_new[:min_len]
    print(f"Max abs difference in b: {np.max(np.abs(b_diff))}")
    print(f"Number of differing elements in b > {tol}: {(np.abs(b_diff) > tol).sum()}")

    print("\n=== Zero Rows in Matrices ===")
    zero_rows_old = np.where(~A_old.any(axis=1))[0]
    zero_rows_new = np.where(~A_new.any(axis=1))[0]
    print(f"Zero rows in A_old: {zero_rows_old}")
    print(f"Zero rows in A_new: {zero_rows_new}")

    # Optional: print detailed blocks if large differences
    if np.max(np.abs(A_diff)) > tol:
        print("\nBlocks with largest differences (some rows):")
        max_diff_indices = np.unravel_index(np.argmax(np.abs(A_diff)), A_diff.shape)
        print(f"Max difference at position {max_diff_indices}, value = {A_diff[max_diff_indices]}")
        
def summarize_array_differences(arr1, arr2, name1="arr1", name2="arr2", rtol=1e-12, atol=1e-15):
    print(f"\nComparing {name1} and {name2}")
    for i, (v1, v2) in enumerate(zip(arr1, arr2)):
        print(f"{i}: {v1:.15f} vs {v2:.15f} => close? {np.isclose(v1, v2, rtol=rtol, atol=atol)}")


def compare_v_diagonal_block_e(bd, NMK, a, h, m0, m_k_arr, atol=1e-12, rtol=1e-8):
    print(f"--- Comparing v_diagonal_block_e vs. v_diagonal_block_e_entry at bd={bd} ---")
    
    # Generate the full diagonal block using the block function
    block = v_diagonal_block_e(bd, h, NMK, a, m0, m_k_arr) # Call without k,r as fixed args

    num_k_modes = NMK[bd+1] # Number of modes for this block (M)
    all_close = True

    for m in range(num_k_modes): # m is the local row index, corresponds to k mode in diag
        # The diagonal entry from the block
        val_matrix = block[m, m] 
        
        # The scalar entry from the entry function for the diagonal
        # Here, 'k' in v_diagonal_block_e_entry is the mode index, which is 'm' for diagonal.
        # This function takes the individual parameters needed.
        val_entry = v_diagonal_block_e_entry(m, m, bd, m0, m_k_arr, a, h) # k=m for diagonal
        
        if not np.isclose(val_matrix, val_entry, atol=atol, rtol=rtol):
            print(f"Mismatch at index {m}:")
            print(f"  matrix: {val_matrix}")
            print(f"  entry : {val_entry}")
            all_close = False

    if all_close:
        print("✅ All entries match!")
    else:
        print("❌ Some entries did not match. See above for details.")
        
def test_v_dense_block_e_entry():
    m, k, bd = 0, 0, 2
    # Setup dummy I_mk_vals of appropriate shape
    I_mk_vals = np.eye(3, dtype=complex)
    a = [1.0, 2.0, 10.0]
    h = 1.5
    d = [0.0, 0.0, 0.0]
    val = v_dense_block_e_entry(m, k, bd, I_mk_vals, a, h, d)
    print("Entry (0, 0):", val)
        
def run_comparison_test():
    # Define a small test problem parameters here (example)
    NMK = [20, 20, 20, 20]   # Small problem size for fast testing
    d = [29, 7, 4]    # example depths
    a = [3, 5, 10]          # example parameters per region
    
    # --- FIX: Ensure only ONE body is heaving to satisfy Geometry assertion ---
    heaving = [0, 1, 0]     
    
    h = 100                 # example characteristic length
    m0 = 1
    # --- Assemble old matrix and vector ---
    #  need to implement or import the old assemble function here
    A_old, b_old = assemble_old_A_and_b(h, d, a, NMK, heaving, m0)

    # --- Setup problem and m0 for package assembly ---
    #  must implement or mock a 'problem' object and m0 index for  new code
    # --- Geometry Setup ---
    bodies = []
    for i in range(len(a)):
        body = SteppedBody(
            a=np.array([a[i]]),
            d=np.array([d[i]]),
            slant_angle=np.array([0.0]),  # Assuming zero slant for the test
            heaving=bool(heaving[i])
        )
        bodies.append(body)
    
    # 2. Create the body arrangement.
    arrangement = ConcentricBodyGroup(bodies)

    # 3. Instantiate the CONCRETE geometry class.
    #    This object will now correctly create the fluid domains internally.
    geometry = BasicRegionGeometry(arrangement, h, NMK)
    
    problem = MEEMProblem(geometry)

    # --- MEEM Engine Operations ---
    engine = MEEMEngine(problem_list=[problem])
    engine.build_problem_cache(problem)

    problem_cache = engine.cache_list[problem]
    engine._ensure_m_k_and_N_k_arrays(problem, m0)
    
    m_k_arr = problem_cache.m_k_arr
    N_k_arr = problem_cache.N_k_arr
    m_k_old_arr = m_k_old(NMK, m0, h)

    # Debugging: Check the values of m_k_arr and N_k_arr before passing to plotting functions
    print(f"DEBUG: m_k_arr in main() before plotting functions: {m_k_arr.shape if m_k_arr is not None else 'None'}")
    print(f"DEBUG: N_k_arr in main() before plotting functions: {N_k_arr.shape if N_k_arr is not None else 'None'}")
    
    def compare_N_k_old_vs_new(m0, h, NMK, m_k_old_arr, m_k_arr):
        print(f"Comparing N_k_old vs N_k_multi for m0={m0}, h={h}")

        print("k\tN_k_old\t\tN_k_multi\tClose?")
        for k in range(NMK[-1]):
            old_val = N_k_old(k, m0, h, m_k_old_arr)
            new_val = N_k_multi(k, m0, h, m_k_arr)
            close = np.isclose(old_val, new_val, rtol=1e-12, atol=1e-15)
            print(f"{k}\t{old_val:.15f}\t{new_val:.15f}\t{close}")
            
    compare_N_k_old_vs_new(m0, h, NMK, m_k_old_arr, m_k_arr)

    boundary_count = len(NMK) - 1

    print("Old m_k[0]:", m_k_old_arr[0])
    print("New m_k[0]:", m_k_arr[0])
    print("Equal?", np.isclose(m_k_old_arr[0], m_k_arr[0], rtol=1e-12, atol=1e-15))
    
    print("Old m_k[1]:", m_k_old_arr[1])
    print("New m_k[1]:", m_k_arr[1])
    print("Equal?", np.isclose(m_k_old_arr[1], m_k_arr[1], rtol=1e-12, atol=1e-15))
    
    print(f"m_k_old[1]: {m_k_old_arr[1]} (from m_k_entry_old)")
    print(f"m_k_new[1]: {m_k_arr[1]} (from m_k_entry)")


    print("Old I_mk[0,0]:", I_mk_old(0, 0, boundary_count - 1, d, h, m0, NMK))
    print("New I_mk[0,0]:", I_mk(0, 0, boundary_count - 1, d, m0, h, m_k_arr, N_k_arr))
    
    summarize_array_differences(
        [N_k_old(k, m0, h, m_k_old_arr) for k in range(NMK[-1])],
        [N_k_multi(k, m0, h, m_k_arr) for k in range(NMK[-1])],
        "N_k_old", "N_k_multi"
    )

    # --- Assemble new matrix and vector ---
    A_new = engine.assemble_A_multi(problem, m0)
    b_new = engine.assemble_b_multi(problem, m0)

    # --- Compare ---
    compare_matrices_and_vectors(A_old, b_old, A_new, b_new)
    diagnose_large_differences(A_old, A_new, threshold=1e8)

    print("A_old[4:6, 10:12]:", A_old[4:6, 10:12])
    print("A_new[4:6, 10:12]:", A_new[4:6, 10:12])
    print("A_old[10:12, 6:10]:", A_old[10:12, 6:10])
    print("A_new[10:12, 6:10]:", A_new[10:12, 6:10])
    
    print("Normalization in I_mk_old vs I_mk")
    print(I_mk_old(0, 0, boundary_count - 1, d, h, m0, NMK))
    print(I_mk(0, 0, boundary_count - 1, d, m0, h, m_k_arr, N_k_arr))

    print("Normalizing constants comparison:")
    print("h (height):", h)
    print("Depths (d):", d)
    print("Widths (a):", a)
    
    print("\nAre A_old and A_new close (atol=1e-10)?", np.allclose(A_old, A_new, atol=1e-10))
    
    print(f"\nTotal m₀-dependent A entries registered: {len(problem_cache.m0_dependent_A_indices)}")
    print(f"Total m₀-dependent b entries registered: {len(problem_cache.m0_dependent_b_indices)}")
    
    diff_indices = np.where(np.abs(A_old - A_new) > 1e-10)
    print("Differing indices:", list(zip(diff_indices[0], diff_indices[1]))[:20])

    for i, j in [(99,38), (20,20), (20, 40), (80,20), (80,40), (81,20), (81,21), (81,22), (81,23), (81,24), (81,25),(81,26),(81,27),(81,28),(81,29),(81,30),(81,31),(81,32),(81,33),(81,34),(81,35)]:
        print(f"A_old[{i}, {j}] = {A_old[i, j]}")
        print(f"A_new[{i}, {j}] = {A_new[i, j]}")
        
    print("scale_old(a)[-1]:", scale_old(a)[-1])
    print("scale(a)[-1]:", scale(a)[-1])
    
    bd_test = 2 
     
    validate_closures(problem, engine, m0)
    
    # In test_matrices.py, somewhere after parameters are defined
    n_test = 18
    r_test = a[2] # which is 10
    i_test = 2

    old_diff_R_1n_val = diff_R_1n_old(n_test, r_test, i_test, h, d, a)
    new_diff_R_1n_val = diff_R_1n(n_test, r_test, i_test, h, d, a) # Pass 'a' to new diff_R_1n

    print(f"\n--- Direct Comparison for diff_R_1n (n={n_test}, r={r_test}, i={i_test}) ---")
    print(f"Old diff_R_1n_old result: {old_diff_R_1n_val}")
    print(f"New diff_R_1n result: {new_diff_R_1n_val}")
    print(f"Are they close? {np.isclose(old_diff_R_1n_val, new_diff_R_1n_val, atol=1e-10)}")
    
    k_test = 0
    
    compare_v_diagonal_block_e(bd_test, NMK, a, h, m0, m_k_arr)


    old_diff_Lambda_k_val = diff_Lambda_k_old(k_test, r_test, m0, a, NMK, h)
    new_diff_Lambda_k_val = diff_Lambda_k(k_test, r_test, m0, a, m_k_arr)
    print(f"\n--- Direct Comparison for diff_Lambda_k (k={k_test}, r={r_test}, m0={m0}, a={a}, m_k_arr={m_k_arr}, NMK={NMK}, h={h}) ---")
    print(f"Old diff_Lamda_k_old result: {old_diff_Lambda_k_val}")
    print(f"New diff_Lambda_k result: {new_diff_Lambda_k_val}")
    print(f"Are they close? {np.isclose(old_diff_Lambda_k_val, new_diff_Lambda_k_val, atol=1e-10)}")
    print(f"Are they close (real)? {np.isclose(old_diff_Lambda_k_val.real, new_diff_Lambda_k_val.real, atol=1e-10)}")
    print(f"Are they close (imag)? {np.isclose(old_diff_Lambda_k_val.imag, new_diff_Lambda_k_val.imag, atol=1e-10)}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot A_old
    im0 = axes[0].imshow(np.abs(A_old), cmap="viridis")
    axes[0].set_title("A_old (|values|)")
    fig.colorbar(im0, ax=axes[0])

    # Plot A_new
    im1 = axes[1].imshow(np.abs(A_new), cmap="viridis")
    axes[1].set_title("A_new (|values|)")
    fig.colorbar(im1, ax=axes[1])

    # Plot difference
    diff = A_old - A_new
    im2 = axes[2].imshow(np.abs(diff), cmap="hot")
    axes[2].set_title("|A_old - A_new| (abs diff)")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")

    plt.suptitle("Matrix Comparison: A_old vs A_new", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # A_old
    im0 = axes[0].imshow(A_old.real, cmap="viridis")
    axes[0].set_title("A_old (real)")
    fig.colorbar(im0, ax=axes[0])

    # A_new
    im1 = axes[1].imshow(A_new.real, cmap="viridis")
    axes[1].set_title("A_new (real)")
    fig.colorbar(im1, ax=axes[1])

    # Signed diff heatmap
    diff_signed = A_old.real - A_new.real
    im2 = axes[2].imshow(diff_signed, cmap="seismic", vmin=-np.max(np.abs(diff_signed)), vmax=np.max(np.abs(diff_signed)))
    axes[2].set_title("Signed Difference (A_old - A_new)")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")

    plt.suptitle("Matrix Comparison with Signed Difference", fontsize=16)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    run_comparison_test()
    test_v_dense_block_e_entry()