import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.integrate import quad
import matplotlib.pyplot as plt

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)
from equations import (
    phi_p_i1, phi_p_i2, diff_phi_i1, diff_phi_i2, Z_n_i1, Z_n_i2, Z_n_e,
    m_k, Lambda_k_r, diff_Lambda_k_a2, R_1n_1, R_1n_2, R_2n_2,
    diff_R_1n_1, diff_R_1n_2, diff_R_2n_2,R_2n_1
)

# 设置路径
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

# 定义可视化 A 矩阵的函数
def visualize_A_matrix(A, title="Matrix Visualization"):
    rows, cols = np.nonzero(A)
    plt.figure(figsize=(6, 6))
    plt.scatter(cols, rows, color='blue', marker='o', s=100)
    plt.gca().invert_yaxis()
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))

    # 绘制分隔线用于可视化 A 矩阵的块结构
    N, M = 4, 4
    block_dividers = [N, N + M, N + 2 * M]
    for val in block_dividers:
        plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
        plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)

    plt.grid(True)
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()


N = 4
M = 4
K = 4

# 期望值设置
expected_b = np.array([
    0.0069, 0.0120, -0.0030, 0.0013, 
    0.1560, 0.0808, -0.0202, 0.0090, 
    0, -0.1460, 0.0732, -0.0002, 
    -0.4622, -0.2837, 0.1539, -0.0673
], dtype=np.complex128)

expected_A_path = "../value/A_values.csv"
df = pd.read_csv(expected_A_path, header=None)

# 转换为复数
def to_complex(val):
    try:
        return np.complex128(val)
    except ValueError:
        return np.nan + 1j * np.nan
    
df_complex = df.applymap(to_complex)
expected_A = df_complex.to_numpy()
expected_A[-4][-4] = np.complex128(-0.45178 + 1.0741j)

# 设置误差容限
tolerance = 1e-3
threshold = 0.01

# 几何配置和问题实例
a2=1.0
a1=0.5
h=1.001
d1=0.5
d2=0.25
r_coordinates = {'a1': 0.5, 'a2': 1.0}
z_coordinates = {'h': 1.001}
domain_params = [
    {'number_harmonics': N, 'height': 1, 'radial_width': 0.5, 'top_BC': None, 'bottom_BC': None, 'category': 'inner', 'di': 0.5},
    {'number_harmonics': M, 'height': 1, 'radial_width': 1.0, 'top_BC': None, 'bottom_BC': None, 'category': 'outer', 'di': 0.25},
    {'number_harmonics': K, 'height': 1, 'radial_width': 1.5, 'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
]
geometry = Geometry(r_coordinates, z_coordinates, domain_params)
problem = MEEMProblem(geometry)
engine = MEEMEngine([problem])

# 生成并验证 A 矩阵
generated_A = engine.assemble_A(problem)
visualize_A_matrix(generated_A, title="Generated A Matrix")

# 保存 A 矩阵到文件
np.savetxt("../value/A.txt", generated_A)

# 设置匹配阈值并检查匹配情况
threshold = 0.001
is_within_threshold = np.isclose(expected_A, generated_A, rtol=threshold)

# 保存 A 矩阵的匹配结果到文件
np.savetxt("../value/A_match.txt", is_within_threshold)

# 显示不匹配的索引和值并绘制不匹配位置
rows, cols = np.nonzero(~is_within_threshold)
# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(cols, rows, color='blue', marker='o', s=100) 
plt.gca().invert_yaxis()
plt.xticks(range(is_within_threshold.shape[1]))
plt.yticks(range(is_within_threshold.shape[0]))

# N = is_within_threshold.shape[1] // 3
# M = is_within_threshold.shape[0] // 3
cols = [4, 8, 12]

for val in cols:
    plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
    plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)

plt.grid(True)
plt.title('Non-Zero Entries of the Matrix Not Matching the Threshold')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()

# 生成并验证 b 向量
generated_b = engine.assemble_b(problem)
try:
    np.testing.assert_allclose(generated_b, expected_b, atol=tolerance, err_msg="b 向量与期望值不匹配")
    print("b 向量匹配成功。")
except AssertionError as e:
    print("b 向量与期望值不匹配。详细信息:")
    print(e)

# 保存 b 向量的匹配结果
is_within_threshold_b = np.isclose(expected_b, generated_b, atol=threshold)
np.savetxt("b_match.txt", is_within_threshold_b, fmt='%d')
print("b 向量匹配结果已保存至 b_match.txt")

# Solve the system A x = b
X = linalg.solve(generated_A, generated_b)

# Extract coefficients
C_1n_1s = X[:N]
C_1n_2s = X[N:N+M]
C_2n_2s = X[N+M:N+2*M]
C_2n_1s = np.zeros(N, dtype=complex)  # Assuming C_2n_1s are zeros
B_ks = X[N+2*M:]

# Define functions for the potentials
def phi_h_n_i1_func(n, r, z):
    return (C_1n_1s[n] * R_1n_1(n, r) + C_2n_1s[n] * R_2n_1(n)) * Z_n_i1(n, z)

def phi_h_m_i2_func(m, r, z):
    return (C_1n_2s[m] * R_1n_2(m, r) + C_2n_2s[m] * R_2n_2(m, r)) * Z_n_i2(m, z)

def phi_e_k_func(k, r, z):
    return B_ks[k] * Lambda_k_r(k, r) * Z_n_e(k, z)

# Create spatial grid
spatial_res = 50
r_vec = np.linspace(0, 2 * a2, spatial_res)
z_vec = np.linspace(-h, 0, spatial_res)
R, Z = np.meshgrid(r_vec, z_vec)

# Define regions
region1 = (R <= a1) & (Z < -d1)
region2 = (R > a1) & (R <= a2) & (Z < -d2)
regione = R > a2
region_body = ~region1 & ~region2 & ~regione  # The body of the cylinder

# Initialize potential arrays
phiH = np.zeros_like(R, dtype=complex)
phiP = np.zeros_like(R, dtype=complex)

# Compute the homogeneous potential in each region
# Region 1
for n in range(N):
    phiH[region1] += phi_h_n_i1_func(n, R[region1], Z[region1])

# Region 2
for m in range(M):
    phiH[region2] += phi_h_m_i2_func(m, R[region2], Z[region2])

# Exterior region
for k in range(K):
    phiH[regione] += phi_e_k_func(k, R[regione], Z[regione])

# Compute the particular potential in each region
phiP[region1] = phi_p_i1(R[region1], Z[region1])
phiP[region2] = phi_p_i2(R[region2], Z[region2])
phiP[regione] = 0  # No particular potential in the exterior region

# Compute the total potential
phi = phiH + phiP

# Plotting functions
def plot_potential(phi, R, Z, title):
    plt.figure(figsize=(12, 6))

    # Real part
    plt.subplot(1, 2, 1)
    contour_real = plt.contourf(R, Z, np.real(phi), levels=50, cmap='viridis')
    plt.colorbar(contour_real)
    plt.title(f'{title} - Real Part')
    plt.xlabel('Radial Distance (R)')
    plt.ylabel('Axial Distance (Z)')

    # Imaginary part
    plt.subplot(1, 2, 2)
    contour_imag = plt.contourf(R, Z, np.imag(phi), levels=50, cmap='viridis')
    plt.colorbar(contour_imag)
    plt.title(f'{title} - Imaginary Part')
    plt.xlabel('Radial Distance (R)')
    plt.ylabel('Axial Distance (Z)')

    plt.tight_layout()
    plt.show()

def plot_matching(phi1, phi2, phie, a1, a2, R, Z, title):
    idx_a1 = np.argmin(np.abs(R[0, :] - a1))
    idx_a2 = np.argmin(np.abs(R[0, :] - a2))

    Z_line = Z[:, 0]

    # Potential at r = a1
    phi1_a1 = phi1[:, idx_a1]
    phi2_a1 = phi2[:, idx_a1]

    # Potential at r = a2
    phi2_a2 = phi2[:, idx_a2]
    phie_a2 = phie[:, idx_a2]

    plt.figure(figsize=(8, 6))
    plt.plot(Z_line, np.abs(phi1_a1), 'r--', label=f'|{title}_1| at r = a1')
    plt.plot(Z_line, np.abs(phi2_a1), 'b-', label=f'|{title}_2| at r = a1')
    plt.plot(Z_line, np.abs(phi2_a2), 'g--', label=f'|{title}_2| at r = a2')
    plt.plot(Z_line, np.abs(phie_a2), 'k-', label=f'|{title}_e| at r = a2')
    plt.legend()
    plt.xlabel('Z')
    plt.ylabel(f'|{title}|')
    plt.title(f'{title} Matching at Interfaces')
    plt.show()

# Plot the potentials
plot_potential(phiH, R, Z, 'Homogeneous Potential')
plot_potential(phiP, R, Z, 'Particular Potential')
plot_potential(phi, R, Z, 'Total Potential')

# Extract potentials in different regions for matching
phi1 = np.where(region1, phi, np.nan)
phi2 = np.where(region2, phi, np.nan)
phie = np.where(regione, phi, np.nan)

# Plot matching at interfaces
plot_matching(phi1, phi2, phie, a1, a2, R, Z, 'Potential')

# Additional verification: Continuity at interfaces
def verify_continuity_at_interfaces():
    # Potential at r = a1 from region1 and region2
    idx_a1 = np.argmin(np.abs(R[0, :] - a1))
    phi_a1_region1 = phi1[:, idx_a1]
    phi_a1_region2 = phi2[:, idx_a1]

    # Compute the difference
    difference_a1 = np.abs(phi_a1_region1 - phi_a1_region2)

    # Potential at r = a2 from region2 and exterior region
    idx_a2 = np.argmin(np.abs(R[0, :] - a2))
    phi_a2_region2 = phi2[:, idx_a2]
    phi_a2_exterior = phie[:, idx_a2]

    # Compute the difference
    difference_a2 = np.abs(phi_a2_region2 - phi_a2_exterior)

    # Plot the differences
    plt.figure(figsize=(8, 6))
    plt.plot(Z[:, 0], difference_a1, label='Difference at r = a1')
    plt.plot(Z[:, 0], difference_a2, label='Difference at r = a2')
    plt.legend()
    plt.xlabel('Z')
    plt.ylabel('Potential Difference')
    plt.title('Continuity of Potential at Interfaces')
    plt.show()

verify_continuity_at_interfaces()