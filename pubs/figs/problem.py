import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['Execution Time', 'Memory Usage']
bem_values = [162, 1000]
openflash_values = [1, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1: Runtime
bars1 = ax1.bar(['Standard BEM', 'OpenFLASH'], [162, 1], color=['#808080', '#228B22'])
ax1.set_title('Runtime Comparison', fontsize=16, fontweight='bold')
ax1.set_ylabel('Relative Scale', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add labels for Runtime
ax1.text(0, 162 + 1, 'Minutes', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax1.text(1, 1 + 5, 'Seconds', ha='center', va='bottom', fontsize=14, color='#228B22', fontweight='bold')
ax1.text(0.5, 80, '162x Faster', ha='center', va='center', fontsize=16, color='white', fontweight='bold', 
         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

# Subplot 2: Memory
bars2 = ax2.bar(['Standard BEM', 'OpenFLASH'], [1000, 1], color=['#808080', '#228B22'])
ax2.set_title('Memory Usage Comparison', fontsize=16, fontweight='bold')
ax2.set_ylabel('Relative Scale', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add labels for Memory
ax2.text(0, 1000 + 1, 'Server', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax2.text(1, 1 + 20, 'Laptop', ha='center', va='bottom', fontsize=14, color='#228B22', fontweight='bold')
ax2.text(0.5, 500, '1,000x Less Memory', ha='center', va='center', fontsize=16, color='white', fontweight='bold', 
         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

plt.tight_layout()
plt.savefig('performance_benchmarking.png')