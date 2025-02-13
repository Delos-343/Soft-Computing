import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np

# LaTeX code to render
latex_code = r"""
\begin{latex}[h!]
    \text{insert content here}
\end{latex}
"""

# Create a figure to render LaTeX
fig = plt.figure(figsize=(10, 6))
plt.text(0.5, 0.5, f"${latex_code}$", fontsize=14, ha='center', va='center')
plt.axis('off')

# Save the figure to a PNG image
output_path = "/mnt/data/latex_table.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

output_path
