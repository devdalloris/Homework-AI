import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd

data = {
    'Optimization': ['DM = 1.0', 'DM = 1.0 + TensorRT', 'DM = 0.5', 'DM = 0.5 + TensorRT'],
    'OS = 8': [2.51, None, 1.19, None],
    'OS = 16': [0.85, 0.57, 0.43, 0.3],
    'OS = 32': [0.77, 0.31, 0.33, 0.17]
}

df = pd.DataFrame(data)
print(df)

# Melt the DataFrame for easier plotting with seaborn
df_melted = df.melt(id_vars='Optimization', var_name='OS', value_name='Inference Time (s)')
# Convert 'OS' to a numerical type for proper sorting if needed
df_melted['OS_num'] = df_melted['OS'].str.replace('OS = ', '').astype(int)
df_melted = df_melted.sort_values(by=['Optimization', 'OS_num'])

plt.figure(figsize=(12, 7))
sns.barplot(x='OS', y='Inference Time (s)', hue='Optimization', data=df_melted, palette='viridis')

plt.title('Inference Time Performance (s) by Optimization and OS')
plt.xlabel('Operating System (OS) / Output Size')
plt.ylabel('Inference Time (s)')
plt.legend(title='Optimization Strategy')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()