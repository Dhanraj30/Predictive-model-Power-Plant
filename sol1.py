import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset into a DataFrame
df = pd.read_csv("train.csv")  # Replace "your_dataset.csv" with the actual path to your dataset

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Rest of your script...


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
