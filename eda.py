import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

def generate_eda():
    print("Fetching dataset for EDA...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    print("\n--- Summary Statistics ---")
    print(df.describe())

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_correlation.png")
    print("\nSaved correlation heatmap as 'eda_correlation.png'")

    # Target Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['MedHouseVal'], kde=True, bins=30, color='teal')
    plt.title("Distribution of Median House Values ($100k)")
    plt.xlabel("Median House Value")
    plt.tight_layout()
    plt.savefig("eda_target_dist.png")
    print("Saved target distribution plot as 'eda_target_dist.png'")

if __name__ == "__main__":
    generate_eda()