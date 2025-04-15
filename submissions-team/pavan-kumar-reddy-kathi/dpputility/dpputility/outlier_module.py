import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

def analyze_and_cap_outliers_iqr(df, column):
    """
    Analyze, visualize, and cap outliers from a dataframe column using the IQR method.

    Parameters:
    df (pandas.DataFrame): Input dataframe
    column (str): Name of the column to cap outliers

    Returns:
    pandas.DataFrame: Dataframe with outliers capped
    """
    # Create a copy of the dataframe
    df_clean = df.copy()

    # Calculate Q1, Q3, and IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f'lower_bound ={lower_bound}')
    print(f'upper_bound ={upper_bound}')
    # Print statistics before removal
    print(f"\n=== Analysis for {column} ===")
    print("\nBefore outlier removal:")
    print(f"Count: {df[column].count()}")
    print(f"Mean: {df[column].mean():.2f}")
    print(f"Median: {df[column].median():.2f}")
    print(f"Std: {df[column].std():.2f}")
    print(f"Min: {df[column].min():.2f}")
    print(f"Max: {df[column].max():.2f}")
    print(f'Skew: {skew(df[column])}')

    # Identify outliers
    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]
    print(f"\nNumber of outliers detected: {len(outliers)}")
    print(f"Outliers percentage: {(len(outliers)/len(df))*100:.2f}%")

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Before removal boxplot
    plt.subplot(131)
    sns.boxplot(y=df[column])
    plt.title('Before Removal\nBoxplot')

    # Before removal distribution
    plt.subplot(132)
    sns.histplot(df[column], kde=True)
    plt.axvline(lower_bound, color='r', linestyle='--', label='Lower bound')
    plt.axvline(upper_bound, color='r', linestyle='--', label='Upper bound')
    plt.title('Before Removal\nDistribution')
    plt.legend()

    # Cap the outliers at the bounds
    df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
    df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound

    # After removal distribution
    plt.subplot(133)
    sns.histplot(df_clean[column], kde=True)
    plt.title('After Removal\nDistribution')

    plt.tight_layout()
    plt.show()

    # Print statistics after removal
    print("\nAfter outlier removal:")
    print(f"Count: {df_clean[column].count()}")
    print(f"Mean: {df_clean[column].mean():.2f}")
    print(f"Median: {df_clean[column].median():.2f}")
    print(f"Std: {df_clean[column].std():.2f}")
    print(f"Min: {df_clean[column].min():.2f}")
    print(f"Max: {df_clean[column].max():.2f}")
    print(f'Skew: {skew(df_clean[column])}')
    return df_clean