import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Set display options
pd.set_option('display.max_columns', 100)
sns.set(style='whitegrid')

def main():
    # Load the dataset
    print('Loading dataset...')
    df = pd.read_csv('data/complaints.csv', low_memory=False)
    print(f'Dataset shape: {df.shape}')
    print(df.head())

    # 1. Initial Data Exploration
    print('\n--- Data Info ---')
    print(df.info())
    print('\n--- Missing Values ---')
    print(df.isnull().sum())

    # 2. Distribution of complaints across Products
    product_counts = df['Product'].value_counts()
    plt.figure(figsize=(10,5))
    sns.barplot(x=product_counts.index, y=product_counts.values)
    plt.title('Distribution of Complaints by Product')
    plt.ylabel('Number of Complaints')
    plt.xlabel('Product')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/product_distribution.png')
    plt.close()
    print('\nSaved product distribution plot to data/product_distribution.png')

    # 3. Narrative Length Analysis
    df['narrative_length'] = df['Consumer complaint narrative'].fillna('').apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10,5))
    sns.histplot(df['narrative_length'], bins=50, kde=True)
    plt.title('Distribution of Narrative Lengths (Word Count)')
    plt.xlabel('Word Count')
    plt.ylabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig('data/narrative_length_distribution.png')
    plt.close()
    print('\nSaved narrative length distribution plot to data/narrative_length_distribution.png')
    print('Very short narratives:', (df['narrative_length'] < 10).sum())
    print('Very long narratives:', (df['narrative_length'] > 500).sum())

    # 4. Complaints With and Without Narratives
    with_narrative = df['Consumer complaint narrative'].notnull().sum()
    without_narrative = df['Consumer complaint narrative'].isnull().sum()
    print(f'Complaints with narrative: {with_narrative}')
    print(f'Complaints without narrative: {without_narrative}')

    # 5. Filter Dataset for Project Requirements
    products_of_interest = [
        'Credit card',
        'Personal loan',
        'Buy Now, Pay Later (BNPL)',
        'Savings account',
        'Money transfers'
    ]
    filtered_df = df[
        df['Product'].isin(products_of_interest) &
        df['Consumer complaint narrative'].notnull() &
        (df['Consumer complaint narrative'].str.strip() != '')
    ].copy()
    print(f'Filtered dataset shape: {filtered_df.shape}')
    print(filtered_df['Product'].value_counts())

    # 6. Clean Text Narratives
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'i am writing to file a complaint', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_text)
    filtered_df = filtered_df[filtered_df['cleaned_narrative'].str.strip() != '']
    print('\nSample cleaned narratives:')
    print(filtered_df['cleaned_narrative'].head())

    # 7. Save Cleaned and Filtered Dataset
    output_path = 'data/filtered_complaints.csv'
    filtered_df.to_csv(output_path, index=False)
    print(f'Filtered and cleaned dataset saved to {output_path}')

if __name__ == '__main__':
    main() 