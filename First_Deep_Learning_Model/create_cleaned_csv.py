#!/usr/bin/env python3
"""
Create cleaned CSV file from the original Excel file
"""
import pandas as pd
import numpy as np
import csv
from io import StringIO
def create_temu_reviews_cleaned_csv():
    """Create cleaned CSV from the original Excel file"""
    print("Creating temu_reviews_cleaned.csv from the original Excel file...")
    
    try:
        # Read the Excel file
        df_excel = pd.read_excel('data/temu_reviews_1.csv.xlsx')
        print(f"Successfully loaded Excel file with shape: {df_excel.shape}")
        
        # Parse the data - the Excel file has comma-separated data in a single column
        data_rows = []
        columns = ['UserId', 'UserName', 'UserCountry', 'ReviewCount', 'ReviewRating', 
                  'ReviewTitle', 'ReviewText', 'ReviewDate', 'ReviewExperienceDate', 
                  'ReplyText', 'ReplyDate']
        
        print("Parsing comma-separated data...")
        for idx, row in df_excel.iterrows():
            row_data = str(row.iloc[0])
            # Use CSV reader to properly parse the comma-separated values
            reader = csv.reader([row_data])
            parsed_row = next(reader)
            if len(parsed_row) >= len(columns):
                data_rows.append(parsed_row[:len(columns)])
        
        # Create proper DataFrame
        df = pd.DataFrame(data_rows, columns=columns)
        
        # Convert data types
        df['ReviewRating'] = pd.to_numeric(df['ReviewRating'], errors='coerce')
        df['ReviewCount'] = pd.to_numeric(df['ReviewCount'], errors='coerce')
        
        print(f"Processed data shape: {df.shape}")
        print(f"Review Rating distribution:")
        print(df['ReviewRating'].value_counts().sort_index())
        
        # Save the cleaned data
        output_path = 'data/temu_reviews_cleaned.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Successfully created: {output_path}")
        print(f"File contains {len(df)} reviews with {len(df.columns)} columns")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        print(df.head(3).to_string())
        print(df.isna().sum())
        
        return df
        
    except Exception as e:
        print(f"❌ Error creating cleaned CSV: {e}")
        return None
if __name__ == "__main__":
    df = create_temu_reviews_cleaned_csv()