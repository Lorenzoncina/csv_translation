import pandas as pd

# Hardcoded file paths
input_path = "/home/concina/csv_translation/data/tm_field_terms.csv"  # Replace with the path to your input CSV
output_path = "/home/concina/csv_translation/data/output_100.csv"  # Replace with the desired output CSV path

try:
    # Read the CSV file
    df = pd.read_csv(input_path, encoding='Windows-1252')
    print(f"Successfully read the input CSV from {input_path}")

    # Select the first 100 rows
    first_100_rows = df.head(100)

    # Save the new CSV file
    first_100_rows.to_csv(output_path, index=False)
    print(f"Saved the first 100 rows to {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
