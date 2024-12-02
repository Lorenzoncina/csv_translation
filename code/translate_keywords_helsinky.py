import os
import argparse
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

def load_helsinki_model(source_lang, target_lang, resources_folder):
    """Load the Helsinki-NLP translation model and tokenizer."""
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    folder_name = os.path.join(resources_folder, model_name)
    os.makedirs(folder_name, exist_ok=True)
    # Download model and tokenizer if not already available locally
    #if not os.listdir(resources_folder):
    print(f"Downloading and saving the model and tokenizer for {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer.save_pretrained(folder_name)
    model.save_pretrained(folder_name)
    #else:
    #    print(f"Loading model and tokenizer for {model_name} from the saved directory...")
    #    tokenizer = MarianTokenizer.from_pretrained(resources_folder)
    #    model = MarianMTModel.from_pretrained(resources_folder)

    return tokenizer, model

def translate_batch(texts, tokenizer, model):
    """Batch translation function."""
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**encoded)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    # Define the script interface
    parser = argparse.ArgumentParser(description='Script to translate the keyword column of the input CSV and save the output.')
    parser.add_argument('--input_csv', help='Path to input CSV file')
    parser.add_argument('--output_csv', help='Path to output CSV file')
    parser.add_argument('--target_languages', nargs='+', help='List of target languages (e.g., fr, de, es)')
    parser.add_argument('--resources_folder', help='Main folder where to save the model')
    parser.add_argument('--batch_size', help='Batch size for translation', default=32)

    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv = args.output_csv
    target_languages = args.target_languages
    resources_folder = args.resources_folder
    batch_size = int(args.batch_size)

    # Load input CSV
    try:
        df = pd.read_csv(input_csv, encoding='Windows-1252')
        print("Successfully loaded input CSV with Windows-1252 encoding.")
    except UnicodeDecodeError:
        print("Failed to read the file with 'Windows-1252' encoding. Trying 'latin1'...")
        df = pd.read_csv(input_csv, encoding='latin1')

    # We assume the source language of the keyword column is always English (en)
    source_language = "en"
    column_to_translate = "Keywords"

    # Translate the keywords column for each target language
    for lang in target_languages:
        print(f"Translating {column_to_translate} column to {lang}")
        translated_column = f"{column_to_translate}_{lang}"

        # Load the appropriate model and tokenizer for the target language
        tokenizer, model = load_helsinki_model(source_language, lang, resources_folder)

        # Process in batches
        translated_texts = []
        for i in range(0, len(df), batch_size):
            batch = df[column_to_translate].iloc[i:i + batch_size].tolist()
            try:
                translated_texts.extend(translate_batch(batch, tokenizer, model))
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        # Add the translated column to the dataframe
        if len(translated_texts) == len(df):
            df[translated_column] = translated_texts
        else:
            print(f"Translation failed for language: {lang}")

    # Save the dataframe to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Translated CSV saved to {output_csv}")
