import os
import torch
import argparse
import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

#save_directory = "/home/concina/csv_translation/data/translation_model"
# open source translation model from Hugging Face hub
model_name = "facebook/m2m100_418M"

# translation function
def translate_text(text, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# batch translation
def translate_batch(texts, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


if __name__ == "__main__": 
    #define the script interface 
    parser = argparse.ArgumentParser(description='Script that translate the keyword column of the input CSV and save the output at the given location')
    parser.add_argument('--input_csv', help='path to input CSV file')
    parser.add_argument('--output_csv', help='path to output CSV file')
    parser.add_argument('--target_languages', nargs='+', help='List of languages to translate the keyword column')
    parser.add_argument('--resources_folder', help='path to folder where models are dowloaded and saved')
    parser.add_argument('--batch_size', help='Batch size for translation. (the higher the faster)')

    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv = args.output_csv
    target_languages = args.target_languages
    resources_folder = args.resources_folder
    batch_size = int(args.batch_size)

    # load models
    if not os.listdir(resources_folder):  
        print("Downloading and saving the model and tokenizer...")
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer.save_pretrained(resources_folder)
        model.save_pretrained(resources_folder)
    else:
        print("Loading model and tokenizer from the saved directory...")
        tokenizer = M2M100Tokenizer.from_pretrained(resources_folder)
        model = M2M100ForConditionalGeneration.from_pretrained(resources_folder)
    
    # we assume the source language of the keyword column of the CSV to be always EN
    source_language = "en" 
    # we translate only the Keywords column of the input CSV
    column_to_translate = "Keywords"

    # load input csv
    try:
        df = pd.read_csv(input_csv, encoding='Windows-1252')  # Default to UTF-8
        print("Sucefully loaded with Windows-1252 coding")
    except UnicodeDecodeError:
        print("Failed to read the file with UTF-8 encoding. Trying 'latin1'...")
        df = pd.read_csv(input_csv, encoding='latin1')  # Fallback to latin1

    # batch translation
    for lang in target_languages:
        print(f"Translating {column_to_translate} column to {lang}")
        translated_column = f"{column_to_translate}_{lang}"
        # Process in batches 
        translated_texts = []
        for i in range(0, len(df), batch_size):
            batch = df[column_to_translate].iloc[i:i + batch_size].tolist()
            try:
                translated_texts.extend(translate_batch(batch, source_language, lang))
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        if len(translated_texts) == len(df):        
            df[translated_column] = translated_texts  
        else:
            print(f"Translation failed for language: {lang}")
        

    # Save to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Translated CSV saved to {output_csv}")
       
    