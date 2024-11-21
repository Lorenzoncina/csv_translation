import os
import argparse
import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


# required resources lodade from HuggingFace
save_directory = "/home/concina/csv_translation/data/translation_model"
#specifi the open source translation model to use from hugging face
model_name = "facebook/m2m100_418M"

#download or load models
if not os.listdir(save_directory):  
    print("Downloading and saving the model and tokenizer...")
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
else:
    print("Loading model and tokenizer from the saved directory...")
    tokenizer = M2M100Tokenizer.from_pretrained(save_directory)
    model = M2M100ForConditionalGeneration.from_pretrained(save_directory)


# translation function
def translate_text(text, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


if __name__ == "__main__": 
    #define the script interface 
    parser = argparse.ArgumentParser(description='Script that translate the keyword column of the input CSV and save the output at the given location')
    parser.add_argument('--input_csv', help='path to input CSV file')
    parser.add_argument('--output_csv', help='path to output CSV file')
    parser.add_argument('--target_languages', nargs='+', help='List of languages to translate the keyword column')

    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv = args.output_csv
    target_languages = args.target_languages
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

    #run translation for each target language
    for lang in target_languages:
        print(f"Translating {column_to_translate} columns on {lang}")
        translated_column = f"{column_to_translate}_{lang}"
        df[translated_column] = df[column_to_translate].apply(lambda x: translate_text(x, source_language, lang))
       

    # Save to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Translated CSV saved to {output_csv}")
       
    