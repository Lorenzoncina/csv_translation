import pandas as pd
import chardet
import os 
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def translate_text(text, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def translate_batch(texts, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

save_directory = "/home/concina/csv_translation/data/translation_model"
input_csv = '/home/concina/csv_translation/data/output_100.csv'
df = pd.read_csv(input_csv, encoding='Windows-1252') 

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

# translate and print only the first row
input = df['Keywords'].iloc[0]
print(input)

output = translate_text(input, 'en', 'fr')
print(output)

column_to_translate = "Keywords"
lang = "fr"
source_language = "en"
target_languages = ["it"]
print("save an output test csv")
#translated_column = f"{column_to_translate}_{lang}"
#df[translated_column] = df[column_to_translate].apply(lambda x: translate_text(x, source_language, lang))
for lang in target_languages:
    print(f"Translating {column_to_translate} column to {lang}")
    translated_column = f"{column_to_translate}_{lang}"
    # Process in batches of 32
    batch_size = 128
    translated_texts = []
    for i in range(0, len(df), batch_size):
        batch = df[column_to_translate].iloc[i:i + batch_size].tolist()
        translated_texts.extend(translate_batch(batch, source_language, lang))
    df[translated_column] = translated_texts

output_path = "/home/concina/csv_translation/data/output_100.csv"
df.to_csv(output_path, index=False)
print(f"Translated CSV saved to {output_csv}")