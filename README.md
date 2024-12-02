Clone and enter the repository
```
git clone https://github.com/Lorenzoncina/csv_translation.git
cd csv_translation
```

Create a new python virtual enviroment

```
python3 -m venv venv
```

Install all dependencies with requirements.txt
```
pip install requirements.txt
```

Run translate_keywords_helsinky.py script with appropriate arguments
```
#to learn about the arguments
python code/translate_keywords_helsinky.py --help

#for example
python code/translate_keywords_helsinky.py --input_csv data/tm_field_terms.csv --output_csv data/translations/out_es_hensinky.csv --target_language es  --resources_folder data/helsinky_models/
```
