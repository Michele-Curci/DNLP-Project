## Translation from English to French using deep_translator

import json
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='en', target='fr')

input_file = "Extension 2/QG_results/output_ner.jsonl"
output_file = "translation.jsonl"

updated_jsonl = []
# Read the input JSONL file and translate each English text to French
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        try:
            translated_text = translator.translate(data['en'])
            print("Translation: ", translated_text)
            data['fr'] = translated_text
        except Exception as e:
            print(f"Translation failed for: {data['en']} with error: {e}")
            data['fr'] = ""
        updated_jsonl.append(data)

# Write the updated data with translations to the output JSONL file
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in updated_jsonl:
        out_data = {
          "id": entry.get("id", ""),
          "en": entry.get("en", ""),
          "fr": entry.get("fr", "")}
        f.write(json.dumps(out_data, ensure_ascii=False) + '\n')