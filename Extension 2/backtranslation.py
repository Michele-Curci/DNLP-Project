import json
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='fr', target='en')
#perturbations = ["alteration", "expansion_impact", "expansion_noimpact", "intensifier", "omission", "spelling", "synonym", "word_order"]

perturbations = ["synonym", "spelling"]
for perturbation in perturbations:
    input_file = f"{perturbation}.jsonl"
    output_file = f"bt-{perturbation}.jsonl"

    updated_jsonl = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            try:
                translated_text = translator.translate(data['pert_fr'])
                print("Backtranslation: ", translated_text)
                data['bt_pert_fr'] = translated_text
            except Exception as e:
                print(f"Translation failed for: {data['pert_fr']} with error: {e}")
                data['bt_pert_fr'] = ""
            updated_jsonl.append(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in updated_jsonl:
          out_data = {
            "id": entry.get("id", ""),
            "en": entry.get("en", ""),
            "fr": entry.get("fr", ""),
            "pert_fr": entry.get("pert_fr", ""),
            "bt_pert_fr": entry.get("bt_pert_fr", "")}
          f.write(json.dumps(out_data, ensure_ascii=False) + '\n')