import json
from collections import Counter

# percorso del file
file_path = "word_order_entity_answers.jsonl"  # cambia con il nome del tuo file

entity_counter = Counter()

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        for ent in data.get("ner_entities", []):
            entity_type = ent.get("type")
            if entity_type:
                entity_counter[entity_type] += 1

# stampa il risultato
for entity_type, count in entity_counter.most_common():
    print(f"{entity_type}: {count}")