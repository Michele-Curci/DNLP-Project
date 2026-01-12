import json
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np

ORIGINAL_FILE = "Datasets/Extension 1/QA_results/source_answers.jsonl"
PERTURBED_FILE = "Datasets/Extension 1/QA_results/source-bt-word_order_answers.jsonl"
OUTPUT_FILE = "word_order_errors.jsonl"

# 1. SEMANTIC MODEL
model_sem = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine_sim(a, b):
    if not a or not b: # Handle empty strings for cosine similarity
        return 0.0
    emb = model_sem.encode([a, b])
    v1, v2 = emb[0], emb[1]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 2. FORMAL + SEMANTIC CLASSIFIER
def classify_entity_error(o_text, t_text):
    o = o_text.lower().strip() if o_text else ""
    t = t_text.lower().strip() if t_text else ""

    # Similarity metrics
    fuzzy_score = fuzz.partial_ratio(o, t)
    tok_score = fuzz.token_sort_ratio(o, t)
    coss = cosine_sim(o, t)

    # FORMAL CHECKS
    if fuzzy_score > 90 and tok_score > 90 and coss > 0.90:
        return "entity-match"
    if fuzzy_score > 70 or tok_score > 70:
        return "partial-match"

    # SEMANTIC ERROR CHECKS
    if coss > 0.70 and fuzzy_score < 50:
        if len(t.split()) < len(o.split()):
            return "semantic-generalization"
        if len(t.split()) > len(o.split()):
            return "semantic-specification"
        return "semantic-shift"
    if 0.50 <= coss <= 0.70:
        return "semantic-substitution"

    # DEFAULT
    return "wrong-entity"

# 3. ALIGNMENT + CLASSIFICATION
def align_and_classify(original_entities, translated_entities):
    results = []

    # Special Case 1: Translated entity is empty and ALL original entities are also empty
    if (translated_entities == [] and original_entities==[]):
            results.append({
                "original_text": "",
                "translated_text": "",
                "error_type": "no-entity",
                "metrics": {
                    "fuzzy_score": None,
                    "token_sort_ratio": None,
                    "cosine_similarity": None
                }
            })

    # Special Case 2: Translated entity is empty but original entities no (missing)
    elif (translated_entities == [] and original_entities!=[]):
            results.append({
              "original_text": [o_ent.get("text", "") for o_ent in original_entities if o_ent.get("text", "") != ""],
              "translated_text": "",
              "error_type": "missing-entity",
              "metrics": {
                "fuzzy_score": None,
                "token_sort_ratio": None,
                "cosine_similarity": None
              }
            })

    # Special Case 3: Translated entity is not empty but all original entities are empty (hallucination)
    elif (translated_entities != [] and original_entities==[]):
            results.append({
              "original_text": "",
              "translated_text": [t_ent.get("text", "") for t_ent in translated_entities if t_ent.get("text", "") != ""],
              "error_type": "hallucinated-entity",
              "metrics": {
                "fuzzy_score": None,
                "token_sort_ratio": None,
                "cosine_similarity": None
              }
            })
    else:
    # cycle on TARGET entities
      for t_ent in translated_entities:
          t_text = t_ent.get("text", "")

          # If not a special case, proceed with comparison against original entities
          ent_errors = [] # Initialize for the current t_ent
          for o_ent in original_entities:
              o_text = o_ent.get("text", "")

              error_type = classify_entity_error(o_text, t_text)

              cosine_sim_val = cosine_sim(o_text, t_text) if o_text and t_text else None
              if cosine_sim_val is not None:
                  cosine_sim_val = float(cosine_sim_val)

              ent_errors.append({
                "original_text": o_text,
                "translated_text": t_text,
                "error_type": error_type,
                "metrics": {
                  "fuzzy_score": fuzz.partial_ratio(o_text.lower(), t_text.lower()) if o_text and t_text else None,
                  "token_sort_ratio": fuzz.token_sort_ratio(o_text.lower(), t_text.lower()) if o_text and t_text else None,
                  "cosine_similarity": cosine_sim_val
                }
              })

          all_errors=ent_errors

          # Process `all_errors` only if it's not empty
          if all_errors:
              # If all remaining errors are 'wrong-entity'
              if all(e["error_type"] == "wrong-entity" for e in all_errors):
                  results.append({
                    "translated_text": t_text,
                    "error_type": "wrong-entity",
                  })
              else:
                  # Otherwise, extend results with all non-'wrong-entity' errors
                  results.extend([e for e in all_errors if e["error_type"] != "wrong-entity"])

    return results

# 4. LOAD FILE JSONL
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            data.append(row)
    return data

# 5. MAIN FUNCTION
def compare_files(original_path, perturbed_path, output_path):
    print("Loading files...")
    original_data = load_jsonl(original_path)
    perturbed_data = load_jsonl(perturbed_path)

    output_lines = []

    for o_item, t_item in zip(original_data, perturbed_data):
        _id = o_item.get("id", "")

        original_entities = o_item.get("ner_entities", [])
        translated_entities = t_item.get("ner_entities", [])

        alignment = align_and_classify(original_entities, translated_entities)

        result = {
            "id": _id,
            "original_answers": o_item.get("answers", []),
            "perturbed_answers": t_item.get("answers", []),
            "alignment": alignment
        }

        output_lines.append(result)

    print(f"Writing output to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out:
        for row in output_lines:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\u2705 Done! File saved:", output_path)

if __name__ == "__main__":
    compare_files(ORIGINAL_FILE, PERTURBED_FILE, OUTPUT_FILE)