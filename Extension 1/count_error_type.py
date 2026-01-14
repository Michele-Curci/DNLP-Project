import json

INPUT_FILE = "Datasets/Extension 1/error_classification_results/word_order_errors.jsonl"  # change to other perturbation files as needed

def count_error_types(input_path):
    error_counts = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            alignment = row.get("alignment", [])

            for err in alignment:
                etype = err.get("error_type", "unknown")
                error_counts[etype] = error_counts.get(etype, 0) + 1

    return error_counts


counts = count_error_types(INPUT_FILE)

print("\n=== ERROR TYPE SUMMARY ===")
for etype, count in sorted(counts.items()):
    print(f"{etype}: {count}")

with open('error_results.jsonl', "w", encoding="utf-8") as out:
        for etype, count in sorted(counts.items()):
            out.write(json.dumps(f"{etype}: {count}", ensure_ascii=False) + "\n")