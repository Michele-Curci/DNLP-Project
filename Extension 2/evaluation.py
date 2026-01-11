#ASKQE EVALUATION
#SBERT
import json
import nltk
import argparse
import csv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os # Import os for file existence check
from collections import Counter
import string
import re
import sacrebleu
from typing import List, Union


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


nltk.download("punkt")
output_file='sbert_score.jsonl'

#perturbation = ["synonym", "word_order", "spelling", "expansion_noimpact",
                 #"intensifier", "expansion_impact", "omission", "alteration"]

perturbation = "impact_expansion"
language='fr'
pipeline='ner'

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Initialize total_cosine_similarity and num_comparisons outside the file writing logic
total_cosine_similarity = 0
num_comparisons = 0

predicted_file = "word_order_answers.jsonl"
reference_file = "source_answers.jsonl"

try:
    with open(predicted_file, "r", encoding="utf-8") as pred_file, open(reference_file, "r", encoding="utf-8") as ref_file:
        for pred_line, ref_line in zip(pred_file, ref_file):
            try:
                pred_data = json.loads(pred_line)
                ref_data = json.loads(ref_line)

                predicted_answers = pred_data.get("answers", [])
                reference_answers = ref_data.get("answers", [])

                if isinstance(predicted_answers, str):
                    try:
                        predicted_answers = json.loads(predicted_answers)
                    except json.JSONDecodeError:
                        continue

                if isinstance(reference_answers, str):
                    try:
                        reference_answers = json.loads(reference_answers)
                    except json.JSONDecodeError:
                        continue

                if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list):
                    continue
                if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers):
                    continue
                for pred, ref in zip(predicted_answers, reference_answers):
                    if not isinstance(pred, str) or not isinstance(ref, str):
                        continue
                    if pred.strip() == "" or ref.strip() == "":
                        continue

                    encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
                    encoded_ref = tokenizer(ref, padding=True, truncation=True, return_tensors='pt')

                    with torch.no_grad():
                        pred_output = model(**encoded_pred)
                        ref_output = model(**encoded_ref)

                    pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
                    pred_embeds = F.normalize(pred_embed, p=2, dim=1)

                    ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
                    ref_embeds = F.normalize(ref_embed, p=2, dim=1)

                    cos_sim = F.cosine_similarity(pred_embeds, ref_embeds, dim=1).mean().item()
                    total_cosine_similarity += cos_sim
                    num_comparisons += 1

            except json.JSONDecodeError as e:
                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                continue

except FileNotFoundError as e:
    print(f"File not found: {e}")

# Now, after all calculations are done, write to the output file.
if num_comparisons > 0:
    avg_cosine_similarity = total_cosine_similarity / num_comparisons

    print("-" * 80)
    print("Average Scores:")
    print(f"Num comparisons: {num_comparisons}")
    print(f"Cosine Similarity: {avg_cosine_similarity:.3f}")
    print("=" * 80)

    # Check if file exists and is not empty to decide whether to write header
    file_exists = os.path.exists(output_file)
    file_is_empty = not file_exists or os.path.getsize(output_file) == 0

    with open(output_file, mode="a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        if file_is_empty:
            csv_writer.writerow(["language", "perturbation", "pipeline", "cosine_similarity", "num_comparison"])

        csv_writer.writerow([language, perturbation, pipeline, avg_cosine_similarity, num_comparisons])

else:
    print("No valid comparisons found in the JSONL files.")

def split_list(mylist: List, chunk_size: Union[int]):
    return [mylist[offs:offs + chunk_size] for offs in range(0, len(mylist), chunk_size)]


def normalize_answer(s: Union[str]):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_articles(remove_punc(s)))



def f1_score(prediction: Union[str], ground_truth: Union[str], normalize=False):
    """Compute word-level F1 score"""
    if normalize:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
    else:
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: Union[str], ground_truth: Union[str], normalize=False):
    """Compute word-level EM score"""
    if normalize:
        return normalize_answer(prediction) == normalize_answer(ground_truth)
    return prediction == ground_truth


def chrf_score(prediction: Union[str], golden_truth: Union[str], normalize=False):
    """Compute sentence-level chrf score"""
    if normalize:
        return sacrebleu.sentence_chrf(normalize_answer(prediction),
                                       [normalize_answer(golden_truth)]).score
    else:
        return sacrebleu.sentence_chrf(prediction,
                                       [golden_truth]).score


def bleu_score(prediction: Union[str], golden_truth: Union[str], normalize=False):
    """Compute sentence-level bleu score"""
    if normalize:
        return sacrebleu.sentence_bleu(normalize_answer(prediction),
                                       [normalize_answer(golden_truth)]).score
    else:
        return sacrebleu.sentence_bleu(prediction,
                                       [golden_truth]).score


def compare_answers(prediction: Union[str], golden_truth: Union[str], normalize=True):
    return (
        f1_score(prediction, golden_truth, normalize),
        exact_match_score(prediction, golden_truth, normalize),
        chrf_score(prediction, golden_truth, normalize),
        bleu_score(prediction, golden_truth, normalize)
    )

def safe_mean(values):
    return sum(values) / len(values) if values else 0

languages = "fr"
pipelines = "ner"
#perturbations = ["alteration", "expansion_impact", "expansion_noimpact", "intensifier", "omission", "spelling", "synonym", "word_order"]
perturbations = "alteration"

predicted_file = "word_order_answers.jsonl"
reference_file = "source_answers.jsonl"
output_file='string_score.jsonl'

results_list = []
global_f1 = []
global_em = []
global_chrf = []
global_bleu = []
try:
    with open(predicted_file, "r", encoding="utf-8") as pred_file, open(reference_file, "r", encoding="utf-8") as ref_file:
        for pred_line, ref_line in zip(pred_file, ref_file):
            try:
                pred_data = json.loads(pred_line)
                ref_data = json.loads(ref_line)

                predicted_answers = pred_data.get("answers", [])
                reference_answers = ref_data.get("answers", [])

                if isinstance(predicted_answers, str):
                    try:
                        predicted_answers = json.loads(predicted_answers)
                    except json.JSONDecodeError:
                        continue

                if isinstance(reference_answers, str):
                    try:
                        reference_answers = json.loads(reference_answers)
                    except json.JSONDecodeError:
                        continue

                if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list):
                    continue
                if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers):
                    continue

                row_scores = []
                for pred, ref in zip(predicted_answers, reference_answers):
                    f1, EM, chrf, bleu = compare_answers(pred, ref)
                    row_scores.append({
                        "f1": f1,
                        "em": EM,
                        "chrf": chrf,
                        "bleu": bleu
                    })


                # Save per-row result
                mean_f1 = safe_mean([s["f1"] for s in row_scores])
                mean_em = safe_mean([s["em"] for s in row_scores])
                mean_chrf = safe_mean([s["chrf"] for s in row_scores])
                mean_bleu = safe_mean([s["bleu"] for s in row_scores])


                row_data = {
                "id": pred_data.get("id", "unknown"),
                "en": pred_data.get("backtranslatd", "unknown"),
                "mean_f1": mean_f1,
                "mean_em": mean_em,
                "mean_chrf": mean_chrf,
                "mean_bleu": mean_bleu
}
                results_list.append(row_data)

                global_f1.append(mean_f1)
                global_em.append(mean_em)
                global_chrf.append(mean_chrf)
                global_bleu.append(mean_bleu)

            except json.JSONDecodeError as e:
                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                continue

except FileNotFoundError as e:
          print(f"File not found: {e}")

with open(output_file, "w", encoding="utf-8") as jsonl_file:
                for row in results_list:
                    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

final_scores = {
    "mean_f1": safe_mean(global_f1),
    "mean_em": safe_mean(global_em),
    "mean_chrf": safe_mean(global_chrf),
    "mean_bleu": safe_mean(global_bleu),
    "Sbert": avg_cosine_similarity
}

# Salvo anche le metriche aggregate in un file separato
with open("string_score_summary.json", "w", encoding="utf-8") as summary_file:
    json.dump(final_scores, summary_file, ensure_ascii=False, indent=4)