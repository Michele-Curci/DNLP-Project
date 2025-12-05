from transformers import AutoTokenizer, AutoModelForCausalLM
from QG_prompt import prompts
import torch
import json
import argparse
import os

INPUT_FILE = f"Extension 1/entity_dataset.jsonl"
OUTPUT_FILE = f"Extension 1/QG_dataset.jsonl"
PROMPT = "entities"
TOKEN = ""

own_cache_dir = "Extension 1/QG/.cache"
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

model_id = "google/gemma-2-9b-it"

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=TOKEN, cache_dir=own_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,  # Using bfloat16 for lower memory usage
        device_map="auto",
        token=TOKEN,
        cache_dir=own_cache_dir
    )

    # =========================================== Load Dataset ===========================================    
    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
        for i, line in enumerate(f_in):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {i+1} due to JSON error.")
                continue
            sentence = data.get('en', None)
            if sentence:
                prompt_template = prompts.get(PROMPT, prompts["vanilla"])

                # Default to 'vanilla' prompt format if semantic_roles or atomic_facts are missing/empty
                if PROMPT == "entities":
                    entities = data.get('ner_entities', [])
                    if entities:
                        ner_entities_str = json.dumps(entities, ensure_ascii=False)
                        prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{ner_entities}}", ner_entities_str)
                    else:
                        #prompt = prompt_template.replace("{{sentence}}", sentence)
                        prompt = prompts["vanilla"].replace("{{sentence}}", sentence)

                else:  # Default case (e.g., vanilla)
                    prompt = prompt_template.replace("{{sentence}}", sentence)
                
                input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

                # Use inference mode to reduce memory usage
                with torch.no_grad():
                    outputs = model.generate(
                        **input_ids,
                        max_new_tokens=1024,
                        num_beams=1,  # Reduce to a single beam if memory is an issue
                    )
                

                generated_questions = tokenizer.decode(outputs[0], skip_special_tokens=True)

                answer_start = "Questions: "
                if answer_start in generated_questions:
                    generation = generated_questions.split(answer_start)[-1].strip()
                else:
                    generation = generated_questions

                try:
                    data['questions'] = eval(generation)
                except:
                    data['questions'] = [generation]

                print(f"{prompt}")
                print(f"> {generation}")
                print("\n======================================================\n")

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                # Clear CUDA cache after processing each input
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()