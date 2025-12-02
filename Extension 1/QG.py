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

own_cache_dir = "/fs/clip-scratch/dayeonki/.cache"
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

model_id = "google/gemma-2-9b-it"

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Using bfloat16 for lower memory usage
        device_map="auto",
        use_auth_token=TOKEN
    )

    # =========================================== Load Dataset ===========================================    
    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            sentence = data.get('en', None)
            if sentence:
                prompt_template = PROMPT

                # Default to 'vanilla' prompt format if semantic_roles or atomic_facts are missing/empty
                if PROMPT == "semantic":
                    semantic = data.get('semantic_roles', None)
                    if semantic:
                        prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{semantic_roles}}", semantic)
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                elif PROMPT == "atomic":
                    atomics = data.get('atomic_facts', None)
                    if atomics:
                        prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{atomic_facts}}", str(atomics))
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                elif PROMPT == "entities":
                    entities = data.get('ner_entities', None)
                    if entities:
                        prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{ner_entities}}", str(entities))
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                else:  # Default case (e.g., vanilla)
                    prompt = prompt_template.replace("{{sentence}}", sentence)
                
                input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

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
                    generation = generation.split("<")[0].strip()
                else:
                    generation = generated_questions

                print(f"{prompt}")
                print(f"> {generation}")
                print("\n======================================================\n")

                data['questions'] = str(generation)
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                # Clear CUDA cache after processing each input
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()