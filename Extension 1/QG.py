from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from QG_prompt import prompts
import torch
import json
import argparse
import os

INPUT_FILE = f"Extension 1/entity_dataset.jsonl"
OUTPUT_FILE = f"Extension 1/QG_dataset.jsonl"
VANILLA_FILE = "vanilla_gemma-9b.jsonl"
PROMPT = "entities"
TOKEN = ""

own_cache_dir = "Extension 1/QG/.cache"
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

model_id = "google/gemma-2-9b-it"

def main():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        # Use bfloat16 for computations as it's the native format for Gemma
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=TOKEN, cache_dir=own_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,  # Using bfloat16 for lower memory usage
        device_map="auto",
        token=TOKEN,
        cache_dir=own_cache_dir
    )

    # Load the vanilla data into a dictionary for fast lookup by 'id'
    vanilla_questions = {}
    if os.path.exists(VANILLA_FILE):
        try:
            with open(VANILLA_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # Store only the questions associated with the unique ID
                    vanilla_questions[data.get('id')] = data.get('questions', [])
        except Exception as e:
            print(f"Warning: Could not load vanilla data from {VANILLA_FILE}. Error: {e}")
            vanilla_questions = {}

    # =========================================== Load Dataset ===========================================    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {i+1} due to JSON error.")
                continue
            sentence = data.get('en', None)
            data_id = data.get('id') # Get the unique ID for lookup
            if sentence:
                prompt_template = prompts.get(PROMPT, prompts["vanilla"])
                entities = data.get('ner_entities', [])

                if PROMPT == "entity" and not entities:
                    # Case 1: 'entities' mode is active AND the entity list is empty
                    if data_id in vanilla_questions:
                        # Copy pre-computed vanilla questions
                        data['questions'] = vanilla_questions[data_id]
                        print(f"Skipping LLM generation for line {i} (ID: {data_id}). Copied {len(data['questions'])} questions from vanilla file.")
                        generation = str(data['questions']) # Used for print statement below

                        # Write to output file and continue to next line
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        torch.cuda.empty_cache()
                        continue # Skip the LLM generation block
                    else:
                        # Case 1b: Entities empty, but ID not found in vanilla file. Fall through to LLM generation using vanilla prompt.
                        prompt = prompts["vanilla"].replace("{{sentence}}", sentence)
                        print(f"ID {data_id} not found in vanilla file. Falling back to LLM (vanilla prompt)...")

                if PROMPT == "entities":
                    ner_entities_str = json.dumps(entities, ensure_ascii=False)
                    prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{ner_entities}}", ner_entities_str)
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