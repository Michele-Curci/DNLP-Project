from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
from QG_prompt import prompts
from huggingface_hub import notebook_login

# ====================================== Cache and device ======================================
own_cache_dir = "/content/.cache"
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

model_id = "google/gemma-2-9b-it"

# ======================================
output_path = "output_ner.jsonl" 
selected_prompt = "entity"            # <-- "vanilla", "atomic", "semantic", "entity"
input_file = "Datasets/Extension 1/Perturbations/entity_dataset.jsonl"
vanilla_file = "Datasets/Extension 1/original_files/vanilla_gemma-9b.jsonl"
# ======================================


# Login to Hugging Face (useful when using a nootebook)
#notebook_login()

# Load model with 4-bit quantization
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Use bfloat16 for computations as it's the native format for Gemma
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=own_cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    cache_dir=own_cache_dir,
    device_map="auto",
)

# Load the vanilla data into a dictionary for fast lookup by 'id'
vanilla_questions = {}
if os.path.exists(vanilla_file):
    try:
        with open(vanilla_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Store only the questions associated with the unique ID
                vanilla_questions[data.get('id')] = data.get('questions', [])
    except Exception as e:
        print(f"Warning: Could not load vanilla data from {vanilla_file}. Error: {e}")
        vanilla_questions = {}

with open(input_file, 'r') as f_in, open(output_path, 'w') as f_out:
    for i, line in enumerate(f_in, start=1):
        data = json.loads(line)
        sentence = data.get('en', None)

        if not sentence:
            continue

        data_id = data.get('id') # Get the unique ID for lookup

        # Get the template
        prompt_template = prompts.get(selected_prompt, prompts["vanilla"])

        # ------------------------- SPECIFIC PROMPTS -------------------------
        entities = data.get('ner_entities', [])

        if selected_prompt == "entity" and not entities:
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

        elif selected_prompt == "entity":
            ner_entities_str = json.dumps(entities, ensure_ascii=False)
            prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{ner_entities}}", ner_entities_str)
        else:  # vanilla
            prompt = prompt_template.replace("{{sentence}}", sentence)
        # ---------------------------------------------------------------------

        # Tokenizer and generation
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=1024,
                num_beams=1,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract part after "Questions:"
        answer_start = "Questions:"
        if answer_start in generated:
            generation = generated.split(answer_start)[-1].strip()
        else:
            generation = generated

        # Convert in Python list
        try:
            data['questions'] = eval(generation)
        except:
            data['questions'] = [generation]

        # Debug print
        print("\nGENERATO:\n", data['questions'])
        print("="*80)

        # Salva nel file
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()