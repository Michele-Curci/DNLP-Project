from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
import re
from QG_prompt import prompts
from huggingface_hub import notebook_login

# ====================================== Cache and device ======================================
own_cache_dir = "/content/.cache"
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

model_id = "google/gemma-2-9b-it"

# ======================================
#   PARAMETERS
# ======================================
output_path = "output_ner.jsonl"
selected_prompt = "entity"            # <-- "vanilla", "atomic", "semantic", "entity"
input_file = "entity.jsonl"         # <-- file input

# Login to Hugging Face
notebook_login()

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

with open(input_file, 'r') as f_in, open(output_path, 'w') as f_out:
    for i, line in enumerate(f_in, start=1):
        if i < 1: continue
        if i > 100: break
        data = json.loads(line)
        sentence = data.get('en', None)

        if not sentence:
            continue

        # Template
        prompt_template = prompts.get(selected_prompt, prompts["vanilla"])

        # ------------------------- PROMPTS -------------------------
        entities = data.get('ner_entities', [])

        if selected_prompt == "entity":
            ner_entities_str = json.dumps(entities, ensure_ascii=False)
            prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{ner_entities}}", ner_entities_str)
        else:  # vanilla
            prompt = prompt_template.replace("{{sentence}}", sentence)
        # ---------------------------------------------------------------------

        # Tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=1024,
                num_beams=1,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer_start = "Questions:"
        if answer_start in generated:
            generation = generated.split(answer_start)[-1].strip()
        else:
            generation = generated

        try:
            data['questions'] = eval(generation)
        except:
            data['questions'] = [generation]

        print("\nGENERATED:\n", data['questions'])
        print("="*80)

        # Save
        out_data = {
          "id": i,
          "en": data.get("en", ""),
          "ner_entities": data.get("ner_entities", []),
          "questions": data.get("questions", [])}

        f_out.write(json.dumps(out_data, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()