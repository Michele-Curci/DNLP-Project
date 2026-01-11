import torch
import json
import argparse
from QA_prompt import qa_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import notebook_login
import os

model_id = "google/gemma-2-9b-it"

own_cache_dir = "/content/.cache"
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

questions_file = "output_ner.jsonl"
backtrans_file = "bt-alteration.jsonl"
output_file = "source_answers.jsonl"

def main():
    notebook_login()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Use bfloat16 for computations as it's the native format for Gemma
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        #torch_dtype=torch.bfloat16,
        cache_dir="",
        device_map="auto"
    )

    # =========================================== Load Dataset ===========================================

    with open(backtrans_file, 'r') as f_bk:
          bk_data = {json.loads(l)["id"]: json.loads(l) for l in f_bk}

    with open(questions_file, 'r') as f_q, open(output_file, 'w') as f_out:
      for i, q_line in enumerate(f_q, start=1):
          if i < 1: continue
          if i > 971: break
          q_data = json.loads(q_line)
          entry_id = q_data.get("id")
          questions = q_data.get("questions", [])
          bk_sentence = bk_data.get(entry_id, {}).get("bt_pert_fr", "")
          #bk_sentence = bk_data.get(entry_id, {}).get("en", "")

          if not questions or not bk_sentence:
             continue

          # build the prompt using qa_prompt
          questions_str = json.dumps(questions, ensure_ascii=False)
          prompt = qa_prompt.replace("{{sentence}}", bk_sentence).replace("{{questions}}", questions_str)

          input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
          with torch.no_grad():
                        outputs = model.generate(
                            **input_ids,
                            max_new_tokens=1024,
                            num_beams=1,
                        )

          generated_questions = tokenizer.decode(outputs[0], skip_special_tokens=True)

          answer_start = "Answers: "
          if answer_start in generated_questions:
                    generation = generated_questions.split(answer_start)[-1].strip()
                    generation = generation.split("<")[0].strip()
          else:
                    generation = generated_questions
          try:
            answers = eval(generation)  # converte in lista Python
          except:
            answers = [generation]


          print(f"{prompt}")
          print(f"> {generation}")
          print("\n======================================================\n")

          # Salvataggio
          out_data = {
            "id": entry_id,
            "source": bk_sentence,
            "questions": questions,
            "answers": answers}
          f_out.write(json.dumps(out_data, ensure_ascii=False) + "\n")

          # Svuoto cache GPU
          torch.cuda.empty_cache()
if __name__ == "__main__":
    main()