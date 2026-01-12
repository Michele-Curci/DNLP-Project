from openai import OpenAI
import json
from perturbation_prompt import prompts

OPENAI_API_KEY = "<YOUR_ACTUAL_OPENAI_API_KEY>"

# Initialize token counters globally
input_tokens = 0
output_tokens = 0
total_tokens = 0

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENAI_API_KEY,
)

def call_nvidia_llm(prompt):
    # This function now returns the full response object
    response_obj = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response_obj

language = "fr"
perturbations = [
    "synonym", "word_order", "spelling", "expansion_noimpact",
    "intensifier", "expansion_impact", "omission", "alteration"
]

custom_pert = ["synonym","spelling"]



input_file = "Datasets/Extension 2/translation.jsonl"

for perturbation in custom_pert:
    print("Perturbation: ", perturbation)
    output_file = f"{perturbation}.jsonl"

    i = 1
    with open(input_file, "r", encoding="utf-8") as file, open(output_file, "w", encoding="utf-8") as out_file:
        for line in file:
            data = json.loads(line)

            if "fr" in data:
                sentence = data["fr"]
                prompt = prompts[f"{perturbation}_fr"].replace("{{original}}", sentence)

                print(prompt)
                # Get the full response object
                full_response = call_nvidia_llm(prompt)
                # Extract content
                pert_fr_content = full_response.choices[0].message.content
                print("> ", pert_fr_content)
                print("=" * 80)


                data["perturbation"] = perturbation
                data["pert_fr"] = pert_fr_content
                data["id"] = i

                out_file.write(json.dumps(data, ensure_ascii=False) + "\n")

                i+=1

                # Update global token counters from the full_response object
                if hasattr(full_response, 'usage') and full_response.usage:
                    input_tokens += full_response.usage.prompt_tokens
                    output_tokens += full_response.usage.completion_tokens
                    total_tokens += full_response.usage.total_tokens



print("Input tokens:", input_tokens)
print("Output tokens:", output_tokens)
print("Total tokens:", total_tokens)