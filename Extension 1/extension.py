import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import os
import pandas as pd


INPUT_FILE = f"QG/code/qg_variants.jsonl"
OUTPUT_FILE = f"Extension 1/entity_dataset.jsonl"

BIOBERT_MODEL_NAME = "Ishan0612/biobert-ner-disease-ncbi"
# General Domain NER Model: Recognizes Persons (PER), Organizations (ORG), Locations (LOC), and Misc. (MISC)
MODEL = "NeuronZero/MED-NER"

GENERAL_NER_MODEL = MODEL


def setup_ner_pipeline():
    """Initializes the Named Entity Recognition pipeline."""
    # Prioritize GPU (CUDA device 0)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    try:
        ner_pipeline = pipeline(
            "ner", 
            model=GENERAL_NER_MODEL, 
            tokenizer=GENERAL_NER_MODEL,
            device=device,
            aggregation_strategy="simple" # Groups multi-token entities (e.g., 'New York')
        )
        print(f"Successfully loaded NER model: {GENERAL_NER_MODEL}")
        return ner_pipeline
    except Exception as e:
        print(f"Error loading NER pipeline: {e}")
        return None

def extract_entities(text, ner_pipeline):
    """Processes text to extract and format unique entities."""
    
    # 1. CRITICAL: Handle non-string/empty input immediately
    if not isinstance(text, str) or not text.strip():
        return []
        
    try:
        # Run the NER pipeline for a SINGLE text input
        # The result here should be a list of dictionaries if entities are found, or an empty list []
        results = ner_pipeline(text)
    except Exception as e:
        # Catch issues during processing (e.g., extremely long sentence)
        print(f"Error processing sentence: {text[:50]}... Error: {e}")
        return []
        
    entities = []
    
    # 2. Iterate through the results only if the pipeline found entities
    # Note: If the pipeline is processing a single string, 'results' is a list of entity dictionaries.
    for entity_data in results:
        word = entity_data.get('word', '').strip()
        entity_type = entity_data.get('entity_group', '')
        
        # 3. Check for valid entity output
        # The aggregation_strategy="simple" usually ensures 'word' is a full entity name.
        if word and entity_type and not entity_type.startswith('O'):
            entities.append({
                "text": word, 
                "type": entity_type
            })

    unique_entities = []
    seen = set()
    for entity in entities:
        # Create a hashable representation
        entity_tuple = (entity['text'], entity['type'])
        if entity_tuple not in seen:
            unique_entities.append(entity)
            seen.add(entity_tuple)
            
    return unique_entities

def process_dataset():
    """Main function to load data, process NER, and save results."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found. Please ensure it is in the correct directory.")
        return

    # 1. Setup NER
    ner_pipeline = setup_ner_pipeline()
    if not ner_pipeline:
        return

    # 2. Load Data
    print(f"\nLoading data from {INPUT_FILE}...")
    try:
        # Load the JSONL file line by line
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return

    # 3. Process and Enrich Data
    print(f"Processing {len(data)} sentences for NER...")
    
    # Iterate through the data and add the new 'ner_entities' field
    for i, item in enumerate(data):
        sentence = item.get('en', '') 
        
        # Run the NER extraction
        item['ner_entities'] = extract_entities(sentence, ner_pipeline)
    
    # 4. Save New Dataset
    print(f"Saving enriched data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("\n✅ Processing Complete.")
    print(f"New dataset saved to {OUTPUT_FILE}")


process_dataset()