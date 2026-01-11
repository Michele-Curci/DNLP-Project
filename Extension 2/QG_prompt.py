vanilla = """Task: You will be given a full English medical abstract. Your goal is to generate a list of relevant and comprehensive questions that can be answered using the information contained in the abstract. The questions should cover the main objectives, methods, results, entities, and conclusions described in the abstract.
Output only the list of questions in Python list format without giving any additional explanation.

*** Example Starts ***
Abstract: This study investigates the effects of an anti-inflammatory agent on tissue changes around loose prostheses using a canine model. Biological responses were measured using interleukin-1 and prostaglandin E2 activity.
Questions: ["What is the main objective of the study?", "What model is used in the study?", "What biological responses are measured?", "Which agents are investigated in relation to prosthetic loosening?", "What outcomes are evaluated in the canine model?"]
*** Example Ends ***

Abstract: {{sentence}}
Questions: """


nli = """Task: You will be given a full English medical abstract and a list of atomic facts, each expressing a single piece of information derived from the abstract. Your goal is to generate a list of relevant questions that collectively cover all the atomic facts.
Output the list of questions in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Abstract: Serum NSE levels were normal in benign pheochromocytoma and elevated in malignant cases.
Atomic facts: [
  "Serum NSE levels are normal in benign pheochromocytoma.",
  "Serum NSE levels are elevated in malignant pheochromocytoma."
]
Questions: ["What are the serum NSE levels in benign pheochromocytoma?", "How do serum NSE levels differ in malignant pheochromocytoma?"]
*** Example Ends ***

Abstract: {{sentence}}
Atomic facts: {{atomic_facts}}
Questions: """


srl = """Task: You will be given a full English medical abstract and a dictionary of semantic roles extracted from the abstract. Your goal is to generate a list of relevant questions that reflect the actions, participants, measurements, conditions, and outcomes described across the abstract.
Output the list of questions in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Abstract: The authors evaluated whether serum NSE could distinguish between benign and malignant pheochromocytoma.
Semantic roles: {
  'Verb1': {'Verb': 'evaluated', 'ARG0': 'the authors', 'ARG1': 'whether serum NSE could distinguish between benign and malignant pheochromocytoma'},
  'Verb2': {'Verb': 'distinguish', 'ARG0': 'serum NSE', 'ARG1': 'benign and malignant pheochromocytoma'}
}
Questions: ["What did the authors evaluate?", "What is serum NSE evaluated for?", "Between which conditions is serum NSE used to distinguish?"]
*** Example Ends ***

Abstract: {{sentence}}
Semantic roles: {{semantic_roles}}
Questions: """




ner = """Task: You will be given a full English medical abstract and a list of named entities extracted from it. Each entity is represented as a dictionary with its text and type.

Rules:
- Generate AT LEAST 5 questions.
- DO NOT return an empty list.
- EVERY entity MUST appear in at least one question.
- If an entity is unclear or generic, create a generic question that mentions it.
- If necessary, rephrase entities to fit a question.
- Output questions in the format ["Q1", "Q2"]

*** Example Starts ***
Abstract: Neuropeptide Y and neuron-specific enolase levels were evaluated in patients with benign and malignant pheochromocytoma.
Named entities: [
  {"text": "Neuropeptide Y", "type": "DIAGNOSTIC_PROCEDURE"},
  {"text": "neuron-specific enolase", "type": "DIAGNOSTIC_PROCEDURE"},
  {"text": "pheochromocytoma", "type": "DISEASE_DISORDER"},
  {"text": "benign", "type": "DETAILED_DESCRIPTION"},
  {"text": "malignant", "type": "DETAILED_DESCRIPTION"}
]
Questions: [
  "Which diagnostic markers are evaluated in the study?",
  "What disease is investigated in the abstract?",
  "How are benign and malignant forms of pheochromocytoma compared?",
  "What measurements are used to differentiate disease types?"
]
*** Example Ends ***

Abstract: {{sentence}}
Named entities: {{ner_entities}}
Questions: """



prompts = {
    "vanilla": vanilla,
    "atomic": nli,
    "semantic": srl,
    "entity": ner
}