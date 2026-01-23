## DNLP-Project

# Extensions of ASKQE: entity recognition and abstract-level shift

This repository contains the code and the dataset related to our extensions of the ASKQE framework explained in the report: 'Extensions of ASKQE: entity recognition and abstract-level shift'.

## Abstract

Automatic evaluation of Machine Translation (MT) quality is a complex task, especially in sensitive domains such as medicine. ASKQE addresses this challenge by generating questions about the source text and comparing answers obtained from the original sentence and from the back-translated MT output. This work presents two extensions of the ASKQE framework aimed at enhancing both its interpretability and generalization. The first extension integrates Named Entity Recognition (NER) into the ASKQE pipeline, leveraging extracted entities both as additional context during question generation (QG) and as a tool in question answering (QA) evaluation phase. Experimental results on the synthetic medical dataset defined by ASKQE (CONTRACTICO) show that NER-based questions exhibit trends comparable to those observed with vanilla, atomic-fact, and semantic-role questions, effectively distinguishing between minor and critical translation perturbations. Moreover, the entity-level error classification, performed computing the similarity among the entities generated from original and back-translated answers, enables a more interpretable analysis of translation errors for the different perturbations. The second extension investigates a domain shift from single sentences to longer texts, specifically medical abstracts, subject to the same perturbations introduced in CONTRACTICO. The QA evaluation results for the collection of abstracts maintain the same trend of those of single sentences, exhibiting lower similarity scores for critical perturbations. In general, the results obtained for abstracts exceed those of single sentences, benefiting from the richer contextual information they provide. Overall, the findings show that entity enrichment enhances the explainability of ASKQE and that the framework can be effectively generalized to a document domain, supporting its applicability to larger textual units.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ebb29d1f-1cf2-43b7-a907-a5178446bf0f" width="500">
</p>


## Organization of the repository:

The repository is divided into three folders:

1. Extension1: contains all the code for the first extension, “entity enrichment”.
2. Extension2: contains all the code for the second extension, “domain shift from single sentences to abstracts”.
3. Datasets: contains all the results file and the original files taken by ASKQE work.

## **Extension 1**

The proposed extension adopts a structured pipeline to assess whether incorporating named entity recognition (NER) can strengthen the ASKQE framework.
More specifically, the goal is to determine whether NER provides a meaningful contribution to both the question-generation stage and the QA evaluation process.

The folder is structured as follows:

1) **entity_extraction**.py contains the code used for extracting the entities both from original sentences and from the answers depending on the input file selected:

    * use 'qg.variants.json' as ‘INPUT FILE’ to extract entities from original sentences;

    * use files present in 'QA_results' folder to extract entities from answers.

2) **QG**.py, **QG_prompt**.py contain respectively the code and the prompt for NER-based questions generation, the selected prompt is equal to ‘entity’ to generate questions using the entities extracted in step 1.
3) **QA**.py and **QA_prompt**.py contain respectively the code and the prompt for generating answers from original sentences and back-translated ones. ‘backtranslate_file’ must be selected among the ones present in the folder 'Perturbations' contained in 'Datasets/Extension1'.
4) **evaluationQA**.ipynb is a notebook containing all the ASKQE metrics used to assess the performance for all the perturbations, select as ‘predicted_file’ one among those contained in QA_results folder (depending on the perturbation you want to analyze) present in 'Datasets/Extension1'.
5) **count_entities**.py is used to see the types of entities extracted from answers, use as input file (depending on the perturbation you want to analyze) one among the files contained in ‘answer_entity_results’ folder present in 'Datasets/Extension1'.
6) **alignment**.py contains the code for the alignment between entities extracted from original and perturbed answers. The input files are contained in ‘answer_entity_results’ folder present in 'Datasets/Extension1', ‘PERTURBED FILE’ must be changed each time to consider the entities extracted from all the type of perturbations.
7) **count_error_type**.py is used to classify the type of entity errors, use as input file (depending on the perturbation you want to analyze) one among the files contained in ‘error_classification_results’ folder present in 'Datasets/Extension1'.

Execution order of  the code for Extension1:

* run entity_extraction.py to generate the entities from original sentences;
* run QG.py, QG_prompt.py to generate questions based on the entities extracted in the previous step;
* run QA.py and QA_prompt.py to generate the answers from original sentences and back-translated ones;
* run evaluationQA.ipynb to assess the performance of the model, in particular the similarity between original and back translated answers for all the perturbations;
* run entity_extraction.py to generate the entities for original and perturbed answers;
* run count_entities.py to see the types of entities extracted from answers;
* run alignment.py to evaluate the similarity between the entities extracted from original and perturbed answers;
* run count_error_type.py to classify the type of entity errors.

## Extension 2

The aim of this extension is to test if the ASKQE framework could be generalized to longer pieces of text. In particular, the work focuses on a domain shift from single medical English sentences to a collection of English abstracts in the same field.
In order to replicate the ASKQE pipeline for a collection of documents, a synthetic dataset was created introducing in the original abstracts the same perturbations of the CONTRACTICO dataset considered by ASKQE.

The folder is structured as follows:

1) **entity_generation**.py contains the code used for extracting the entities from original abstracts coming from "TimSchopf/medical_abstracts" dataset.
2) **QG**.py, **QG_prompt**.py contain respectively the code and the prompt for NER-based questions generation, the selected prompt is equal to ‘entity’ to generate questions using the entities extracted.
3) **translation**.py contains the code for translating original English abstracts into French.
4) **perturbation**.py and **perturbation_prompt**.py contain respectively the code and the prompt for generating perturbations of French abstracts.
5) **backtranslation**.py contains the code for translating perturbed French abstracts back into English.
6) **QA**.py and **QA_prompt**.py contain respectively the code and the prompt for generating answers from original abstracts and back-translated ones. ‘backtranslate_file’ must be selected among the ones (depending on the perturbation you want to analyze) in the folder 'Backtranslation' present in 'Datasets/Extension2'.
7) **evaluation**.ipynb is a notebook containing all the ASKQE metrics used to assess the performance for all the perturbations, select as ‘predicted_file’ one among those contained in 'QA_results' folder (depending on the perturbation you want to analyze) present in 'Datasets/Extension2'.

Execution order of  the code for Extension2:

* run entity_generation.py to generate the entities from original abstracts;
* run Qg.py, QG_prompt.py to generate questions based on the entities extracted in the previous step;
* run translation.py to translate English abstracts into French;
* run perturbation.py and perturbation_prompt.py to generate perturbations of French abstracts;
* run backtranslation.py to backtranslate perturbed French abstracts into English;
* run QA.py and QA_prompt.py to generate the answers from original abstracts and back-translated ones;
* run evaluationQA.ipynb to assess the performance of the model, in particular the similarity between original and back translated answers for all the perturbations.

## **Datasets**

The folder is divided in two subfolders: Extension1, Extension2.

**Extension 1** contains the following folders:

* **original_files** and **Perturbations** contain the files taken by ASKQE. In particular, the folder ‘original_files’ include ‘qg_variants.json’ that is used to retrieve original sentences and 'vanilla_gemma-9b.jsonl' that contains questions generated in the vanilla case, used as default when it is not possible to generate questions from entities. The folder ‘Perturbations’ contains all the perturbations of CONTRACTICO dataset built in ASKQE framework.

* **entity_dataset** file contains the entities extracted from original sentences.

* **QG_result** contains the file ‘output_ner.json’, which is the outcome of questions generation, in particular it includes the questions generated using entities.

* **QA_result** contains the file with the answers extracted from original sentences (‘source_answer.json’) and all the files with the answers extracted from backtranslated perturbed sentences, one output file for each perturbation.

* **evaluation_results** contains the files with the results obtained from the evaluation phase for all the perturbations, one output file for each perturbation.

* **answer_entity_results** contains the file with the entities associated to answers extracted from original sentences (‘source_entity_answer.json’) and all the files with the entities extracted from answers coming from backtranslated perturbed sentences, one output file for each perturbation.

* **error_classification_results** includes all the files reporting the types of entity-errors made with their scores, one output file for each perturbation.

**Extension 2** includes the following folders and files:

* **entity.jsonl** file contains the entities extracted from original abstracts.

* **perturbation_results** folder includes all perturbed French abstracts, one output file for each type of perturbation.

* **translation.jsonl** file contains the translation of English abstracts into French.

* **Backtranslations** folder includes the backtranslation from French to English of perturbed abstracts, one output file for each perturbation.

* **QG_result** contains the file ‘output_ner.json’, which is the outcome of questions generation, in particular it includes the questions generated from original abstracts using entities.

* **QA_result** contains the file with the answers extracted from original abstracts (‘source_answer.json’) and all the files with the answers extracted from backtranslated perturbed abstracts, one output file for each perturbation.

* **evaluation_results** contains the files with the results obtained from the evaluation phase for all the perturbations, one output file for each perturbation.
