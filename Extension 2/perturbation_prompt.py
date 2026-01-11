perturb_synonym_fr = """Task: You will be given a french abstract and your goal is to perturb the french abstract by replacing a few words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed french abstract without giving any additional explanation. Output ONLY the modified french text. Do NOT include labels like "Perturbed:", "Original:", or explanations. Output MUST contain the same number of sentences as the input. Do NOT summarize
*** Example Starts ***
Original: Il peut s’agir d’un membre du secrétariat ou du personnel clinique, selon le protocole de chaque cabinet.
Perturbed: Il peut s’agir d’un membre du secrétariat ou du personnel médical, selon le protocole de chaque cabinet.

Original: En outre, nous recruterons de nouveaux cabinets de surveillance.
Perturbed: De plus, nous engagerons de nouveaux cabinets de suivi.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_word_order_fr = """Task: You will be given a french abstract and your goal is to perturb the french abstract by changing the word order. Please make sure it does not changes the meaning in a significant way. Just output the perturbed french abstract without giving any additional explanation. Do NOT add or remove information. Only reorder words or clauses. Output ONLY the modified french text. Do NOT include labels like "Perturbed:", "Original:", or explanations. Output MUST contain the same number of sentences as the input. Do NOT summarize

*** Example Starts ***
Original: Il peut s’agir d’un membre du secrétariat ou du personnel clinique, selon le protocole de chaque cabinet.
Perturbed: Il peut s’agir d’un membre du personnel clinique ou du secrétariat, selon le protocole de chaque cabinet.

Original: Nous développerons un observatoire visant à présenter les données à l’échelle nationale ainsi qu’un tableau de bord pour faire des remarques aux cabinets quant à la qualité de leurs données et la collecte d’échantillons virologiques et sérologiques.
Perturbed: Un observatoire sera développé pour présenter les données à l’échelle nationale, accompagné d’un tableau de bord destiné à fournir des remarques aux cabinets concernant la qualité des données et la collecte des échantillons virologiques et sérologiques.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_spelling_fr = """Taks: You will be given a french abstract and your goal is to perturb the french abstract by making spelling of a few words wrong. The words should be important words in the french abstract but not words like "le", "et", "la" or "des". Just output the perturbed french abstract without giving any additional explanation. IMPORTANT: Intentionally introduce spelling mistakes. Do NOT correct spelling. Output ONLY the modified french text. Do NOT include labels like "Perturbed:", "Original:", or explanations. Output MUST contain the same number of sentences as the input. Do NOT summarize

*** Example Starts ***
Original: Il peut s’agir d’un membre du secrétariat ou du personnel clinique, selon le protocole de chaque cabinet.
Perturbed: Il peut s’agir d’un membre du serétariat ou du personnel cinique, selon le protocole de chaque cabinet.

Original: En outre, nous recruterons de nouveaux cabinets de surveillance.
Perturbed: En outre, nous recruterons de nouveaux cainets de sureillance.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_noimpact_fr = """Task: You will be given a french abstract and your goal is to perturb the french abstract by adding a few words in the french abstract. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the french abstract and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the french abstract. Just output the perturbed french abstract without giving any additional explanation. Output ONLY the modified french text. Do NOT include labels like "Perturbed:", "Original:", or explanations. Output MUST contain the same number of sentences as the input. Do NOT summarize

*** Example Starts ***
Original: si vous pensez que vos symptômes ou problèmes justifient un examen plus approfondi.
Perturbed: si vous pensez que vos symptômes ou problèmes justifient un examen médical plus approfondi.

Original: En cas de réponse affirmative à ces questions de filtrage, le patient doit être prié de ne pas venir au cabinet et de suivre l’organigramme de PHE à la place.
Perturbed: En cas de réponse affirmative à ces questions de filtrage, le patient adulte doit être prié de ne pas venir au cabinet et de suivre l’organigramme de PHE à la place.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_intensifier_fr = """Task: You will be given a french abstract and your goal is to perturb the french abstract by adding a few words that changes the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the french abstract. Just output the perturbed french abstract without giving any additional explanation. Output ONLY the modified french text. Do NOT include labels like "Perturbed:", "Original:", or explanations. Output MUST contain the same number of sentences as the input. Do NOT summarize

*** Example Starts ***
Original: Les symptômes courants comprennent la fièvre, une toux sèche et la fatigue.
Perturbed: Les symptômes courants comprennent la forte fièvre, une toux sèche et la fatigue.

Original: L’essoufflement, le mal de gorge, les maux de tête, les courbatures ou la production d’expectorations comptent parmi les autres symptômes moins courants.
Perturbed: L’essoufflement sévère, le mal de gorge, les maux de tête, les courbatures ou la production d’expectorations comptent parmi les autres symptômes moins courants.

Original: j’ai fait sur le corps autour de la poitrine ?
Perturbed: j’ai fortement fait sur le corps autour de la poitrine ?

Original: Des formulaires de rapport avaient été soumis aux CDC pour 74 439 cas (60,7 %).
Perturbed: Des formulaires de rapport avaient été soumis aux CDC pour un grand total de 74 439 cas (60,7 %).
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_impact_fr = """Task: You will be given a french abstract and your goal is to perturb the french abstract by adding words in the french abstract. Please make sure that the added word does not disturb the grammaticality of the french abstract but should change the meaning in a significant way. Just output the perturbed french abstract without giving any additional explanation. Output ONLY the modified french text. Do NOT include labels like "Perturbed:", "Original:", or explanations. Output MUST contain the same number of sentences as the input. Do NOT summarize

*** Example Starts ***
Original: Les symptômes courants comprennent la fièvre, une toux sèche et la fatigue.
Perturbed: Les symptômes courants comprennent la fièvre et des douleurs musculaires, une toux sèche et la fatigue.

Original: L’essoufflement, le mal de gorge, les maux de tête, les courbatures ou la production d’expectorations comptent parmi les autres symptômes moins courants.
Perturbed: L’essoufflement, le mal de gorge, les maux de tête, les courbatures, la production d’expectorations et des troubles digestifs comptent parmi les autres symptômes moins courants.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_omission_fr = """Task: You will be given a french abstract and your goal is to perturb the french abstract by removing information from the french abstract. Remove only a few words from the french abstract. Please make sure that the removed information does not disturb the grammaticality of the french abstract but should change the meaning in a significant way. Just output the perturbed french abstract without giving any additional explanation. Output ONLY the modified french text. Do NOT include labels like "Perturbed:", "Original:", or explanations. Output MUST contain the same number of sentences as the input. Do NOT summarize

*** Example Starts ***
Original: Les symptômes courants comprennent la fièvre, une toux sèche et la fatigue.
Perturbed: Les symptômes courants comprennent la fatigue et la fatigue.

Original: Des recherches sur un vaccin ou un traitement antiviral spécifique sont en cours.
Perturbed: Des recherches sur un traitement antiviral spécifique sont en cours.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_alteration_fr = """Task: You will be given a french abstract and your goal is to perturb the french abstract by changing the affirmative sentences into negation (vice versa) or changing one word (noun, verb, adjective or adverb) to its antonym or completely different word. Please make sure that the perturbation does not disturb the grammaticality of the french abstract but should change the meaning in a significant way. Just output the perturbed french abstract without giving any additional explanation. Output ONLY the modified french text. Do NOT include labels like "Perturbed:", "Original:", or explanations. Output MUST contain the same number of sentences as the input. Do NOT summarize

*** Example Starts ***
Original: Il n'a pas réussi à soulager la douleur avec des médicaments que la compétition interdit aux concurrents de prendre.
Perturbed: Il n'a pas réussi à soulager le plaisir avec des médicaments, que la compétition interdit aux concurrents de prendre.

Original: Le mois dernier, un comité présidentiel a recommandé la démission de l'ancien CEP dans le cadre de mesures visant à pousser le pays vers de nouvelles élections.
Perturbed: Le mois dernier, un comité présidentiel n'a pas recommandé la démission de l'ancien CEP dans le cadre de mesures visant à pousser le pays vers de nouvelles élections.

Original: et votre nez coule-t-il ?
Perturbed: et votre nez danse-t-il ?
*** Example Ends ***

Original: {{original}}
Perturbed: """

prompts = {
    # French
    "synonym_fr": perturb_synonym_fr,
    "word_order_fr": perturb_word_order_fr,
    "spelling_fr": perturb_spelling_fr,
    "expansion_noimpact_fr": perturb_expansion_noimpact_fr,
    "intensifier_fr": perturb_intensifier_fr,
    "expansion_impact_fr": perturb_expansion_impact_fr,
    "omission_fr": perturb_omission_fr,
    "alteration_fr": perturb_alteration_fr}