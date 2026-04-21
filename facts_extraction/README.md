# Facts Extraction

Benchmark d'extraction de faits structurés à partir de documents biologiques (PDF, PNG, JPEG).

## Objectif

Pour un document donné, un LLM doit extraire des **faits codifiés** et leur **localisation** dans le document. Les résultats sont comparés à une liste de faits attendus (`expected_facts`) pour calculer des métriques de qualité.

## Structure du projet

```
facts_extraction/
├── data/                  # Documents biologiques (PDF, PNG, JPEG)
├── approaches/            # Différentes stratégies d'extraction
│   └── base.py            # Interface commune (BaseExtractionApproach)
├── benchmark/
│   ├── runner.py          # Orchestration du benchmark
│   └── questions.json     # Documents + expected_facts
├── library.json           # Bibliothèque des fact_codes autorisés
└── README.md
```

## Bibliothèque de faits (`library.json`)

Chaque type de fait est défini par un `fact_code` avec ses métadonnées :

```json
{
  "fact_codes": {
    "NOM_CODE": {
      "description": "Ce que représente ce fait",
      "unit": "unité ou null",
      "allowed_values": []
    }
  }
}
```

## Format des faits extraits

Un fait extrait par le LLM doit respecter cette structure :

```json
{
  "fact_code": "NOM_CODE",
  "fact_value": 42,
  "fact_unit": "unité",
  "location": {
    "page": 3,
    "context_snippet": "extrait de texte où le fait apparaît"
  }
}
```

## Benchmark (`benchmark/questions.json`)

Pour chaque document, une liste de `expected_facts` sert de référence :

```json
{
  "documents": [
    {
      "id": "identifiant_unique",
      "path": "data/mon_document.pdf",
      "type": "pdf",
      "expected_facts": [
        {
          "fact_code": "NOM_CODE",
          "fact_value": 42,
          "fact_unit": "unité",
          "location": {
            "page": 3,
            "context_snippet": "extrait de texte..."
          }
        }
      ]
    }
  ]
}
```

## Évaluation

L'évaluation se fait par comparaison exacte sur le triplet `(fact_code, fact_value, fact_unit)` :

| Métrique | Définition |
|----------|-----------|
| **Précision** | Faits extraits corrects / Total faits extraits |
| **Rappel** | Faits attendus trouvés / Total faits attendus |
| **F1** | Moyenne harmonique Précision / Rappel |

Les métriques sont calculées par document et agrégées sur l'ensemble du benchmark.

## Ajouter un nouveau document

1. Déposer le fichier dans `data/`
2. Définir les nouveaux `fact_codes` dans `library.json` si nécessaire
3. Ajouter le document et ses `expected_facts` dans `benchmark/questions.json`
