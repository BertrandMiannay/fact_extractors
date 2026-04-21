import json
import time
from pathlib import Path
from api.base import BaseAPI
from data_extraction.approaches.base import BaseApproach, ApproachResponse
from data_extraction.approaches.utils import load_pdf_texts

SUMMARY_PROMPT = """Analyse ce document réglementaire de plongée et retourne UNIQUEMENT un objet JSON valide avec les champs suivants :
{
  "niveau": <numéro entier du niveau concerné>,
  "role": <"plongeur" | "encadrant" | "directeur">,
  "profondeur_max_encadre": <profondeur max en mètres en étant encadré, ou null>,
  "profondeur_max_autonome": <profondeur max en mètres en autonomie, ou null>,
  "sujets": <liste des thèmes principaux couverts par le document>,
  "prerequis": <liste des conditions d'accès principales>
}
Ne retourne rien d'autre que le JSON."""

FILTER_PROMPT = """Tu es un assistant de sélection documentaire. On te fournit des fiches JSON de documents réglementaires de plongée et une question.

Procède en deux étapes :

ÉTAPE 1 — Analyse la question et identifie :
- Le niveau mentionné (1, 2, 3, 4 ou 5) → sélectionne le document dont "niveau" correspond exactement
- Le rôle mentionné ("directeur" → niveau 5, "guide de palanquée" → niveau 4) → sélectionne le document dont "role" correspond
- Si la question porte sur des prérequis d'accès à un niveau N → sélectionne uniquement le document de niveau N
- Si aucun niveau ni rôle n'est identifiable → sélectionne au maximum 2 documents dont les "sujets" correspondent

ÉTAPE 2 — Retourne UNIQUEMENT un tableau JSON avec les noms exacts des documents retenus.
Sélectionne le minimum possible. Maximum 2 documents.

Exemples :
- "prérogatives du niveau 1" → ["Plongeur Niveau 1 - PE20"]
- "profondeur autonome niveau 2" → ["Plongeur Niveau 2 - PA20 - PE40"]
- "rôle du directeur de plongée" → ["Niveau 5 - Directeur de plongée en exploration"]
- "conditions d'accès niveau 3" → ["Plongeur Niveau 3 - PA40 - PE60 - PA60"]"""

ANSWER_PROMPT = """Tu es un expert en plongée sous-marine.
Réponds uniquement en français, de manière succincte, précise et sourcée, en te basant exclusivement sur les documents fournis.
Si la réponse n'est pas dans les documents, dis-le clairement.
Ne fournis jamais de liens externes."""

CACHE_FILE = ".cache/summaries.json"


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        return text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text


class StepByStepApproach(BaseApproach):
    name = "step_by_step"

    def __init__(self, api: BaseAPI, pdf_dir: str = "data"):
        self.api = api
        self.documents = load_pdf_texts(pdf_dir)
        self.summaries = self._load_or_generate_summaries()
        self._summaries_text = "\n\n".join(
            f"[{name}]\n{json.dumps(summary, ensure_ascii=False)}"
            if isinstance(summary, dict)
            else f"[{name}]\n{summary}"
            for name, summary in self.summaries.items()
        )

    def _load_or_generate_summaries(self) -> dict:
        cache_path = Path(CACHE_FILE)
        cached = {}
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)

        missing = [name for name in self.documents if name not in cached]
        if missing:
            print(f"Génération des résumés pour : {missing}")
            for name in missing:
                response = self.api.send(
                    messages=[{"role": "user", "content": self.documents[name]}],
                    system=SUMMARY_PROMPT,
                )
                try:
                    cached[name] = json.loads(_strip_code_fence(response.content.strip()))
                except json.JSONDecodeError:
                    cached[name] = response.content

            cache_path.parent.mkdir(exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)

        return cached

    def _select_documents(self, question: str) -> tuple[list[str], int, int, float]:
        messages = [{
            "role": "user",
            "content": f"{self._summaries_text}\n\n---\n\nQuestion : {question}",
        }]
        start = time.monotonic()
        response = self.api.send(messages, system=FILTER_PROMPT)
        latency_ms = (time.monotonic() - start) * 1000

        try:
            selected = json.loads(_strip_code_fence(response.content.strip()))
            selected = [s for s in selected if s in self.documents]
        except (json.JSONDecodeError, TypeError):
            selected = list(self.documents.keys())

        if not selected:
            selected = list(self.documents.keys())

        return selected, response.input_tokens, response.output_tokens, latency_ms

    def ask(self, question: str) -> ApproachResponse:
        selected, filter_in, filter_out, filter_latency = self._select_documents(question)

        context = "\n\n".join(f"=== {name} ===\n{self.documents[name]}" for name in selected)
        messages = [{
            "role": "user",
            "content": f"{context}\n\n---\n\nQuestion : {question}",
        }]
        answer_response = self.api.send(messages, system=ANSWER_PROMPT)

        return ApproachResponse(
            answer=answer_response.content,
            input_tokens=filter_in + answer_response.input_tokens,
            output_tokens=filter_out + answer_response.output_tokens,
            latency_ms=filter_latency + answer_response.latency_ms,
            metadata={
                "selected_documents": selected,
                "filter_tokens": {"input": filter_in, "output": filter_out},
                "answer_tokens": {
                    "input": answer_response.input_tokens,
                    "output": answer_response.output_tokens,
                },
            },
        )
