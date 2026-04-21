"""
Approche simple : un seul appel Mistral avec le document et la bibliothèque de faits.

- PDF  → texte extrait via pypdf, envoyé comme contexte textuel
- Image (PNG/JPEG) → encodé en base64, envoyé via l'API vision (pixtral-large-latest)
"""

import base64
import json
import logging
import os
import time
from pathlib import Path

from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError
from pypdf import PdfReader

from facts_extraction.approaches.base import BaseExtractionApproach, ExtractionResult

logger = logging.getLogger(__name__)

# Chemin vers la bibliothèque de faits, relatif à ce fichier
LIBRARY_PATH = Path(__file__).parent.parent / "library.json"

# Mistral Large 3 supporte texte et vision — un seul modèle pour tout
MODEL = "mistral-large-latest"

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'extraction de faits structurés à partir de documents.

Voici la bibliothèque des codes de faits autorisés :
{library}

Règles :
- Extrait uniquement les faits dont le fact_code figure dans la bibliothèque.
- Pour fact_unit, utilise uniquement une valeur présente dans le champ allowed_units du fact_code correspondant. Si aucune unité n'est applicable, mets null.
- Réponds UNIQUEMENT avec un tableau JSON valide, sans texte autour.

Format de sortie attendu :
[
  {{
    "fact_code": "NOM_CODE",
    "fact_value": "valeur",
    "fact_unit": "unité ou null"
  }}
]"""

USER_PROMPT_TEXT = """Voici le contenu du document :

{document_text}

Extrait tous les faits présents qui correspondent à la bibliothèque."""

USER_PROMPT_IMAGE = "Extrait tous les faits présents dans ce document qui correspondent à la bibliothèque."


_RETRY_MAX = 3
_RETRY_FALLBACK_WAIT = 30  # secondes, si Retry-After absent de la réponse


def _call_with_retry(fn):
    """Appelle fn(), relance jusqu'à _RETRY_MAX fois en cas de rate limit (429).

    Utilise le header Retry-After de la réponse Mistral si disponible,
    sinon applique un backoff exponentiel à partir de _RETRY_FALLBACK_WAIT.
    """
    fallback_wait = _RETRY_FALLBACK_WAIT
    for attempt in range(_RETRY_MAX + 1):
        try:
            return fn()
        except SDKError as e:
            if attempt < _RETRY_MAX and e.status_code == 429:
                retry_after = e.headers.get("retry-after")
                if retry_after is not None:
                    wait = float(retry_after)
                    logger.warning(
                        "Rate limit (429) — tentative %d/%d, Retry-After: %.0fs",
                        attempt + 1, _RETRY_MAX, wait,
                    )
                else:
                    wait = fallback_wait
                    logger.warning(
                        "Rate limit (429) — tentative %d/%d, pas de Retry-After, attente fallback: %.0fs",
                        attempt + 1, _RETRY_MAX, wait,
                    )
                    fallback_wait *= 2
                time.sleep(wait)
            else:
                raise


def _load_library() -> dict:
    logger.debug("Chargement de la bibliothèque : %s", LIBRARY_PATH)
    with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
        library = json.load(f)
    logger.debug("%d fact_code(s) chargé(s)", len(library.get("fact_codes", {})))
    return library


def _extract_pdf_text(file_path: str) -> str:
    logger.debug("Extraction du texte PDF : %s", file_path)
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    logger.debug("%d page(s) extraite(s), %d caractères au total", len(pages), len(text))
    return text


def _encode_image_base64(file_path: str) -> tuple[str, str]:
    """Retourne (base64_data, mime_type)."""
    logger.debug("Encodage de l'image en base64 : %s", file_path)
    suffix = Path(file_path).suffix.lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_types.get(suffix, "image/jpeg")
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    logger.debug("Image encodée — mime: %s, taille base64: %d octets", mime_type, len(data))
    return data, mime_type


def _parse_facts(raw: str) -> list[dict]:
    """Parse la réponse JSON du modèle en liste de faits."""
    logger.debug("Parsing de la réponse brute (%d caractères)", len(raw))
    raw = raw.strip()
    # Parfois le modèle entoure le JSON de ```json ... ```
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"La réponse du modèle n'est pas un JSON valide : {e}\n\nRéponse brute :\n{raw}") from e
    if not isinstance(result, list):
        raise ValueError(f"La réponse du modèle devrait être un tableau JSON, reçu : {type(result).__name__}")
    return result


class MistralSimpleApproach(BaseExtractionApproach):
    """
    Extraction par appel unique à Mistral.

    Le document entier (texte ou image) est envoyé dans le contexte,
    accompagné de la bibliothèque de faits autorisés.
    """

    name = "mistral_simple"

    def __init__(self):
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("La variable d'environnement MISTRAL_API_KEY est manquante.")
        self.client = Mistral(api_key=api_key)
        logger.info("MistralSimpleApproach initialisée")

    def extract(self, file_path: str) -> ExtractionResult:
        logger.info("Début extraction — fichier : %s", file_path)
        try:
            library = _load_library()
            library_str = json.dumps(library, ensure_ascii=False, indent=2)
            system_message = SYSTEM_PROMPT.format(library=library_str)

            suffix = Path(file_path).suffix.lower()
            is_image = suffix in {".png", ".jpg", ".jpeg"}

            start = time.monotonic()

            if is_image:
                logger.info("Type : image — modèle : %s", MODEL)
                b64, mime = _encode_image_base64(file_path)
                logger.info("Appel API en cours...")
                response = _call_with_retry(lambda: self.client.chat.complete(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                                },
                                {"type": "text", "text": USER_PROMPT_IMAGE},
                            ],
                        },
                    ],
                ))
            else:
                logger.info("Type : PDF — modèle : %s", MODEL)
                document_text = _extract_pdf_text(file_path)
                logger.info("Appel API en cours...")
                response = _call_with_retry(lambda: self.client.chat.complete(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": USER_PROMPT_TEXT.format(document_text=document_text),
                        },
                    ],
                ))

            latency_ms = (time.monotonic() - start) * 1000
            logger.info("Réponse reçue — tokens: %d input / %d output — latence: %.0fms",
                        response.usage.prompt_tokens, response.usage.completion_tokens, latency_ms)

            raw_content = response.choices[0].message.content
            facts = _parse_facts(raw_content)
            logger.info("%d fait(s) extrait(s)", len(facts))

            return ExtractionResult(
                facts=facts,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
                metadata={"model": MODEL, "file": file_path},
            )

        except Exception as e:
            logger.error("Échec de l'extraction pour '%s' : %s", file_path, e)
            raise
