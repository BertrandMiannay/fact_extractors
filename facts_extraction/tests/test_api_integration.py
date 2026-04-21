"""
Test d'intégration réel : appel à la vraie API Mistral.

Objectif : débuguer les erreurs 429 et vérifier que l'extraction fonctionne
sur le fichier PNG avec le vrai modèle.

Lancer avec :
    python -m pytest facts_extraction/tests/test_api_integration.py -v -s
"""

import logging
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PNG_PATH = Path(__file__).parent.parent / "data" / "basic patient data.png"


def requires_api_key():
    if not os.environ.get("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY absent — test ignoré")


def test_real_api_png_extraction():
    """Appelle la vraie API Mistral sur le fichier PNG et affiche le résultat."""
    requires_api_key()

    if not PNG_PATH.exists():
        pytest.skip(f"Fichier PNG absent : {PNG_PATH}")

    from facts_extraction.approaches.mistral_simple import MistralSimpleApproach

    logger.info("=== Début du test d'intégration réel ===")
    logger.info("Fichier : %s", PNG_PATH)
    logger.info("Taille fichier : %d octets", PNG_PATH.stat().st_size)

    approach = MistralSimpleApproach()
    result = approach.extract(str(PNG_PATH))

    logger.info("=== Résultat ===")
    logger.info("Faits extraits : %s", result.facts)
    logger.info("Tokens input   : %d", result.input_tokens)
    logger.info("Tokens output  : %d", result.output_tokens)
    logger.info("Latence        : %.0f ms", result.latency_ms)

    assert isinstance(result.facts, list), "Le résultat doit être une liste"
    assert result.input_tokens > 0, "Des tokens input doivent avoir été consommés"
