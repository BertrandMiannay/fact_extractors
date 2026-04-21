"""
Tests d'intégration pour MistralSimpleApproach sur le fichier PNG.

L'appel API Mistral est mocké : on teste toute la chaîne
(lecture fichier → encodage base64 → prompt → parsing réponse)
sans consommer de quota.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from facts_extraction.approaches.mistral_simple import (
    MistralSimpleApproach,
    _encode_image_base64,
    _parse_facts,
)

PNG_PATH = Path(__file__).parent.parent / "data" / "basic patient data.png"

# Réponse fictive que "Mistral" renvoie
FAKE_FACTS = [
    {"fact_code": "ABC", "fact_value": "1.12", "fact_unit": "gr"}
]


def _make_fake_response(facts: list[dict]) -> MagicMock:
    """Construit un objet réponse qui imite mistralai ChatCompletionResponse."""
    response = MagicMock()
    response.choices[0].message.content = json.dumps(facts)
    response.usage.prompt_tokens = 500
    response.usage.completion_tokens = 42
    return response


@pytest.fixture
def approach():
    """Instancie MistralSimpleApproach avec une clé API factice."""
    with patch.dict(os.environ, {"MISTRAL_API_KEY": "fake-key"}):
        with patch("facts_extraction.approaches.mistral_simple.Mistral"):
            return MistralSimpleApproach()


# ---------------------------------------------------------------------------
# Tests unitaires des fonctions utilitaires
# ---------------------------------------------------------------------------

def test_encode_image_base64_returns_jpeg_mime():
    """Le fichier PNG est encodé en base64 avec le bon mime type."""
    if not PNG_PATH.exists():
        pytest.skip(f"Fichier PNG absent : {PNG_PATH}")
    b64, mime = _encode_image_base64(str(PNG_PATH))
    assert mime == "image/png"
    assert len(b64) > 0
    # Un PNG valide commence par iVBOR en base64
    assert b64.startswith("iVBOR")


def test_parse_facts_valid_json():
    raw = json.dumps(FAKE_FACTS)
    result = _parse_facts(raw)
    assert result == FAKE_FACTS


def test_parse_facts_strips_markdown_fence():
    raw = f"```json\n{json.dumps(FAKE_FACTS)}\n```"
    result = _parse_facts(raw)
    assert result == FAKE_FACTS


def test_parse_facts_invalid_json_raises():
    with pytest.raises(ValueError, match="JSON valide"):
        _parse_facts("ceci n'est pas du JSON")


def test_parse_facts_not_a_list_raises():
    with pytest.raises(ValueError, match="tableau JSON"):
        _parse_facts('{"fact_code": "ABC"}')


# ---------------------------------------------------------------------------
# Test d'intégration : extract() sur le fichier PNG
# ---------------------------------------------------------------------------

def test_extract_png_returns_expected_facts(approach):
    """extract() sur le PNG renvoie les faits mockés correctement parsés."""
    if not PNG_PATH.exists():
        pytest.skip(f"Fichier PNG absent : {PNG_PATH}")

    approach.client.chat.complete.return_value = _make_fake_response(FAKE_FACTS)

    result = approach.extract(str(PNG_PATH))

    # L'appel API a bien été déclenché
    approach.client.chat.complete.assert_called_once()

    # Le message envoyé contient bien une image_url (format vision)
    call_args = approach.client.chat.complete.call_args
    messages = call_args.kwargs["messages"]
    user_message = next(m for m in messages if m["role"] == "user")
    content_types = [block["type"] for block in user_message["content"]]
    assert "image_url" in content_types

    # Les faits retournés correspondent à la réponse mockée
    assert result.facts == FAKE_FACTS
    assert result.input_tokens == 500
    assert result.output_tokens == 42
    assert result.total_tokens == 542
    assert result.metadata["model"] == "mistral-large-latest"


def test_extract_png_empty_facts(approach):
    """extract() gère correctement une réponse vide (aucun fait trouvé)."""
    if not PNG_PATH.exists():
        pytest.skip(f"Fichier PNG absent : {PNG_PATH}")

    approach.client.chat.complete.return_value = _make_fake_response([])

    result = approach.extract(str(PNG_PATH))

    assert result.facts == []
    assert result.total_tokens == 542
