import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from facts_extraction.approaches.mistral_simple import MistralSimpleApproach

QUESTIONS_PATH = Path(__file__).parent.parent / "facts_extraction" / "benchmark" / "questions.json"
DATA_ROOT = Path(__file__).parent.parent


def _get_approach():
    try:
        return MistralSimpleApproach()
    except EnvironmentError as e:
        st.error(str(e))
        return None


def _compute_metrics(predicted: list[dict], expected: list[dict]) -> dict:
    """Comparaison exacte sur le triplet (fact_code, fact_value, fact_unit)."""
    pred_set = {(f["fact_code"], str(f["fact_value"]), f["fact_unit"]) for f in predicted}
    exp_set = {(f["fact_code"], str(f["fact_value"]), f["fact_unit"]) for f in expected}

    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(exp_set) if exp_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "n_predicted": len(pred_set),
        "n_expected": len(exp_set),
    }


def _render_benchmark():
    st.markdown("Lance les prédictions sur tous les documents du benchmark et compare avec les faits attendus.")

    if not QUESTIONS_PATH.exists():
        st.warning("Aucun fichier `benchmark/questions.json` trouvé.")
        return

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    documents = questions.get("documents", [])
    st.caption(f"{len(documents)} document(s) dans le benchmark")

    if not st.button("Lancer le benchmark", type="primary", key="facts_run_benchmark"):
        return

    approach = _get_approach()
    if not approach:
        return

    all_precision, all_recall, all_f1 = [], [], []

    for doc in documents:
        file_path = DATA_ROOT / doc["path"]
        st.markdown(f"#### {doc['id']}")

        if not file_path.exists():
            st.error(f"Fichier introuvable : {doc['path']}")
            continue

        with st.spinner("Extraction en cours..."):
            result = approach.extract(str(file_path))

        metrics = _compute_metrics(result.facts, doc["expected_facts"])
        all_precision.append(metrics["precision"])
        all_recall.append(metrics["recall"])
        all_f1.append(metrics["f1"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Précision", f"{metrics['precision']:.0%}")
        col2.metric("Rappel", f"{metrics['recall']:.0%}")
        col3.metric("F1", f"{metrics['f1']:.0%}")

        col_pred, col_exp = st.columns(2)
        with col_pred:
            st.caption(f"Faits extraits ({metrics['n_predicted']})")
            st.json(result.facts)
        with col_exp:
            st.caption(f"Faits attendus ({metrics['n_expected']})")
            st.json(doc["expected_facts"])

        st.caption(
            f"Tokens : {result.total_tokens} | "
            f"Latence : {result.latency_ms:.0f} ms | "
            f"Modèle : {result.metadata['model']}"
        )
        st.divider()

    if len(all_precision) > 1:
        st.markdown("### Résultats globaux")
        col1, col2, col3 = st.columns(3)
        col1.metric("Précision moyenne", f"{sum(all_precision) / len(all_precision):.0%}")
        col2.metric("Rappel moyen", f"{sum(all_recall) / len(all_recall):.0%}")
        col3.metric("F1 moyen", f"{sum(all_f1) / len(all_f1):.0%}")


def _render_run():
    st.markdown("Charge un document et lance l'extraction en direct.")

    uploaded = st.file_uploader("Choisir un document", type=["pdf", "png", "jpg", "jpeg"])

    if not uploaded:
        return

    if not st.button("Extraire les faits", type="primary"):
        return

    approach = _get_approach()
    if not approach:
        return

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Extraction en cours..."):
            result = approach.extract(tmp_path)
    finally:
        os.unlink(tmp_path)

    st.success(f"{len(result.facts)} fait(s) extrait(s)")
    st.json(result.facts)
    st.caption(
        f"Tokens : {result.total_tokens} | "
        f"Latence : {result.latency_ms:.0f} ms | "
        f"Modèle : {result.metadata['model']}"
    )


def render(api):
    st.subheader("Extraction de faits")

    tab_run, tab_benchmark = st.tabs(["Run", "Benchmark"])

    with tab_benchmark:
        _render_benchmark()

    with tab_run:
        _render_run()
