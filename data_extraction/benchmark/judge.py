import re
from api.base import BaseAPI

JUDGE_PROMPT = """Tu es un évaluateur expert. Compare la réponse obtenue à la réponse attendue.
Attribue un score de 1 à 5 selon ces critères :
1 - Réponse incorrecte ou hors sujet
2 - Réponse partiellement correcte avec erreurs importantes
3 - Réponse correcte mais incomplète
4 - Réponse correcte et complète
5 - Réponse parfaite, précise et bien sourcée

Réponds UNIQUEMENT avec le chiffre (1, 2, 3, 4 ou 5), rien d'autre."""


class Judge:
    def __init__(self, api: BaseAPI):
        self.api = api

    def score(self, question: str, expected: str, actual: str) -> int:
        messages = [{
            "role": "user",
            "content": (
                f"Question : {question}\n\n"
                f"Réponse attendue : {expected}\n\n"
                f"Réponse obtenue : {actual}"
            ),
        }]
        response = self.api.send(messages, system=JUDGE_PROMPT)
        match = re.search(r"[1-5]", response.content)
        return int(match.group()) if match else 0
