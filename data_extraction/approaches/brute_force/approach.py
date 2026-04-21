from api.base import BaseAPI
from data_extraction.approaches.base import BaseApproach, ApproachResponse
from data_extraction.approaches.utils import load_pdf_texts

SYSTEM_PROMPT = """Tu es un expert en plongée sous-marine.
Réponds uniquement en français, de manière précise et sourcée, en te basant exclusivement sur les documents fournis.
Si la réponse n'est pas dans les documents, dis-le clairement.
Ne fournis jamais de liens externes."""


class BruteForceApproach(BaseApproach):
    name = "brute_force"

    def __init__(self, api: BaseAPI, pdf_dir: str = "data"):
        self.api = api
        docs = load_pdf_texts(pdf_dir)
        self.context = "\n\n".join(f"=== {name} ===\n{text}" for name, text in docs.items())

    def ask(self, question: str) -> ApproachResponse:
        messages = [{
            "role": "user",
            "content": f"{self.context}\n\n---\n\nQuestion : {question}",
        }]
        response = self.api.send(messages, system=SYSTEM_PROMPT)
        return ApproachResponse.from_api_response(response)
