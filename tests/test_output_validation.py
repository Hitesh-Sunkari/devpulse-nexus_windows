import unittest

from app.output_validation import validate_output


CONTEXT = {
    "question_parts": [{"question": "Which file retrieves knowledge from ChromaDB?"}],
    "rag": ["The rag.py module retrieves knowledge from the ChromaDB collection."],
    "sourcegraph": [{"path": "app/rag.py", "preview": "collection.query", "code": "def retrieve_evidence(question):\n    return collection.query(...)"}],
    "diagnosis": {},
}


class OutputValidationTests(unittest.TestCase):
    def test_supported_answer_is_accepted(self):
        result = validate_output(
            "Which file retrieves knowledge from ChromaDB?",
            "app/rag.py retrieves knowledge from the ChromaDB collection.",
            "Code Retrieval",
            CONTEXT,
            mode="fast",
        )
        self.assertTrue(result["accepted"])

    def test_prompt_echo_is_rejected(self):
        result = validate_output(
            "Which file retrieves knowledge from ChromaDB?",
            "LIVE FACTS: no answer.",
            "Code Retrieval",
            CONTEXT,
            mode="fast",
        )
        self.assertFalse(result["accepted"])
        self.assertIn("prompt_echo", result["failures"])

    def test_secret_leak_is_rejected(self):
        fake_credential = "github_" + "pat_" + "A" * 24
        result = validate_output(
            "Which file retrieves knowledge from ChromaDB?",
            "The token is " + fake_credential + ".",
            "Code Retrieval",
            CONTEXT,
        )
        self.assertFalse(result["accepted"])
        self.assertIn("credential_leak", result["failures"])

    def test_unsupported_answer_is_rejected(self):
        result = validate_output(
            "Which file retrieves knowledge from ChromaDB?",
            "Weather is sunny today.",
            "Code Retrieval",
            CONTEXT,
        )
        self.assertFalse(result["accepted"])
        self.assertIn("not_relevant_to_question", result["failures"])
