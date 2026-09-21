import unittest

from app.evaluator import evaluate_response
from app.question_analysis import analyse_question, split_question


QUESTION = "is docker responsible for the memory usage and what is digital twin()"


def evidence_context():
    return {
        "question_parts": analyse_question(QUESTION),
        "rag": [
            "Docker may contribute to WSL memory pressure but telemetry does not prove sole causation.",
            "The Digital Twin combines host telemetry and Docker telemetry.",
        ],
        "sourcegraph": [
            {
                "path": "app/digital_twin.py",
                "line": 37,
                "preview": "def get_digital_twin():",
                "code": "38: def get_digital_twin():\n39:     return snapshot",
            },
        ],
        "telemetry": {"system": {"memory": {"usage_percent": 26.2}}},
        "diagnosis": {"assessment": "potentially_significant"},
    }


class QuestionAnalysisTests(unittest.TestCase):
    def test_explicit_compound_question_is_split(self):
        self.assertEqual(len(split_question(QUESTION)), 2)


class LiveEvaluationTests(unittest.TestCase):
    def test_answer_missing_a_request_is_not_decision_eligible(self):
        result = evaluate_response(
            QUESTION,
            "Explanation",
            {"answer": "Docker may contribute to memory usage, but it is not the sole cause.", "error": None},
            evidence_context(),
        )
        self.assertFalse(result["decision_eligible"])
        self.assertLessEqual(result["score"], 59.0)

    def test_complete_evidence_aligned_answer_is_eligible(self):
        result = evaluate_response(
            QUESTION,
            "Explanation",
            {
                "answer": (
                    "1. Docker may contribute to observed memory usage, but the diagnosis does not prove sole responsibility. "
                    "2. The Digital Twin combines live host and Docker telemetry [app/digital_twin.py:38]."
                ),
                "error": None,
            },
            evidence_context(),
        )
        self.assertTrue(result["decision_eligible"])
        self.assertGreater(result["score"], 59.0)

    def test_naming_digital_twin_without_explaining_it_is_incomplete(self):
        result = evaluate_response(
            QUESTION,
            "Explanation",
            {
                "answer": "1. Docker may contribute to memory use. 2. Digital Twin is",
                "error": None,
            },
            evidence_context(),
        )
        self.assertFalse(result["decision_eligible"])
        self.assertIn("Answer appears to end mid-sentence", result["evaluation_warnings"])

    def test_repeated_numbered_questions_do_not_count_as_answers(self):
        result = evaluate_response(
            QUESTION,
            "Explanation",
            {
                "answer": "2. What is Digital Twin?\n3. What is Docker Container Count?\n4. Digital Twin Telemetry",
                "error": None,
            },
            evidence_context(),
        )
        self.assertFalse(result["completion"]["parts"][1]["addressed"])
        self.assertFalse(result["decision_eligible"])

    def test_nearby_docker_topic_does_not_count_as_memory_answer(self):
        result = evaluate_response(
            "is docker responsible for memory usage",
            "Explanation",
            {
                "answer": (
                    "Docker can be responsible for network traffic because "
                    "containers exchange data through its networking layer."
                ),
                "error": None,
            },
            {
                **evidence_context(),
                "question_parts": analyse_question(
                    "is docker responsible for memory usage"
                ),
            },
        )
        self.assertFalse(result["decision_eligible"])
        self.assertIn("memory", result["completion"]["parts"][0]["subject_terms"])


if __name__ == "__main__":
    unittest.main()
