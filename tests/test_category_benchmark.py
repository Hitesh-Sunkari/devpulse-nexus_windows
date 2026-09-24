import unittest

from app.category_benchmark import CATEGORIES, _aggregate_category, benchmark_tasks


class CategoryBenchmarkTests(unittest.TestCase):
    def test_dataset_contains_all_seven_categories(self):
        tasks = benchmark_tasks()
        self.assertEqual({task["category"] for task in tasks}, set(CATEGORIES))
        self.assertEqual(len(tasks), 30)

    def test_category_winner_uses_quality_not_global_accuracy(self):
        rows = [
            {"model": "quality", "error": None, "answer": "ok", "correctness_percent": 90, "coverage_percent": 90, "relevance_percent": 90, "grounding_percent": 90, "hallucination_rate_percent": 0, "retrieval_support_percent": 90, "output_guard_pass": True, "static_code_validation_pass": None, "latency_seconds": 10, "response_tokens": 20, "ollama_memory_mib": 100},
            {"model": "fast", "error": None, "answer": "ok", "correctness_percent": 40, "coverage_percent": 40, "relevance_percent": 40, "grounding_percent": 40, "hallucination_rate_percent": 0, "retrieval_support_percent": 40, "output_guard_pass": True, "static_code_validation_pass": None, "latency_seconds": 1, "response_tokens": 10, "ollama_memory_mib": 100},
        ]
        result = _aggregate_category("Explanation", rows, ["quality", "fast"])
        self.assertEqual(result["winner"], "quality")
