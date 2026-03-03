import unittest

from src.sciq_baselines import (
    extract_letter,
    format_prompt,
    format_rag_prompt,
    make_mcq,
)


class TestSciQBaselines(unittest.TestCase):
    def setUp(self):
        self.example = {
            "question": "What is H2O?",
            "correct_answer": "Water",
            "distractor1": "Stone",
            "distractor2": "Fire",
            "distractor3": "Air",
        }

    def test_make_mcq_is_deterministic_per_seed(self):
        q1, c1, a1 = make_mcq(self.example, seed=7)
        q2, c2, a2 = make_mcq(self.example, seed=7)
        self.assertEqual((q1, c1, a1), (q2, c2, a2))

    def test_extract_letter_standalone_only(self):
        self.assertEqual(extract_letter("Answer: C"), "C")
        self.assertEqual(extract_letter("I think the answer is b."), "B")
        self.assertIsNone(extract_letter("because maybe"))

    def test_format_prompt_validates_choice_count(self):
        with self.assertRaises(ValueError):
            format_prompt("Q", ["A", "B", "C"])

    def test_format_rag_prompt_contains_context(self):
        p = format_rag_prompt(["ctx1", "ctx2"], "Q", ["1", "2", "3", "4"])
        self.assertIn("[Context 1]", p)
        self.assertIn("ctx2", p)


if __name__ == "__main__":
    unittest.main()
