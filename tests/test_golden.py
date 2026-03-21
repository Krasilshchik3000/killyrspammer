"""
Регрессионный тест на золотом dataset.

Вызывает РЕАЛЬНЫЙ OpenAI API. Запускается отдельно:
    python -m pytest tests/test_golden.py -v --tb=long

Требует:
    OPENAI_API_KEY в окружении или .env

Пороги:
    - NOT_SPAM false positive rate < 5% (не более 5% легитимных помечены как спам)
    - SPAM detection rate > 80% (не менее 80% спама пойман)
    - Prompt injection: 100% пойман (ни один injection не должен пройти)
"""
import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Skip if no API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SKIP_REASON = "OPENAI_API_KEY not set" if not OPENAI_API_KEY else None

from tests.golden_dataset import (
    NOT_SPAM_CASUAL, NOT_SPAM_MONEY, NOT_SPAM_LINKS, NOT_SPAM_EDGE,
    SPAM_CLASSIC, SPAM_PROMOTION, SPAM_ADVERSARIAL, SPAM_INJECTION,
    MAYBE_SPAM, GOLDEN_DATASET,
)


async def classify_with_api(message_text: str) -> str:
    """Классифицирует сообщение через реальный API с текущим промптом."""
    from openai import AsyncOpenAI
    from database import DEFAULT_PROMPT
    from text_normalize import normalize_text
    from config import LLM_MODEL

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    normalized = normalize_text(message_text)

    system_prompt = DEFAULT_PROMPT.replace("{few_shot_block}", "")

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "spam_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "enum": ["SPAM", "NOT_SPAM", "MAYBE_SPAM"]
                    }
                },
                "required": ["result"],
                "additionalProperties": False,
            }
        }
    }

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<message>\n{normalized}\n</message>\n\nClassify the message above. Respond with JSON."},
        ],
        response_format=schema,
        max_completion_tokens=20,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)
    return parsed["result"]


# Map golden dataset labels to API labels
LABEL_MAP = {
    "НЕ_СПАМ": "NOT_SPAM",
    "СПАМ": "SPAM",
    "ВОЗМОЖНО_СПАМ": "MAYBE_SPAM",
}


def is_correct(predicted: str, expected_label: str) -> bool:
    """Проверяет правильность классификации с учётом допущений."""
    expected = LABEL_MAP.get(expected_label, expected_label)

    if predicted == expected:
        return True

    # MAYBE_SPAM вместо SPAM — приемлемо (поймает через ревью)
    if expected == "SPAM" and predicted == "MAYBE_SPAM":
        return True

    # MAYBE_SPAM вместо NOT_SPAM — допустимо но нежелательно (false positive)
    # Считаем это ошибкой для строгого теста
    return False


@unittest.skipIf(SKIP_REASON, SKIP_REASON)
class TestGoldenDatasetRegression(unittest.TestCase):
    """Регрессионный тест на золотом dataset."""

    @classmethod
    def setUpClass(cls):
        """Классифицируем весь dataset один раз."""
        async def run_all():
            results = {}
            for text, label, category, desc in GOLDEN_DATASET:
                try:
                    predicted = await classify_with_api(text)
                    results[(text, label)] = predicted
                except Exception as e:
                    results[(text, label)] = f"ERROR: {e}"
            return results

        cls.results = asyncio.run(run_all())

    def test_not_spam_false_positive_rate(self):
        """Не более 5% легитимных сообщений помечены как SPAM."""
        not_spam = [r for (t, l), r in self.results.items() if l == "НЕ_СПАМ"]
        false_positives = [r for r in not_spam if r == "SPAM"]
        rate = len(false_positives) / len(not_spam) if not_spam else 0
        self.assertLessEqual(rate, 0.05,
            f"False positive rate {rate:.1%} ({len(false_positives)}/{len(not_spam)}) exceeds 5% threshold. "
            f"FP messages: {[t for (t, l), r in self.results.items() if l == 'НЕ_СПАМ' and r == 'SPAM']}")

    def test_spam_detection_rate(self):
        """Не менее 80% спама пойман (SPAM или MAYBE_SPAM)."""
        spam = [(t, r) for (t, l), r in self.results.items() if l == "СПАМ"]
        caught = [(t, r) for t, r in spam if r in ("SPAM", "MAYBE_SPAM")]
        rate = len(caught) / len(spam) if spam else 0
        self.assertGreaterEqual(rate, 0.80,
            f"Spam detection rate {rate:.1%} ({len(caught)}/{len(spam)}) below 80% threshold. "
            f"Missed: {[t for t, r in spam if r == 'NOT_SPAM']}")

    def test_prompt_injection_caught(self):
        """100% prompt injection попыток пойман."""
        injections = [(t, l) for t, l, c, d in SPAM_INJECTION]
        for text, label in injections:
            predicted = self.results.get((text, label), "MISSING")
            self.assertIn(predicted, ("SPAM", "MAYBE_SPAM"),
                f"Prompt injection NOT caught: '{text[:60]}...' → {predicted}")

    def test_overall_accuracy(self):
        """Общая точность > 75%."""
        correct = sum(1 for (t, l), r in self.results.items()
                      if is_correct(r, l))
        total = len(self.results)
        accuracy = correct / total if total else 0
        self.assertGreaterEqual(accuracy, 0.75,
            f"Overall accuracy {accuracy:.1%} ({correct}/{total}) below 75%")

    def test_money_discussion_not_flagged(self):
        """Обсуждение денег НЕ помечается как SPAM."""
        money = [(t, l) for t, l, c, d in NOT_SPAM_MONEY]
        for text, label in money:
            predicted = self.results.get((text, label), "MISSING")
            self.assertNotEqual(predicted, "SPAM",
                f"Money discussion flagged as SPAM: '{text[:60]}...'")

    def test_print_full_report(self):
        """Печатает полный отчёт (всегда проходит)."""
        errors = []
        for (text, label), predicted in self.results.items():
            expected = LABEL_MAP.get(label, label)
            correct = is_correct(predicted, label)
            if not correct:
                errors.append(f"  ❌ [{predicted:10s}] expected [{expected:10s}] | {text[:60]}")

        total = len(self.results)
        correct_count = total - len(errors)
        print(f"\n{'='*60}")
        print(f"Golden Dataset: {correct_count}/{total} correct ({correct_count/total:.0%})")
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for e in errors:
                print(e)
        print(f"{'='*60}")


if __name__ == "__main__":
    unittest.main()
