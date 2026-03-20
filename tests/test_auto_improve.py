"""Тесты для системы автоматического улучшения промпта."""
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

os.environ["DATABASE_URL"] = ""
os.environ["DATABASE_PATH"] = ":memory:"
os.environ["BOT_TOKEN"] = "test"
os.environ["OPENAI_API_KEY"] = "test"
os.environ["ADMIN_ID"] = "123456"
os.environ["AUTO_IMPROVE_AFTER_ERRORS"] = "3"
os.environ["MIN_VALIDATION_EXAMPLES"] = "5"
os.environ["MAX_VALIDATION_EXAMPLES"] = "30"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    test_db = str(tmp_path / "test.db")
    import config
    original_path = config.DATABASE_PATH
    config.DATABASE_PATH = test_db
    config.DATABASE_URL = ""
    db.DATABASE_PATH = test_db
    db.DATABASE_URL = ""
    db.init_database()
    yield
    config.DATABASE_PATH = original_path


class TestValidationExamples:
    def test_balanced_examples(self):
        """get_validation_examples возвращает сбалансированную выборку."""
        for i in range(10):
            db.add_training_example(f"spam_{i}", True, "test")
        for i in range(10):
            db.add_training_example(f"ham_{i}", False, "test")

        examples = db.get_validation_examples(10)
        spam_count = sum(1 for _, is_spam in examples if is_spam)
        ham_count = sum(1 for _, is_spam in examples if not is_spam)
        assert spam_count == 5
        assert ham_count == 5

    def test_empty_examples(self):
        examples = db.get_validation_examples(10)
        assert examples == []


class TestCountErrors:
    def test_counts_errors_since_last_improvement(self):
        """Считает только ошибки после последнего обновления промпта."""
        # Сохраняем начальный промпт (уже есть от init_database)

        # Ошибка: бот сказал НЕ_СПАМ, админ сказал СПАМ
        db.save_message(1, -1001, 10, "u", "spam msg", "НЕ_СПАМ")
        db.update_admin_decision(1, "СПАМ")

        db.save_message(2, -1001, 11, "u", "spam msg2", "НЕ_СПАМ")
        db.update_admin_decision(2, "СПАМ")

        # Не ошибка: бот правильно определил
        db.save_message(3, -1001, 12, "u", "normal", "НЕ_СПАМ")
        db.update_admin_decision(3, "НЕ_СПАМ")

        assert db.count_errors_since_last_improvement() == 2

    def test_resets_after_new_prompt(self):
        """После сохранения нового промпта счётчик ошибок сбрасывается."""
        db.save_message(1, -1001, 10, "u", "spam", "НЕ_СПАМ")
        db.update_admin_decision(1, "СПАМ")
        assert db.count_errors_since_last_improvement() >= 1

        # Сохраняем новую версию промпта
        import time
        time.sleep(0.1)  # Чтобы timestamp отличался
        db.save_prompt_version("new prompt {message_text} СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ", "test")

        # Новых ошибок после промпта нет
        assert db.count_errors_since_last_improvement() == 0


class TestCountTrainingExamples:
    def test_counts(self):
        assert db.count_training_examples() == 0
        db.add_training_example("a", True, "test")
        db.add_training_example("b", False, "test")
        assert db.count_training_examples() == 2


@pytest.mark.asyncio
class TestEvaluatePrompt:
    async def test_perfect_accuracy(self):
        """Промпт с идеальной точностью на простых примерах."""
        from main import evaluate_prompt, SpamResult

        # Мокаем classify_message чтобы всегда возвращал правильный ответ
        async def mock_classify(prompt, text, few_shot="", umc=0, cas=False):
            if "spam" in text:
                return SpamResult.SPAM
            return SpamResult.NOT_SPAM

        with patch('main.classify_message', side_effect=mock_classify):
            examples = [("spam_1", True), ("spam_2", True), ("ham_1", False), ("ham_2", False)]
            accuracy, correct, total = await evaluate_prompt("test prompt", examples)
            assert accuracy == 1.0
            assert correct == 4
            assert total == 4

    async def test_partial_accuracy(self):
        """Промпт с частичной точностью."""
        from main import evaluate_prompt, SpamResult

        call_count = 0

        async def mock_classify(prompt, text, few_shot="", umc=0, cas=False):
            nonlocal call_count
            call_count += 1
            # Первые два правильно, остальные нет
            if call_count <= 2:
                return SpamResult.SPAM if "spam" in text else SpamResult.NOT_SPAM
            return SpamResult.NOT_SPAM  # Всегда НЕ_СПАМ — ошибка на спаме

        with patch('main.classify_message', side_effect=mock_classify):
            examples = [("spam_1", True), ("ham_1", False), ("spam_2", True), ("spam_3", True)]
            accuracy, correct, total = await evaluate_prompt("test", examples)
            assert accuracy == 0.5
            assert correct == 2

    async def test_empty_examples(self):
        from main import evaluate_prompt
        accuracy, correct, total = await evaluate_prompt("test", [])
        assert accuracy == 0.0
        assert total == 0


@pytest.mark.asyncio
class TestMaybeTriggerImprovement:
    async def test_does_not_trigger_below_threshold(self):
        """Не запускает улучшение если ошибок меньше порога."""
        from main import maybe_trigger_improvement

        with patch('main.db') as mock_db, \
             patch('main.auto_improve_prompt') as mock_improve:
            mock_db.count_errors_since_last_improvement.return_value = 1
            await maybe_trigger_improvement("missed_spam", "test")
            mock_improve.assert_not_called()

    async def test_triggers_at_threshold(self):
        """Запускает улучшение когда ошибок >= порога."""
        from main import maybe_trigger_improvement

        with patch('main.db') as mock_db, \
             patch('main.asyncio') as mock_asyncio:
            mock_db.count_errors_since_last_improvement.return_value = 3
            await maybe_trigger_improvement("missed_spam", "test")
            mock_asyncio.create_task.assert_called_once()


@pytest.mark.asyncio
class TestAutoImprovePrompt:
    async def test_applies_better_prompt(self):
        """Применяет промпт если он лучше текущего."""
        from main import auto_improve_prompt
        import main

        main._improvement_in_progress = False

        with patch.object(main, 'generate_improved_prompt', return_value=("Анализ", "improved {message_text} СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ")), \
             patch.object(main, 'evaluate_prompt', side_effect=[
                 (0.7, 7, 10),  # текущий: 70%
                 (0.9, 9, 10),  # новый: 90%
             ]), \
             patch.object(main, 'bot') as mock_bot, \
             patch.object(main, 'db') as mock_db:
            mock_db.count_training_examples.return_value = 10
            mock_db.get_validation_examples.return_value = [("x", True)] * 10
            mock_db.get_current_prompt.return_value = "old prompt"
            mock_db.validate_prompt = main.validate_prompt
            mock_bot.send_message = AsyncMock()

            await auto_improve_prompt("missed_spam", "test msg")

            mock_db.save_prompt_version.assert_called_once()
            assert "Авто" in mock_db.save_prompt_version.call_args[0][1]

    async def test_rejects_worse_prompt(self):
        """Не применяет промпт если он хуже текущего."""
        from main import auto_improve_prompt
        import main

        main._improvement_in_progress = False

        with patch.object(main, 'generate_improved_prompt', return_value=("Анализ", "worse {message_text} СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ")), \
             patch.object(main, 'evaluate_prompt', side_effect=[
                 (0.9, 9, 10),  # текущий: 90%
                 (0.6, 6, 10),  # новый: 60% — хуже!
             ]), \
             patch.object(main, 'bot') as mock_bot, \
             patch.object(main, 'db') as mock_db:
            mock_db.count_training_examples.return_value = 10
            mock_db.get_validation_examples.return_value = [("x", True)] * 10
            mock_db.get_current_prompt.return_value = "good prompt"
            mock_bot.send_message = AsyncMock()

            await auto_improve_prompt("missed_spam", "test msg")

            mock_db.save_prompt_version.assert_not_called()
