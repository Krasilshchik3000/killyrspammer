"""Тесты для database.py — работа с БД, промпты, training examples."""
import os
import sys
import pytest
from datetime import datetime, timedelta

# Используем временную SQLite для тестов
os.environ["DATABASE_URL"] = ""
os.environ["DATABASE_PATH"] = ":memory:"
os.environ["BOT_TOKEN"] = "test"
os.environ["OPENAI_API_KEY"] = "test"
os.environ["ADMIN_ID"] = "123456"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db
from database import DEFAULT_PROMPT


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Свежая БД для каждого теста."""
    test_db = str(tmp_path / "test.db")
    # Monkey-patch DATABASE_PATH
    import config
    original_path = config.DATABASE_PATH
    original_url = config.DATABASE_URL
    config.DATABASE_PATH = test_db
    config.DATABASE_URL = ""
    # Перезагружаем database модуль чтобы подхватил
    db.DATABASE_PATH = test_db
    db.DATABASE_URL = ""
    db.init_database()
    yield
    config.DATABASE_PATH = original_path
    config.DATABASE_URL = original_url


class TestInitDatabase:
    def test_creates_tables(self):
        """init_database создаёт все необходимые таблицы."""
        # Проверяем что таблицы существуют делая запросы
        db.execute_query("SELECT COUNT(*) FROM messages", fetch='one')
        db.execute_query("SELECT COUNT(*) FROM training_examples", fetch='one')
        db.execute_query("SELECT COUNT(*) FROM prompt_versions", fetch='one')
        db.execute_query("SELECT COUNT(*) FROM bot_state", fetch='one')

    def test_inserts_default_prompt(self):
        """При пустой БД вставляется дефолтный промпт."""
        prompt = db.get_current_prompt()
        assert "{message_text}" in prompt
        assert "СПАМ" in prompt
        assert "НЕ_СПАМ" in prompt
        assert "ВОЗМОЖНО_СПАМ" in prompt


class TestPromptVersioning:
    def test_save_and_get_prompt(self):
        """Сохранение и получение промпта."""
        new_prompt = "Тестовый промпт {message_text} СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ"
        db.save_prompt_version(new_prompt, "тестовое сохранение")
        assert db.get_current_prompt() == new_prompt

    def test_prompt_history(self):
        """История промптов возвращает версии в обратном порядке."""
        db.save_prompt_version("prompt_v2 {message_text}", "версия 2")
        db.save_prompt_version("prompt_v3 {message_text}", "версия 3")
        history = db.get_prompt_history(10)
        assert len(history) >= 3  # дефолт + 2 новых
        assert "версия 3" in history[0][1]

    def test_rollback_prompt(self):
        """Откат к предыдущей версии промпта."""
        original = db.get_current_prompt()
        # Узнаём ID первой версии
        history = db.get_prompt_history(10)
        first_id = history[-1][0]

        db.save_prompt_version("новый промпт {message_text}", "замена")
        assert db.get_current_prompt() == "новый промпт {message_text}"

        result = db.rollback_prompt(first_id)
        assert result is True
        assert db.get_current_prompt() == original

    def test_rollback_nonexistent_version(self):
        """Откат к несуществующей версии возвращает False."""
        assert db.rollback_prompt(99999) is False


class TestTrainingExamples:
    def test_add_and_get_examples(self):
        """Добавление и получение обучающих примеров."""
        db.add_training_example("спам текст", True, "test")
        db.add_training_example("нормальный текст", False, "test")

        examples = db.get_few_shot_examples(10)
        assert len(examples) == 2
        # Последний добавленный — первый в списке (ORDER BY id DESC)
        assert examples[0][0] == "нормальный текст"
        assert examples[0][1] == 0  # SQLite: False = 0

    def test_few_shot_limit(self):
        """Лимит на количество примеров работает."""
        for i in range(20):
            db.add_training_example(f"example_{i}", i % 2 == 0, "test")

        examples = db.get_few_shot_examples(5)
        assert len(examples) == 5


class TestMessages:
    def test_save_and_get_message(self):
        """Сохранение и получение сообщения."""
        db.save_message(100, -1001, 42, "testuser", "hello", "НЕ_СПАМ")
        row = db.get_message_by_id(100)
        assert row is not None
        assert row[0] == "hello"  # text
        assert row[1] == "НЕ_СПАМ"  # llm_result
        assert row[2] == 42  # user_id
        assert row[3] == -1001  # chat_id

    def test_update_admin_decision(self):
        """Обновление решения админа."""
        db.save_message(200, -1001, 42, "testuser", "spam?", "ВОЗМОЖНО_СПАМ")
        db.update_admin_decision(200, "СПАМ")
        # Проверяем через прямой запрос
        row = db.execute_query(
            "SELECT admin_decision FROM messages WHERE message_id = ?",
            (200,), fetch='one'
        )
        assert row[0] == "СПАМ"

    def test_get_user_messages(self):
        """Получение всех сообщений пользователя."""
        db.save_message(301, -1001, 99, "spammer", "msg1", "СПАМ")
        db.save_message(302, -1002, 99, "spammer", "msg2", "СПАМ")
        db.save_message(303, -1001, 100, "normal", "msg3", "НЕ_СПАМ")

        msgs = db.get_user_messages(99)
        assert len(msgs) == 2
        assert all(m[0] in (301, 302) for m in msgs)

    def test_count_user_messages(self):
        """Подсчёт сообщений пользователя в чате."""
        db.save_message(401, -1001, 50, "user", "a", None)
        db.save_message(402, -1001, 50, "user", "b", None)
        db.save_message(403, -1002, 50, "user", "c", None)

        assert db.count_user_messages(50, -1001) == 2
        assert db.count_user_messages(50, -1002) == 1
        assert db.count_user_messages(50, -9999) == 0

    def test_get_recent_mistakes(self):
        """Получение недавних ошибок бота."""
        db.save_message(501, -1001, 60, "user", "missed spam", "НЕ_СПАМ")
        db.update_admin_decision(501, "СПАМ")

        db.save_message(502, -1001, 61, "user", "false positive", "СПАМ")
        db.update_admin_decision(502, "НЕ_СПАМ")

        db.save_message(503, -1001, 62, "user", "correct", "СПАМ")
        db.update_admin_decision(503, "СПАМ")  # Не ошибка

        mistakes = db.get_recent_mistakes(10)
        assert len(mistakes) == 2

    def test_get_stats(self):
        """Статистика корректно считает."""
        db.save_message(601, -1001, 70, "u", "a", "СПАМ")
        db.save_message(602, -1001, 71, "u", "b", "НЕ_СПАМ")
        db.save_message(603, -1001, 72, "u", "c", "ВОЗМОЖНО_СПАМ")
        db.update_admin_decision(601, "СПАМ")
        db.add_training_example("x", True, "test")

        total, spam, maybe, reviewed, training = db.get_stats()
        assert total == 3
        assert spam == 1
        assert maybe == 1
        assert reviewed == 1
        assert training == 1


class TestBotState:
    def test_set_and_get_state(self):
        """Сохранение и получение состояния бота."""
        db.set_bot_state(123, awaiting_prompt_edit=True, pending_prompt="test prompt")
        awaiting, pending = db.get_bot_state(123)
        assert awaiting == 1  # SQLite: True = 1
        assert pending == "test prompt"

    def test_reset_state(self):
        """Сброс состояния."""
        db.set_bot_state(123, awaiting_prompt_edit=True)
        db.set_bot_state(123)  # reset
        awaiting, pending = db.get_bot_state(123)
        assert not awaiting
        assert pending is None

    def test_default_state(self):
        """Состояние по умолчанию для нового админа."""
        awaiting, pending = db.get_bot_state(999999)
        assert awaiting is False
        assert pending is None
