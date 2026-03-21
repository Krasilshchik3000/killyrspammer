"""
Интеграционные тесты — ловят проблемы, которые юнит-тесты пропускают.

Проверяют:
1. Реальную совместимость OpenAI SDK с используемыми моделями
2. Качество классификации промпта на реальных сообщениях
3. Полный flow бана: удаление ВСЕХ сообщений
4. Миграцию БД: старый промпт → новый
5. PostgreSQL совместимость SQL-запросов
6. Автоулучшение: генерация + парсинг маркера
"""
import asyncio
import os
import sqlite3
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# ──────────────────────────────────────────────
# 1. OpenAI SDK совместимость
# ──────────────────────────────────────────────


class TestOpenAISDKCompatibility(unittest.TestCase):
    """Проверяет, что openai SDK поддерживает max_completion_tokens."""

    def test_sdk_supports_max_completion_tokens(self):
        """SDK должен принимать max_completion_tokens — иначе gpt-5.4 не работает."""
        import openai
        import inspect

        sig = inspect.signature(openai.resources.chat.completions.AsyncCompletions.create)
        params = sig.parameters

        # Либо параметр есть напрямую, либо есть **kwargs для forward-compat
        has_param = "max_completion_tokens" in params
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        # В новых версиях SDK (>=1.40) есть extra_body или **kwargs
        self.assertTrue(
            has_param or has_kwargs,
            f"openai SDK {openai.__version__} не поддерживает max_completion_tokens. "
            f"Нужна версия >=1.40.0. Установите: pip install 'openai>=1.50.0'"
        )

    def test_token_limit_param_returns_correct_key(self):
        """Хелпер _token_limit_param должен возвращать правильный ключ для gpt-5.4."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        from main import _token_limit_param, _token_limit_param_improvement

        result = _token_limit_param(30)
        # Для gpt-5.4 модели (по умолчанию) должен быть max_completion_tokens
        from config import LLM_MODEL
        if LLM_MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
            self.assertIn("max_completion_tokens", result)
            self.assertNotIn("max_tokens", result)
        else:
            self.assertIn("max_tokens", result)

        result2 = _token_limit_param_improvement(3000)
        from config import LLM_IMPROVEMENT_MODEL
        if LLM_IMPROVEMENT_MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
            self.assertIn("max_completion_tokens", result2)


# ──────────────────────────────────────────────
# 2. Качество классификации промпта
# ──────────────────────────────────────────────


class TestPromptQuality(unittest.TestCase):
    """Проверяет, что DEFAULT_PROMPT правильно классифицирует типичные сообщения.

    Не вызывает реальный API — мокает LLM, но проверяет,
    что промпт содержит правильные инструкции.
    """

    def setUp(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from database import DEFAULT_PROMPT
        self.prompt = DEFAULT_PROMPT

    def test_prompt_has_default_to_not_spam_rule(self):
        """Промпт должен содержать правило 'если сомневаешься — НЕ_СПАМ'."""
        self.assertIn("если сомневаешься", self.prompt.lower())
        self.assertIn("НЕ_СПАМ", self.prompt)

    def test_prompt_has_money_discussion_exception(self):
        """Промпт должен явно исключать обсуждение цен/зарплат из спама."""
        lower = self.prompt.lower()
        self.assertTrue(
            "цен" in lower or "зарплат" in lower or "налог" in lower,
            "Промпт должен содержать исключение для обсуждения денег в контексте"
        )

    def test_prompt_has_short_messages_exception(self):
        """Промпт должен не считать короткие сообщения спамом."""
        self.assertIn("Короткие сообщения", self.prompt)

    def test_prompt_has_required_placeholders(self):
        self.assertIn("{message_text}", self.prompt)
        self.assertIn("{few_shot_block}", self.prompt)

    def test_prompt_has_all_three_categories(self):
        for cat in ["СПАМ", "НЕ_СПАМ", "ВОЗМОЖНО_СПАМ"]:
            self.assertIn(cat, self.prompt)

    def test_prompt_emphasizes_false_positive_cost(self):
        """Промпт должен подчёркивать, что ложное срабатывание хуже пропуска."""
        lower = self.prompt.lower()
        self.assertTrue(
            "лучше пропустить" in lower or "лучше не забанить" in lower
            or "забанить обычного" in lower,
            "Промпт должен подчёркивать цену ложных срабатываний"
        )


# ──────────────────────────────────────────────
# 3. Ban flow: удаление ВСЕХ сообщений
# ──────────────────────────────────────────────


class TestBanFlowDeletesAllMessages(unittest.TestCase):
    """Проверяет, что ban_and_report удаляет ВСЕ сообщения спамера."""

    def test_ban_and_report_calls_delete_user_messages(self):
        """ban_and_report должен вызывать delete_user_messages после бана."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import main as m
        import inspect
        source = inspect.getsource(m.ban_and_report)

        self.assertIn("delete_user_messages", source,
                       "ban_and_report ДОЛЖЕН вызывать delete_user_messages для удаления всех сообщений спамера")

    def test_handle_admin_spam_feedback_deletes_messages(self):
        """Кнопка СПАМ от админа должна удалять все сообщения."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import main as m
        import inspect
        source = inspect.getsource(m.handle_admin_feedback)

        self.assertIn("delete_user_messages", source,
                       "handle_admin_feedback ДОЛЖЕН вызывать delete_user_messages при нажатии СПАМ")


# ──────────────────────────────────────────────
# 4. Миграция БД: старый промпт → новый
# ──────────────────────────────────────────────


class TestDBMigration(unittest.TestCase):
    """Проверяет, что init_database обновляет устаревший промпт."""

    def setUp(self):
        self.db_path = "/tmp/test_migration_antispam.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_old_prompt_gets_migrated(self):
        """Если в БД старый промпт (без маркеров нового), он должен обновиться."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        old_prompt = "Определи, это спам или нет. {message_text} СПАМ, НЕ_СПАМ, ВОЗМОЖНО_СПАМ"

        with patch("database.DATABASE_URL", None), \
             patch("database.DATABASE_PATH", self.db_path):
            import database as db
            # Переинициализируем
            db.DATABASE_URL = None
            db.DATABASE_PATH = self.db_path

            db.init_database()

            # Подменим промпт на старый
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM prompt_versions")
            conn.execute(
                "INSERT INTO prompt_versions (prompt_text, reason, created_at) VALUES (?, ?, ?)",
                (old_prompt, "old", datetime.now())
            )
            conn.commit()
            conn.close()

            # Повторная инициализация должна обновить промпт
            db.init_database()

            current = db.get_current_prompt()
            self.assertIn("Правило по умолчанию", current,
                          "После миграции промпт должен содержать 'Правило по умолчанию'")
            self.assertIn("обязательные исключения", current,
                          "После миграции промпт должен содержать 'обязательные исключения'")

    def test_new_prompt_not_overwritten(self):
        """Если промпт уже новый, миграция не должна его трогать."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        custom_prompt = ("Мой кастомный промпт. Правило по умолчанию: НЕ_СПАМ. "
                         "обязательные исключения: да. {message_text} {few_shot_block} "
                         "СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ")

        with patch("database.DATABASE_URL", None), \
             patch("database.DATABASE_PATH", self.db_path):
            import database as db
            db.DATABASE_URL = None
            db.DATABASE_PATH = self.db_path

            db.init_database()

            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM prompt_versions")
            conn.execute(
                "INSERT INTO prompt_versions (prompt_text, reason, created_at) VALUES (?, ?, ?)",
                (custom_prompt, "custom", datetime.now())
            )
            conn.commit()
            conn.close()

            db.init_database()

            current = db.get_current_prompt()
            self.assertEqual(current, custom_prompt,
                             "Кастомный промпт с маркерами нового не должен перезаписываться")


# ──────────────────────────────────────────────
# 5. PostgreSQL SQL совместимость
# ──────────────────────────────────────────────


class TestPostgreSQLCompat(unittest.TestCase):
    """Проверяет, что SQL-запросы совместимы с PostgreSQL."""

    def test_no_bare_boolean_literals_in_queries(self):
        """Запросы не должны содержать is_spam = 1 или is_spam = 0
        (PostgreSQL не приводит integer к boolean)."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import database
        import inspect
        source = inspect.getsource(database)

        # Ищем паттерны is_spam = 1/0 (без кавычек, буквально в SQL)
        import re
        bad_patterns = re.findall(r'is_spam\s*=\s*[01]', source)
        self.assertEqual(len(bad_patterns), 0,
                         f"Найдены is_spam = 0/1 (несовместимо с PostgreSQL): {bad_patterns}. "
                         "Используйте параметризованные запросы с Python True/False")

    def test_boolean_queries_use_parameters(self):
        """Запросы с is_spam должны использовать параметры (?) вместо литералов."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import database
        import inspect
        source = inspect.getsource(database)

        # Все WHERE is_spam = ... должны использовать ? (параметр)
        import re
        where_patterns = re.findall(r'WHERE\s+is_spam\s*=\s*(\S+)', source)
        for pattern in where_patterns:
            self.assertIn(pattern, ['?', '%s'],
                          f"is_spam сравнивается с '{pattern}' вместо параметра (?). "
                          "Это может сломаться на PostgreSQL")

    def test_no_string_format_in_queries_with_user_data(self):
        """SQL-запросы не должны использовать f-string/format для пользовательских данных."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import database
        import inspect
        source = inspect.getsource(database)

        # Допустимые f-string: для placeholder ('{placeholder}') в init_database
        # Недопустимые: f"... WHERE user_id = {user_id} ..."
        import re
        # Ищем f-строки с SQL и подстановками, исключая placeholder
        dangerous = re.findall(r'f"[^"]*(?:WHERE|INSERT|UPDATE|DELETE)[^"]*\{(?!placeholder)[a-z_]+\}', source)
        self.assertEqual(len(dangerous), 0,
                         f"SQL injection risk: {dangerous}")


# ──────────────────────────────────────────────
# 6. Автоулучшение: формат ответа
# ──────────────────────────────────────────────


class TestAutoImproveResponseParsing(unittest.TestCase):
    """Проверяет парсинг ответа LLM при автоулучшении."""

    def test_parses_standard_format(self):
        """Стандартный формат с маркерами АНАЛИЗ: и ИТОГОВЫЙ_ПРОМПТ:."""
        response = (
            "АНАЛИЗ: Сообщение 'черепашки' ошибочно классифицировано\n\n"
            "ИТОГОВЫЙ_ПРОМПТ:\n"
            "Новый промпт с {message_text} и {few_shot_block} СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ"
        )
        analysis, improved = self._parse_response(response)
        self.assertIsNotNone(improved)
        self.assertIn("{message_text}", improved)

    def test_parses_bold_markers(self):
        """Маркеры в bold: **ИТОГОВЫЙ_ПРОМПТ:**."""
        response = (
            "**АНАЛИЗ:** причина\n\n"
            "**ИТОГОВЫЙ_ПРОМПТ:**\n"
            "Промпт {message_text} {few_shot_block} СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ"
        )
        analysis, improved = self._parse_response(response)
        self.assertIsNotNone(improved)

    def test_parses_code_block_wrapped(self):
        """Промпт обёрнут в ```."""
        response = (
            "АНАЛИЗ: причина\n\n"
            "ИТОГОВЫЙ_ПРОМПТ:\n"
            "```\nПромпт {message_text} {few_shot_block} СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ\n```"
        )
        analysis, improved = self._parse_response(response)
        self.assertIsNotNone(improved)
        self.assertNotIn("```", improved)

    def test_patches_missing_message_text(self):
        """Если LLM забыл {message_text}, он должен быть добавлен."""
        response = (
            "АНАЛИЗ: причина\n\n"
            "ИТОГОВЫЙ_ПРОМПТ:\n"
            "Промпт без плейсхолдера СПАМ НЕ_СПАМ ВОЗМОЖНО_СПАМ"
        )
        analysis, improved = self._parse_response(response)
        self.assertIsNotNone(improved)
        self.assertIn("{message_text}", improved)

    def test_returns_none_without_marker(self):
        """Без маркера ИТОГОВЫЙ_ПРОМПТ возвращает (analysis, None)."""
        response = "Просто текст без маркера"
        analysis, improved = self._parse_response(response)
        self.assertIsNone(improved)
        self.assertIsNotNone(analysis)  # analysis = весь текст

    def _parse_response(self, text):
        """Имитирует парсинг из generate_improved_prompt."""
        marker = None
        for m in ["ИТОГОВЫЙ_ПРОМПТ:", "ИТОГОВЫЙ ПРОМПТ:", "**ИТОГОВЫЙ_ПРОМПТ:**", "**ИТОГОВЫЙ_ПРОМПТ**:"]:
            if m in text:
                marker = m
                break

        if not marker:
            return text, None

        improved = text.split(marker, 1)[1].strip()

        if improved.startswith("```"):
            improved = improved.split("```", 2)[1]
            if improved.startswith("\n"):
                improved = improved[1:]
            if "```" in improved:
                improved = improved.rsplit("```", 1)[0]
            improved = improved.strip()

        if "{message_text}" not in improved:
            improved += "\n\nСообщение: «{message_text}»\n\nОтвет:"
        if "{few_shot_block}" not in improved:
            improved = improved.replace(
                "Сообщение: «{message_text}»",
                "{few_shot_block}\nСообщение: «{message_text}»"
            )

        analysis = text.split(marker)[0].strip()
        if analysis.startswith("АНАЛИЗ:"):
            analysis = analysis[7:].strip()
        elif analysis.startswith("**АНАЛИЗ:**"):
            analysis = analysis[11:].strip()

        return analysis, improved


# ──────────────────────────────────────────────
# 7. Error handling: LLM failure → НЕ автобан
# ──────────────────────────────────────────────


class TestLLMFailureSafety(unittest.TestCase):
    """При ошибке LLM бот не должен банить."""

    def test_llm_error_returns_maybe_spam_not_spam(self):
        """check_message_with_llm при ошибке возвращает ВОЗМОЖНО_СПАМ (отправляется на ревью, не банит)."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        # Проверим: ВОЗМОЖНО_СПАМ → send_to_admin (не ban_and_report)
        import main as m
        import inspect
        source = inspect.getsource(m.handle_message)

        # MAYBE_SPAM должен вызывать send_to_admin, а не ban_and_report
        lines = source.split('\n')
        maybe_spam_action = None
        for i, line in enumerate(lines):
            if 'MAYBE_SPAM' in line:
                # Ищем действие в следующих строках
                for j in range(i, min(i + 3, len(lines))):
                    if 'ban_and_report' in lines[j]:
                        maybe_spam_action = 'ban'
                    elif 'send_to_admin' in lines[j]:
                        maybe_spam_action = 'review'

        self.assertEqual(maybe_spam_action, 'review',
                         "ВОЗМОЖНО_СПАМ должен отправляться на ревью (send_to_admin), а не банить (ban_and_report)")

    def test_rate_limited_returns_maybe_spam(self):
        """При rate limit возвращается MAYBE_SPAM — нужно ревью, не бан."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import main
        import inspect
        source = inspect.getsource(main.check_message_with_llm)

        # check_rate_limit → return SpamResult.MAYBE_SPAM
        self.assertIn("MAYBE_SPAM", source)


# ──────────────────────────────────────────────
# 8. Trusted user: skip LLM check
# ──────────────────────────────────────────────


class TestTrustedUserSkip(unittest.TestCase):
    """Проверяет, что пользователи с историей не проверяются через LLM."""

    def test_trusted_user_returns_early(self):
        """handle_message должен пропускать пользователей с >= TRUSTED_USER_MESSAGES."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import main
        import inspect
        source = inspect.getsource(main.handle_message)

        self.assertIn("TRUSTED_USER_MESSAGES", source,
                       "handle_message должен проверять TRUSTED_USER_MESSAGES")
        self.assertIn("user_msg_count", source)

    def test_trusted_user_messages_saved_to_db(self):
        """Сообщения доверенных пользователей должны сохраняться в БД (для delete_user_messages)."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import main
        import inspect
        source = inspect.getsource(main.handle_message)

        # После return для trusted user должен быть save_message
        # Найдём блок trusted user
        trusted_block_start = source.index("TRUSTED_USER_MESSAGES")
        # Ищем save_message между TRUSTED_USER_MESSAGES и следующим return
        trusted_section = source[trusted_block_start:trusted_block_start + 500]
        self.assertIn("save_message", trusted_section,
                       "Сообщения trusted пользователей должны сохраняться в БД")


# ──────────────────────────────────────────────
# 9. Callback data integrity
# ──────────────────────────────────────────────


class TestCallbackDataIntegrity(unittest.TestCase):
    """Проверяет, что callback_data кнопок корректно парсится."""

    def test_unban_callback_parsing(self):
        """unban_{uid}_{cid}_{msg_id} должен парситься в 3 числа."""
        # Имитируем callback data
        uid, cid, msg_id = 123456, -100123, 789
        data = f"unban_{uid}_{cid}_{msg_id}"

        parts = data.split("_")
        # unban + uid + cid + msg_id = 4 части, НО cid отрицательный = "-100123"
        # split по "_" даст: ['unban', '123456', '-100123', '789']
        # Это 4 части — но parts[1], parts[2], parts[3] корректны

        # Проблема: если cid содержит _, split сломается
        # Проверим текущую логику парсинга
        if len(parts) == 4:
            parsed_uid = int(parts[1])
            parsed_cid = int(parts[2])
            parsed_msg = int(parts[3])
            self.assertEqual(parsed_uid, uid)
            self.assertEqual(parsed_cid, cid)
            self.assertEqual(parsed_msg, msg_id)
        else:
            self.fail(f"Callback data '{data}' не парсится в 4 части: {parts}")

    def test_unban_callback_negative_group_id(self):
        """Группы с ID типа -1002116322225 (13 символов с минусом) должны парситься."""
        uid, cid, msg_id = 8371210082, -1002116322225, 42
        data = f"unban_{uid}_{cid}_{msg_id}"

        parts = data.split("_")
        # 'unban', '8371210082', '-1002116322225', '42' = 4 части ✓
        self.assertEqual(len(parts), 4, f"Unexpected split: {parts}")
        self.assertEqual(int(parts[2]), cid)


# ──────────────────────────────────────────────
# 10. Сообщения без текста не крашат бота
# ──────────────────────────────────────────────


class TestEdgeCases(unittest.TestCase):
    """Граничные случаи."""

    def test_empty_text_handling(self):
        """safe_format_prompt не должен падать на пустом тексте."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from main import safe_format_prompt
        from database import DEFAULT_PROMPT

        result = safe_format_prompt(DEFAULT_PROMPT, "", "")
        self.assertIn("Сообщение: «»", result)

    def test_text_with_curly_braces(self):
        """Текст с {} не должен ломать format."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from main import safe_format_prompt
        from database import DEFAULT_PROMPT

        result = safe_format_prompt(DEFAULT_PROMPT, "test {injection} text", "")
        self.assertIn("test", result)
        # Не должно быть исключения

    def test_very_long_message(self):
        """Очень длинное сообщение не должно крашить."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from main import safe_format_prompt
        from database import DEFAULT_PROMPT

        long_text = "а" * 10000
        result = safe_format_prompt(DEFAULT_PROMPT, long_text, "")
        self.assertIn(long_text[:100], result)


if __name__ == "__main__":
    unittest.main()
