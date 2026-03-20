"""Тесты для бизнес-логики main.py — парсинг, few-shot, валидация, пропуск сообщений."""
import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

os.environ["DATABASE_URL"] = ""
os.environ["DATABASE_PATH"] = ":memory:"
os.environ["BOT_TOKEN"] = "test"
os.environ["OPENAI_API_KEY"] = "test"
os.environ["ADMIN_ID"] = "123456"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParseLLMResponse:
    """Тесты парсинга ответов LLM."""

    def setup_method(self):
        from main import parse_llm_response, SpamResult
        self.parse = parse_llm_response
        self.SR = SpamResult

    def test_exact_spam(self):
        assert self.parse("СПАМ") == self.SR.SPAM
        assert self.parse("SPAM") == self.SR.SPAM

    def test_exact_not_spam(self):
        assert self.parse("НЕ_СПАМ") == self.SR.NOT_SPAM
        assert self.parse("НЕ СПАМ") == self.SR.NOT_SPAM
        assert self.parse("NOT_SPAM") == self.SR.NOT_SPAM

    def test_exact_maybe_spam(self):
        assert self.parse("ВОЗМОЖНО_СПАМ") == self.SR.MAYBE_SPAM
        assert self.parse("ВОЗМОЖНО СПАМ") == self.SR.MAYBE_SPAM
        assert self.parse("MAYBE_SPAM") == self.SR.MAYBE_SPAM

    def test_with_punctuation(self):
        """Ответ с точками/кавычками парсится корректно."""
        assert self.parse("СПАМ.") == self.SR.SPAM
        assert self.parse('"НЕ_СПАМ"') == self.SR.NOT_SPAM

    def test_mixed_case(self):
        assert self.parse("спам") == self.SR.SPAM
        assert self.parse("Не_спам") == self.SR.NOT_SPAM

    def test_partial_match_maybe_before_spam(self):
        """ВОЗМОЖНО должен матчиться раньше СПАМ."""
        assert self.parse("ВОЗМОЖНО_СПАМ - это подозрительно") == self.SR.MAYBE_SPAM

    def test_partial_match_not_spam_before_spam(self):
        """НЕ_СПАМ должен матчиться раньше СПАМ."""
        assert self.parse("Ответ: НЕ_СПАМ") == self.SR.NOT_SPAM

    def test_too_short_response(self):
        """Слишком короткие ответы → MAYBE_SPAM."""
        assert self.parse("Н") == self.SR.MAYBE_SPAM
        assert self.parse("") == self.SR.MAYBE_SPAM
        assert self.parse("ab") == self.SR.MAYBE_SPAM

    def test_garbage_response(self):
        """Мусор → MAYBE_SPAM."""
        assert self.parse("I don't understand the question") == self.SR.MAYBE_SPAM

    def test_verbose_response_with_spam(self):
        """Длинный ответ с СПАМ внутри."""
        assert self.parse("По всем признакам это СПАМ") == self.SR.SPAM


class TestBuildFewShotBlock:
    def setup_method(self):
        from main import build_few_shot_block
        self.build = build_few_shot_block

    @patch('main.db')
    def test_empty_examples(self, mock_db):
        mock_db.get_few_shot_examples.return_value = []
        assert self.build() == ""

    @patch('main.db')
    def test_with_examples(self, mock_db):
        mock_db.get_few_shot_examples.return_value = [
            ("Купи крипту!", True),
            ("Привет, как дела?", False),
        ]
        result = self.build()
        assert "Примеры" in result
        assert "СПАМ" in result
        assert "НЕ_СПАМ" in result
        assert "Купи крипту!" in result

    @patch('main.db')
    def test_truncates_long_text(self, mock_db):
        """Длинные примеры обрезаются до 120 символов."""
        long_text = "A" * 200
        mock_db.get_few_shot_examples.return_value = [(long_text, True)]
        result = self.build()
        # Текст в результате не длиннее 120
        assert "A" * 121 not in result


class TestSafeFormatPrompt:
    def setup_method(self):
        from main import safe_format_prompt
        self.fmt = safe_format_prompt

    def test_normal_format(self):
        template = "Анализируй: {few_shot_block}Сообщение: «{message_text}»"
        result = self.fmt(template, "hello", "examples here\n")
        assert "hello" in result
        assert "examples here" in result

    def test_without_few_shot_block(self):
        """Старый промпт без {few_shot_block} не падает."""
        template = "Сообщение: «{message_text}»\nОтвет:"
        result = self.fmt(template, "hello", "ignored")
        assert "hello" in result

    def test_broken_template(self):
        """Совсем сломанный шаблон — подставляется вручную."""
        template = "Текст: {message_text} {unknown_var}"
        result = self.fmt(template, "hello", "")
        assert "hello" in result

    def test_curly_braces_in_message(self):
        """Фигурные скобки в тексте сообщения не ломают format()."""
        template = "Сообщение: «{message_text}»"
        result = self.fmt(template, "test {injection} {{double}}", "")
        assert "injection" in result
        # Не должно упасть


class TestValidatePrompt:
    def setup_method(self):
        from main import validate_prompt
        self.validate = validate_prompt

    def test_valid_prompt(self):
        prompt = "Ответь СПАМ, НЕ_СПАМ или ВОЗМОЖНО_СПАМ. Сообщение: {message_text}"
        assert self.validate(prompt) == []

    def test_missing_message_text(self):
        prompt = "Ответь СПАМ, НЕ_СПАМ или ВОЗМОЖНО_СПАМ"
        problems = self.validate(prompt)
        assert any("message_text" in p for p in problems)

    def test_missing_categories(self):
        prompt = "Сообщение: {message_text}"
        problems = self.validate(prompt)
        assert len(problems) == 3  # Все три категории отсутствуют


class TestShouldSkipMessage:
    def setup_method(self):
        from main import should_skip_message
        self.skip = should_skip_message

    def _make_message(self, user_id=1, is_bot=False, sender_chat=None, text="hello"):
        msg = MagicMock()
        msg.from_user = MagicMock()
        msg.from_user.id = user_id
        msg.from_user.is_bot = is_bot
        msg.sender_chat = sender_chat
        msg.text = text
        return msg

    def test_skip_bot(self):
        msg = self._make_message(is_bot=True)
        assert self.skip(msg) is True

    def test_skip_admin(self):
        msg = self._make_message(user_id=123456)  # ADMIN_ID from env
        assert self.skip(msg) is True

    def test_skip_channel(self):
        sender_chat = MagicMock()
        sender_chat.title = "Test Channel"
        sender_chat.id = -1001
        msg = self._make_message(sender_chat=sender_chat)
        assert self.skip(msg) is True

    def test_skip_command(self):
        msg = self._make_message(text="/start")
        assert self.skip(msg) is True

    def test_normal_user_not_skipped(self):
        msg = self._make_message(user_id=999, text="привет")
        assert self.skip(msg) is False


class TestCheckRateLimit:
    def setup_method(self):
        from main import check_rate_limit, _user_request_times
        self.check = check_rate_limit
        self.times = _user_request_times
        self.times.clear()

    def test_allows_first_request(self):
        assert self.check(1) is True

    def test_allows_up_to_limit(self):
        for _ in range(5):
            assert self.check(1) is True

    def test_blocks_over_limit(self):
        for _ in range(5):
            self.check(1)
        assert self.check(1) is False

    def test_different_users_independent(self):
        for _ in range(5):
            self.check(1)
        assert self.check(2) is True
