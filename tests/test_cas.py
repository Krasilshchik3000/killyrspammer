"""Тесты для CAS (Combot Anti-Spam) интеграции."""
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

os.environ["DATABASE_URL"] = ""
os.environ["DATABASE_PATH"] = ":memory:"
os.environ["BOT_TOKEN"] = "test"
os.environ["OPENAI_API_KEY"] = "test"
os.environ["ADMIN_ID"] = "123456"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import check_cas_ban


@pytest.mark.asyncio
class TestCheckCasBan:
    async def test_banned_user(self):
        """CAS возвращает ok=True для забаненного."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"offenses": 1}}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch('main._http_client', mock_client):
            result = await check_cas_ban(12345)
            assert result is True
            mock_client.get.assert_called_once()

    async def test_clean_user(self):
        """CAS возвращает ok=False для чистого пользователя."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch('main._http_client', mock_client):
            result = await check_cas_ban(12345)
            assert result is False

    async def test_api_error_returns_false(self):
        """При ошибке CAS API не блокируем пользователя."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection timeout")

        with patch('main._http_client', mock_client):
            result = await check_cas_ban(12345)
            assert result is False

    async def test_malformed_response(self):
        """Некорректный ответ от CAS не крашит бота."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch('main._http_client', mock_client):
            result = await check_cas_ban(12345)
            assert result is False
