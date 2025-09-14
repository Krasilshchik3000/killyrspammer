#!/usr/bin/env python3
"""
Тест обработчика админских сообщений
"""
import asyncio
import sys
sys.path.insert(0, '.')

from aiogram import types
from unittest.mock import MagicMock
from main import handle_admin_text, ADMIN_ID
from database import init_database, set_bot_state

async def test_handler():
    print("🧪 Тестирование обработчика админских сообщений")
    
    # Инициализируем БД
    init_database()
    
    # Устанавливаем состояние редактирования
    set_bot_state(ADMIN_ID, awaiting_prompt_edit=True)
    
    # Создаем мок сообщения
    message = MagicMock()
    message.from_user.id = ADMIN_ID
    message.text = "Тестовый промпт с {message_text}"
    message.chat.type = "private"
    message.reply = MagicMock()
    
    print(f"📝 Тестовое сообщение: {message.text}")
    print(f"👤 От пользователя: {message.from_user.id}")
    print(f"💬 Тип чата: {message.chat.type}")
    
    try:
        # Вызываем обработчик
        await handle_admin_text(message)
        print("✅ Обработчик выполнился без ошибок")
    except Exception as e:
        print(f"❌ Ошибка в обработчике: {e}")

if __name__ == "__main__":
    asyncio.run(test_handler())
