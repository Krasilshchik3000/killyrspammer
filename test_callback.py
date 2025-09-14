#!/usr/bin/env python3
"""
Тест обработки кнопок
"""
import asyncio
import sys
sys.path.insert(0, '.')

from unittest.mock import MagicMock, AsyncMock
from main import handle_admin_feedback, ADMIN_ID
from database import init_database, execute_query

async def test_callback():
    print("🧪 Тестирование обработки кнопок")
    
    # Инициализируем БД
    init_database()
    
    # Добавляем тестовое сообщение в БД
    test_message_id = 12345
    test_text = "Тестовое сообщение"
    test_llm_result = "ВОЗМОЖНО_СПАМ"
    
    try:
        execute_query('''
            INSERT INTO messages (message_id, chat_id, user_id, username, text, created_at, llm_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (test_message_id, -123, 456, "test", test_text, "2024-01-01", test_llm_result))
        print(f"✅ Тестовое сообщение добавлено в БД")
    except:
        print("⚠️ Сообщение уже есть в БД")
    
    # Создаем мок callback
    callback = MagicMock()
    callback.from_user.id = ADMIN_ID
    callback.data = f"not_spam_{test_message_id}"
    callback.answer = AsyncMock()
    callback.message.edit_text = AsyncMock()
    callback.message.text = "Тестовое сообщение от бота"
    
    print(f"🔘 Тестирую callback: {callback.data}")
    print(f"👤 От пользователя: {callback.from_user.id}")
    
    try:
        # Мокируем бота
        import main
        main.bot = MagicMock()
        main.bot.send_message = AsyncMock()
        
        # Вызываем обработчик
        await handle_admin_feedback(callback)
        print("✅ Обработчик кнопки выполнился без ошибок")
        
        # Проверяем, что callback.answer был вызван
        if callback.answer.called:
            print("✅ callback.answer был вызван")
        else:
            print("❌ callback.answer НЕ был вызван")
            
    except Exception as e:
        print(f"❌ Ошибка в обработчике кнопки: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_callback())
