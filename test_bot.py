#!/usr/bin/env python3
"""
Система тестирования антиспам-бота
Проверяет все функции на работоспособность
"""
import asyncio
import sqlite3
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from main import (
    SpamResult, parse_llm_response, init_database, save_message_to_db,
    update_admin_decision, add_training_example, SPAM_CHECK_PROMPT
)
from config import BOT_TOKEN, OPENAI_API_KEY, ADMIN_ID

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def success(self, test_name):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def fail(self, test_name, error):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n📊 Результаты тестирования:")
        print(f"   Всего тестов: {total}")
        print(f"   ✅ Пройдено: {self.passed}")
        print(f"   ❌ Провалено: {self.failed}")
        
        if self.failed > 0:
            print(f"\n🔍 Ошибки:")
            for error in self.errors:
                print(f"   • {error}")
        
        return self.failed == 0

def test_config():
    """Тест конфигурации"""
    results = TestResults()
    
    # Проверка токенов
    if not BOT_TOKEN or BOT_TOKEN == "your-bot-token-here":
        results.fail("Config: BOT_TOKEN", "Токен бота не настроен")
    else:
        results.success("Config: BOT_TOKEN настроен")
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        results.fail("Config: OPENAI_API_KEY", "API ключ OpenAI не настроен")
    else:
        results.success("Config: OPENAI_API_KEY настроен")
    
    if not ADMIN_ID or ADMIN_ID == 123456789:
        results.fail("Config: ADMIN_ID", "ID администратора не настроен")
    else:
        results.success("Config: ADMIN_ID настроен")
    
    return results

def test_llm_response_parser():
    """Тест парсера ответов LLM"""
    results = TestResults()
    
    test_cases = [
        ("СПАМ", SpamResult.SPAM),
        ("спам", SpamResult.SPAM),
        ("НЕ_СПАМ", SpamResult.NOT_SPAM),
        ("не спам", SpamResult.NOT_SPAM),
        ("ВОЗМОЖНО_СПАМ", SpamResult.MAYBE_SPAM),
        ("возможно спам", SpamResult.MAYBE_SPAM),
        ("непонятный ответ", SpamResult.MAYBE_SPAM),
        ("", SpamResult.MAYBE_SPAM),
    ]
    
    for input_text, expected in test_cases:
        try:
            result = parse_llm_response(input_text)
            if result == expected:
                results.success(f"Parser: '{input_text}' → {expected.value}")
            else:
                results.fail(f"Parser: '{input_text}'", f"Ожидали {expected.value}, получили {result.value}")
        except Exception as e:
            results.fail(f"Parser: '{input_text}'", str(e))
    
    return results

def test_database():
    """Тест базы данных"""
    results = TestResults()
    
    # Создаем временную БД
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        tmp_db = tmp.name
    
    try:
        # Временно подменяем функции для работы с тестовой БД
        original_functions = {}
        
        def mock_init_database():
            conn = sqlite3.connect(tmp_db)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    message_id INTEGER,
                    chat_id INTEGER,
                    user_id INTEGER,
                    username TEXT,
                    text TEXT,
                    created_at TIMESTAMP,
                    llm_result TEXT,
                    admin_decision TEXT,
                    admin_decided_at TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    is_spam BOOLEAN,
                    source TEXT,
                    created_at TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        
        def mock_add_training_example(text, is_spam, source):
            conn = sqlite3.connect(tmp_db)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_examples (text, is_spam, source, created_at)
                VALUES (?, ?, ?, ?)
            ''', (text, is_spam, source, datetime.now()))
            conn.commit()
            conn.close()
        
        # Тест инициализации БД
        try:
            mock_init_database()
            results.success("Database: Инициализация")
        except Exception as e:
            results.fail("Database: Инициализация", str(e))
            return results
        
        # Проверяем структуру таблиц
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        
        # Проверка таблицы messages
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if cursor.fetchone():
            results.success("Database: Таблица messages создана")
        else:
            results.fail("Database: Таблица messages", "Таблица не создана")
        
        # Проверка таблицы training_examples
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_examples'")
        if cursor.fetchone():
            results.success("Database: Таблица training_examples создана")
        else:
            results.fail("Database: Таблица training_examples", "Таблица не создана")
        
        # Тест добавления обучающего примера
        try:
            mock_add_training_example("Тестовое сообщение", True, "TEST")
            cursor.execute("SELECT COUNT(*) FROM training_examples")
            count = cursor.fetchone()[0]
            if count > 0:
                results.success("Database: Добавление обучающего примера")
            else:
                results.fail("Database: Добавление обучающего примера", "Запись не добавлена")
        except Exception as e:
            results.fail("Database: Добавление обучающего примера", str(e))
        
        conn.close()
        
    finally:
        # Удаляем временную БД
        if os.path.exists(tmp_db):
            os.unlink(tmp_db)
    
    return results

def test_prompt():
    """Тест промпта"""
    results = TestResults()
    
    test_message = "Работа без опыта! Пишите в ЛС 💘"
    
    try:
        prompt = SPAM_CHECK_PROMPT.format(message_text=test_message)
        
        # Проверяем, что промпт содержит нужные элементы
        if "СПАМ" in prompt and "НЕ_СПАМ" in prompt and "ВОЗМОЖНО_СПАМ" in prompt:
            results.success("Prompt: Содержит все варианты ответов")
        else:
            results.fail("Prompt: Варианты ответов", "Не все варианты найдены в промпте")
        
        if test_message in prompt:
            results.success("Prompt: Подстановка сообщения работает")
        else:
            results.fail("Prompt: Подстановка сообщения", "Сообщение не найдено в промпте")
        
        # Проверяем критерии спама
        spam_criteria = ["безадресные вакансии", "ЛС", "💘", "💝"]
        found_criteria = sum(1 for criterion in spam_criteria if criterion in prompt)
        
        if found_criteria >= 3:
            results.success("Prompt: Критерии спама присутствуют")
        else:
            results.fail("Prompt: Критерии спама", f"Найдено только {found_criteria} из {len(spam_criteria)} критериев")
        
    except Exception as e:
        results.fail("Prompt: Формирование", str(e))
    
    return results

async def test_openai_connection():
    """Тест подключения к OpenAI"""
    results = TestResults()
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Простой тест запрос
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Ответь одним словом: тест"}],
            max_tokens=5,
            temperature=0,
            timeout=10
        )
        
        if response and response.choices:
            results.success("OpenAI: Подключение работает")
            
            # Тестируем наш промпт
            test_message = "Работа без опыта! Высокий доход! Пишите в ЛС"
            prompt = SPAM_CHECK_PROMPT.format(message_text=test_message)
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
                timeout=10
            )
            
            llm_answer = response.choices[0].message.content.strip()
            parsed_result = parse_llm_response(llm_answer)
            
            if parsed_result in [SpamResult.SPAM, SpamResult.MAYBE_SPAM]:
                results.success(f"OpenAI: Определение спама работает (ответ: {parsed_result.value})")
            else:
                results.fail("OpenAI: Определение спама", f"Неожиданный результат: {parsed_result.value}")
                
        else:
            results.fail("OpenAI: Подключение", "Пустой ответ от API")
            
    except Exception as e:
        results.fail("OpenAI: Подключение", str(e))
    
    return results

def test_telegram_bot_token():
    """Тест токена Telegram бота"""
    results = TestResults()
    
    try:
        from aiogram import Bot
        bot = Bot(token=BOT_TOKEN)
        
        # Проверяем формат токена
        if ":" in BOT_TOKEN and len(BOT_TOKEN.split(":")) == 2:
            bot_id, token_part = BOT_TOKEN.split(":")
            if bot_id.isdigit() and len(token_part) >= 30:
                results.success("Telegram: Формат токена корректный")
            else:
                results.fail("Telegram: Формат токена", "Неверный формат токена")
        else:
            results.fail("Telegram: Формат токена", "Неверный формат токена")
            
    except Exception as e:
        results.fail("Telegram: Токен", str(e))
    
    return results

async def run_all_tests():
    """Запуск всех тестов"""
    print("🧪 Запуск тестирования антиспам-бота\n")
    
    all_results = TestResults()
    
    # Тест конфигурации
    print("📋 Тестирование конфигурации:")
    config_results = test_config()
    all_results.passed += config_results.passed
    all_results.failed += config_results.failed
    all_results.errors.extend(config_results.errors)
    
    # Тест парсера
    print("\n🔍 Тестирование парсера ответов LLM:")
    parser_results = test_llm_response_parser()
    all_results.passed += parser_results.passed
    all_results.failed += parser_results.failed
    all_results.errors.extend(parser_results.errors)
    
    # Тест базы данных
    print("\n💾 Тестирование базы данных:")
    db_results = test_database()
    all_results.passed += db_results.passed
    all_results.failed += db_results.failed
    all_results.errors.extend(db_results.errors)
    
    # Тест промпта
    print("\n💬 Тестирование промпта:")
    prompt_results = test_prompt()
    all_results.passed += prompt_results.passed
    all_results.failed += prompt_results.failed
    all_results.errors.extend(prompt_results.errors)
    
    # Тест токена Telegram
    print("\n📱 Тестирование Telegram токена:")
    telegram_results = test_telegram_bot_token()
    all_results.passed += telegram_results.passed
    all_results.failed += telegram_results.failed
    all_results.errors.extend(telegram_results.errors)
    
    # Тест OpenAI (если ключ настроен)
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here":
        print("\n🤖 Тестирование OpenAI API:")
        openai_results = await test_openai_connection()
        all_results.passed += openai_results.passed
        all_results.failed += openai_results.failed
        all_results.errors.extend(openai_results.errors)
    else:
        print("\n🤖 Пропуск тестирования OpenAI API (ключ не настроен)")
    
    # Итоговые результаты
    success = all_results.summary()
    
    if success:
        print("\n🎉 Все тесты пройдены! Бот готов к запуску.")
    else:
        print("\n⚠️ Некоторые тесты провалены. Проверьте ошибки выше.")
    
    return success

if __name__ == "__main__":
    asyncio.run(run_all_tests())
