#!/usr/bin/env python3
"""
Тест анализа ошибок бота
"""
import asyncio
import sys
sys.path.insert(0, '.')

from main import analyze_bot_error, get_current_prompt
from config import OPENAI_API_KEY
from openai import AsyncOpenAI

async def test_analysis():
    print("🧪 Тестирование анализа ошибок бота")
    
    # Инициализируем OpenAI
    import main
    main.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Тест 1: Пропущенный спам
    print("\n1️⃣ Тестирую анализ пропущенного спама...")
    message_text = "Нужны грузчики. Пишите в лс"
    
    analysis, improved_prompt = await analyze_bot_error(message_text, "missed_spam")
    
    if analysis:
        print(f"✅ Анализ получен: {analysis[:100]}...")
        if improved_prompt:
            print(f"✅ Улучшенный промпт получен: {improved_prompt[:100]}...")
        else:
            print("❌ Улучшенный промпт НЕ извлечен")
    else:
        print("❌ Анализ НЕ получен")
    
    # Тест 2: Ложное срабатывание
    print("\n2️⃣ Тестирую анализ ложного срабатывания...")
    message_text = "Привет всем! Как дела?"
    
    analysis, improved_prompt = await analyze_bot_error(message_text, "false_positive")
    
    if analysis:
        print(f"✅ Анализ получен: {analysis[:100]}...")
        if improved_prompt:
            print(f"✅ Улучшенный промпт получен: {improved_prompt[:100]}...")
        else:
            print("❌ Улучшенный промпт НЕ извлечен")
    else:
        print("❌ Анализ НЕ получен")

if __name__ == "__main__":
    asyncio.run(test_analysis())
