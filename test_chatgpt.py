#!/usr/bin/env python3
"""
Тестирование ChatGPT ответов
"""
import asyncio
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

SPAM_CHECK_PROMPT = """Проанализируй сообщение из телеграм-группы и ответь строго одним из трёх вариантов:
СПАМ
НЕ_СПАМ  
ВОЗМОЖНО_СПАМ

Считай особенно подозрительными: безадресные вакансии/работу "без опыта/высокий доход", призывы писать в ЛС/бота/внешние ссылки, сердечки 💘/💝 с намёком на интим-услуги. Если данных мало — выбирай ВОЗМОЖНО_СПАМ.

Сообщение: «{message_text}»

Ответ:"""

async def test_chatgpt():
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    test_messages = [
        "Привет всем!",  # Должно быть НЕ_СПАМ
        "Работа без опыта! Высокий доход!",  # Должно быть СПАМ
        "💘 Пишите в ЛС",  # Должно быть СПАМ
        "Как дела?",  # Должно быть НЕ_СПАМ
        "asdas"  # Должно быть НЕ_СПАМ или ВОЗМОЖНО_СПАМ
    ]
    
    for msg in test_messages:
        prompt = SPAM_CHECK_PROMPT.format(message_text=msg)
        
        print(f"\n🧪 Тестирую: '{msg}'")
        print("=" * 50)
        
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0,
                timeout=10
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"🎯 ChatGPT ответил: '{answer}' (длина: {len(answer)})")
            print(f"💰 Использовано токенов: {response.usage.total_tokens}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(test_chatgpt())
