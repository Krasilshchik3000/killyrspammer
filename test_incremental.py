#!/usr/bin/env python3
"""
Тест инкрементального обучения
"""

def test_incremental_learning():
    print("🧪 Тестирование инкрементального обучения")
    
    # Существующий промпт
    current_prompt = """Проанализируй сообщение из телеграм-группы и ответь строго одним из трёх вариантов:
СПАМ
НЕ_СПАМ
ВОЗМОЖНО_СПАМ

Считай особенно подозрительными: безадресные вакансии/работу "без опыта/высокий доход", призывы писать в ЛС/бота/внешние ссылки, сердечки 💘/💝 с намёком на интим-услуги. Если данных мало — выбирай ВОЗМОЖНО_СПАМ.

Сообщение: «{message_text}»

Ответ:"""
    
    # Симуляция ответа ChatGPT с дополнением
    analysis = """АНАЛИЗ: Был неуверен, так как сообщение состояло только из одного эмодзи.

ДОПОЛНЕНИЕ_К_КРИТЕРИЯМ: 6. Сообщения, состоящие только из эмодзи-сердечек 💘/💝, даже без текста, считай СПАМ."""
    
    print("📝 Исходный промпт:")
    print(current_prompt[200:400] + "...")
    
    if "ДОПОЛНЕНИЕ_К_КРИТЕРИЯМ:" in analysis:
        addition = analysis.split("ДОПОЛНЕНИЕ_К_КРИТЕРИЯМ:")[1].strip()
        
        # Извлекаем существующие критерии
        current_criteria_start = current_prompt.find("Считай особенно подозрительными")
        current_criteria_end = current_prompt.find("Сообщение:")
        
        if current_criteria_start != -1 and current_criteria_end != -1:
            existing_criteria = current_prompt[current_criteria_start:current_criteria_end].strip()
            
            # Дополняем существующие критерии
            improved_criteria = f"{existing_criteria}\n\n{addition}"
            
            print("\n✅ Дополненные критерии:")
            print(improved_criteria)
            
            # Собираем полный промпт
            improved_prompt = f"""Проанализируй сообщение из телеграм-группы и ответь строго одним из трёх вариантов:
СПАМ
НЕ_СПАМ
ВОЗМОЖНО_СПАМ

{improved_criteria}

Сообщение: {{message_text}}

Ответ:"""
            
            print("\n🔍 Проверяем сохранение знаний:")
            if "без опыта/высокий доход" in improved_prompt:
                print("✅ Старые знания о вакансиях сохранены")
            else:
                print("❌ Потеряны знания о вакансиях")
                
            if "эмодзи-сердечек" in improved_prompt:
                print("✅ Новые знания об эмодзи добавлены")
            else:
                print("❌ Новые знания не добавлены")

if __name__ == "__main__":
    test_incremental_learning()
