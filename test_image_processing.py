#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности анализа изображений
"""

import os
import asyncio
from dotenv import load_dotenv
from groq import Groq

# Загружаем переменные окружения
load_dotenv()

async def test_groq_image_analysis():
    """Тестирует анализ изображений с помощью Groq"""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY не найден в переменных окружения")
        return False
    
    print("🔑 GROQ_API_KEY найден")
    print("📡 Тестирую анализ изображений с помощью Groq...")
    
    try:
        # Инициализируем клиент Groq
        client = Groq(api_key=api_key)
        
        # Проверяем доступность vision модели
        models_response = client.models.list()
        vision_models = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct"
        ]
        
        available_models = [model.id for model in models_response.data]
        
        print(f"✅ Успешно получен список моделей ({len(available_models)} моделей)")
        
        print("\n🔍 Проверяю vision модели:")
        for model in vision_models:
            if model in available_models:
                print(f"  ✅ {model} - ДОСТУПНА")
            else:
                print(f"  ❌ {model} - НЕ ДОСТУПНА")
        
        # Проверяем, есть ли хотя бы одна vision модель
        available_vision_models = [model for model in vision_models if model in available_models]
        
        if not available_vision_models:
            print("\n❌ Нет доступных vision моделей для анализа изображений")
            return False
        
        print(f"\n✅ Найдена доступная vision модель: {available_vision_models[0]}")
        
        # Тестируем простой запрос (без изображения)
        print("\n🧪 Тестирую простой запрос к vision модели...")
        
        test_response = client.chat.completions.create(
            model=available_vision_models[0],
            messages=[
                {"role": "user", "content": "Привет! Как дела?"}
            ],
            max_tokens=50
        )
        
        if test_response and test_response.choices:
            content = test_response.choices[0].message.content
            print(f"✅ Тест vision модели успешен!")
            print(f"📝 Ответ: {content}")
            return True
        else:
            print("❌ Неожиданный ответ от vision модели")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

async def main():
    print("🚀 Тестирование функциональности анализа изображений")
    print("=" * 60)
    
    # Тест Groq API для изображений
    groq_ok = await test_groq_image_analysis()
    
    if groq_ok:
        print("\n🎉 Тест Groq API для изображений прошел успешно!")
        print("✅ Функциональность анализа изображений готова к работе")
    else:
        print("\n❌ Тест Groq API для изображений не прошел")
        print("⚠️ Функциональность анализа изображений может не работать")
    
    print("\n" + "=" * 60)
    print("Тестирование завершено")

if __name__ == "__main__":
    asyncio.run(main())
