# Используем официальный образ Python как базовый
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости и сразу очищаем кэш
RUN pip install --no-cache-dir -r requirements.txt

# Копируем только main.py, а не все файлы
COPY main.py .

# Создаем директорию для данных
RUN mkdir -p data

# Эти переменные окружения здесь указаны с "dummy" значениями.
# Реальные значения должны быть настроены в Coolify в разделе "Variables" вашего сервиса.
ENV OPENAI_API_KEY="dummy" \
    TELEGRAM_BOT_TOKEN="dummy" \
    GROQ_API_KEY="dummy" \
    ELEVENLABS_API_KEY="dummy" \
    ELEVENLABS_API_KEY2="dummy" \
    ADMIN_ID="dummy" \
    GROQ_TRANSLATION_MODEL="deepseek-r1-distill-llama-70b"

# Команда для запуска приложения при старте контейнера
CMD ["python", "main.py"]
