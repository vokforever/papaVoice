# Используем официальный образ Python как базовый
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем директорию для аудио файлов
RUN mkdir -p audio_files/cache

# Устанавливаем переменную окружения для режима API
ENV USE_LOCAL=false

# Устанавливаем порт для CapRover (если понадобится)
EXPOSE 8080

# Команда для запуска приложения при старте контейнера
CMD ["python", "main.py"]
