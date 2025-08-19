# Автоматический деплой Python Telegram бота на CapRover

Этот документ описывает настройку автоматического деплоя вашего Python Telegram бота на CapRover через GitHub Actions.

## Предварительные требования

1. **CapRover сервер** - настроенный и работающий
2. **GitHub репозиторий** с вашим ботом
3. **Docker** - для сборки образов

## Настройка CapRover

### 1. Создание приложения в CapRover

1. Войдите в панель CapRover
2. Создайте новое приложение с именем `papavoicetg` (или любым другим)
3. В настройках приложения:
   - **App Type**: `Docker`
   - **Source**: `GitHub`
   - **Repository**: `your-username/papaVoiceTG`
   - **Branch**: `main`

### 2. Получение токена приложения

1. В настройках приложения перейдите в раздел **App Tokens**
2. Создайте новый токен
3. Скопируйте токен (он понадобится для GitHub Secrets)

## Настройка GitHub Secrets

Добавьте следующие секреты в ваш GitHub репозиторий:

1. Перейдите в **Settings** → **Secrets and variables** → **Actions**
2. Добавьте новые секреты:

```
CAPROVER_SERVER=https://your-caprover-domain.com
CAPROVER_APP=papavoicetg
CAPROVER_APP_TOKEN=your-app-token-here
```

## Настройка переменных окружения в CapRover

В настройках приложения CapRover добавьте следующие переменные окружения:

```
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
GROQ_API_KEY=your-groq-api-key
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_API_KEY2=your-elevenlabs-api-key-2
ADMIN_ID=your-admin-telegram-id
USE_LOCAL=false
```

## Структура файлов для деплоя

```
papaVoiceTG/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow
├── .dockerignore               # Исключения для Docker
├── captain-definition          # Конфигурация CapRover
├── Dockerfile                  # Docker образ
├── requirements.txt            # Python зависимости
├── main.py                     # Основной код бота
└── CAPROVER_DEPLOY.md         # Эта документация
```

## Процесс деплоя

1. **Автоматический деплой**: При каждом пуше в ветку `main` GitHub Actions автоматически задеплоит бота на CapRover
2. **Ручной деплой**: Можно запустить деплой вручную через GitHub Actions

## Мониторинг

- **Логи приложения**: Доступны в панели CapRover
- **Статус деплоя**: Отслеживается в GitHub Actions
- **Метрики**: Доступны в CapRover Dashboard

## Устранение неполадок

### Проблемы с деплоем
1. Проверьте логи GitHub Actions
2. Убедитесь, что все секреты настроены правильно
3. Проверьте логи приложения в CapRover

### Проблемы с запуском
1. Проверьте переменные окружения в CapRover
2. Убедитесь, что все зависимости установлены
3. Проверьте логи контейнера

## Полезные команды CapRover

```bash
# Подключение к контейнеру
caprover connect

# Просмотр логов
caprover logs

# Перезапуск приложения
caprover restart
```

## Безопасность

- Никогда не коммитьте секреты в код
- Используйте GitHub Secrets для хранения чувствительной информации
- Регулярно обновляйте токены и API ключи
- Мониторьте доступы и логи
