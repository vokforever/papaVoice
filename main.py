# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
import asyncio
import io
import re  # Добавляем недостающий импорт
import json  # Добавляем импорт для работы с JSON
from pathlib import Path
from datetime import datetime, date  # Добавляем date для работы с месяцами
import hashlib
import gc
from functools import partial  # Добавляем импорт partial для run_blocking
from dataclasses import dataclass, asdict  # Добавляем dataclass для структурирования данных
from typing import List, Dict, Optional, Tuple  # Добавляем типы для аннотаций
from dotenv import load_dotenv
from pydub import AudioSegment
from aiogram import Bot, Dispatcher, F, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from groq import Groq, APIError
# Добавляем импорт ElevenLabs
from elevenlabs import ElevenLabs
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings

load_dotenv()

# --- ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ ---
DEBUG = False
USE_LOCAL_STR = os.getenv("USE_LOCAL", "False")
USE_LOCAL = USE_LOCAL_STR.lower() in ('false', '1', 't')
USE_NVIDIA_GPU = True
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID_STR = os.getenv("ADMIN_ID")  # ID мой
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # Добавляем API ключ ElevenLabs
ELEVENLABS_API_KEY2 = os.getenv("ELEVENLABS_API_KEY2")  # Добавляем второй API ключ ElevenLabs
AUDIO_FOLDER = Path("audio_files")

# --- КОНФИГУРАЦИЯ МОДЕЛЕЙ ---
GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3"  # Модель для распознавания речи
GROQ_TRANSLATION_MODEL = "deepseek-r1-distill-llama-70b"  # Модель для обработки текста
SILERO_MODEL_ID = 'v4_ru'
SILERO_SAMPLE_RATE = 48000
USE_HALF_PRECISION_TTS = True

# --- КОНФИГУРАЦИЯ ГОЛОСОВ ---
SILERO_SPEAKER = 'eugene'  # Мужской голос
# Конфигурация ElevenLabs
ELEVENLABS_MALE_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam - мужской голос
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"  # Модель для ElevenLabs
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

# --- УСЛОВНЫЕ ИМПОРТЫ ДЛЯ ЛОКАЛЬНОГО РЕЖИМА ---
if USE_LOCAL:
    try:
        import torch
        import numpy as np
        logger.info("Running in local mode. Torch and numpy imported.")
    except ImportError:
        logger.error("torch and numpy are required for local mode. Please install them.")
        sys.exit("torch and numpy are required for local mode. Please install them.")
else:
    # Отключаем GPU, если не используем локальные зависимости
    USE_NVIDIA_GPU = False
    logger.info("Running in API-only mode. Skipping torch and numpy imports.")

# --- КЛАССЫ ДЛЯ УПРАВЛЕНИЯ ELEVENLABS ---
@dataclass
class ElevenLabsKeyUsage:
    """Класс для хранения информации об использовании ключа"""
    api_key: str
    key_index: int
    character_count: int
    character_limit: int
    last_updated: str
    monthly_usage: Dict[str, int]  # ключ: "YYYY-MM", значение: количество символов

class ElevenLabsManager:
    """Менеджер для управления несколькими ключами ElevenLabs и отслеживания лимитов"""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.usage_file = Path("elevenlabs_usage.json")
        self.key_usage: Dict[str, ElevenLabsKeyUsage] = {}
        self.clients: List[ElevenLabs] = []
        self.async_clients: List[AsyncElevenLabs] = []
        
        # Загружаем статистику использования
        self._load_usage_stats()
        
        # Инициализируем клиентов
        self._initialize_clients()
        
        logger.info(f"ElevenLabsManager инициализирован с {len(api_keys)} ключами")
    
    def _initialize_clients(self):
        """Инициализация клиентов для каждого ключа"""
        self.clients.clear()
        self.async_clients.clear()
        
        for i, api_key in enumerate(self.api_keys):
            try:
                client = ElevenLabs(api_key=api_key)
                async_client = AsyncElevenLabs(api_key=api_key)
                self.clients.append(client)
                self.async_clients.append(async_client)
                
                # Инициализируем статистику для ключа, если её нет
                if api_key not in self.key_usage:
                    self.key_usage[api_key] = ElevenLabsKeyUsage(
                        api_key=api_key,
                        key_index=i,
                        character_count=0,
                        character_limit=0,
                        last_updated=datetime.now().isoformat(),
                        monthly_usage={}
                    )
                
                logger.info(f"Ключ {i+1} ({api_key[:10]}...) инициализирован")
                
            except Exception as e:
                logger.error(f"Ошибка инициализации ключа {i+1}: {e}")
    
    def _load_usage_stats(self):
        """Загрузка статистики использования из файла"""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key_data in data:
                        usage = ElevenLabsKeyUsage(**key_data)
                        self.key_usage[usage.api_key] = usage
                logger.info(f"Статистика использования загружена из {self.usage_file}")
            except Exception as e:
                logger.error(f"Ошибка загрузки статистики: {e}")
    
    def _save_usage_stats(self):
        """Сохранение статистики использования в файл"""
        try:
            data = [asdict(usage) for usage in self.key_usage.values()]
            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Статистика использования сохранена в {self.usage_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения статистики: {e}")
    
    def get_current_month_key(self) -> str:
        """Получить ключ для текущего месяца в формате YYYY-MM"""
        return date.today().strftime("%Y-%m")
    
    async def check_key_limits(self, key_index: int) -> Tuple[bool, str]:
        """Проверить лимиты для конкретного ключа"""
        if key_index >= len(self.clients):
            return False, "Недействительный индекс ключа"
        
        try:
            client = self.clients[key_index]
            user_info = await run_blocking(client.user.get)
            subscription = user_info.subscription
            
            character_count = subscription.character_count
            character_limit = subscription.character_limit
            
            # Обновляем информацию о ключе
            api_key = self.api_keys[key_index]
            if api_key in self.key_usage:
                self.key_usage[api_key].character_count = character_count
                self.key_usage[api_key].character_limit = character_limit
                self.key_usage[api_key].last_updated = datetime.now().isoformat()
            
            # Проверяем доступные символы
            available_chars = character_limit - character_count
            
            # Если осталось меньше 1000 символов, считаем что лимит исчерпан
            if available_chars < 1000:
                return False, f"Ключ {key_index+1}: осталось мало символов ({available_chars})"
            
            return True, f"Ключ {key_index+1}: доступно {available_chars} символов"
            
        except Exception as e:
            logger.error(f"Ошибка проверки лимитов ключа {key_index+1}: {e}")
            return False, f"Ошибка проверки ключа {key_index+1}: {str(e)}"
    
    async def get_available_key(self) -> Tuple[Optional[int], Optional[ElevenLabs], Optional[AsyncElevenLabs], str]:
        """Получить первый доступный ключ с лимитами"""
        start_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.api_keys):
            key_index = (self.current_key_index + attempts) % len(self.api_keys)
            
            is_available, message = await self.check_key_limits(key_index)
            
            if is_available:
                # Обновляем текущий ключ
                self.current_key_index = key_index
                logger.info(f"Выбран ключ {key_index+1}: {message}")
                return key_index, self.clients[key_index], self.async_clients[key_index], message
            
            logger.warning(message)
            attempts += 1
        
        logger.error("Все ключи ElevenLabs исчерпали лимиты")
        return None, None, None, "Все ключи исчерпали лимиты"
    
    async def record_usage(self, key_index: int, characters_used: int):
        """Записать использование символов для ключа"""
        if key_index >= len(self.api_keys):
            return
        
        api_key = self.api_keys[key_index]
        current_month = self.get_current_month_key()
        
        if api_key in self.key_usage:
            usage = self.key_usage[api_key]
            
            # Обновляем месячную статистику
            if current_month not in usage.monthly_usage:
                usage.monthly_usage[current_month] = 0
            usage.monthly_usage[current_month] += characters_used
            
            # Обновляем общее количество
            usage.character_count += characters_used
            usage.last_updated = datetime.now().isoformat()
            
            logger.info(f"Записано использование ключа {key_index+1}: +{characters_used} символов, всего за месяц: {usage.monthly_usage[current_month]}")
            
            # Сохраняем статистику
            self._save_usage_stats()
    
    def get_monthly_usage(self, month_key: str = None) -> Dict[str, int]:
        """Получить статистику использования за месяц"""
        if month_key is None:
            month_key = self.get_current_month_key()
        
        monthly_stats = {}
        for api_key, usage in self.key_usage.items():
            if month_key in usage.monthly_usage:
                key_name = f"Ключ {usage.key_index + 1}"
                monthly_stats[key_name] = usage.monthly_usage[month_key]
        
        return monthly_stats
    
    def get_all_usage_stats(self) -> Dict[str, ElevenLabsKeyUsage]:
        """Получить всю статистику использования"""
        return self.key_usage.copy()
    
    def reset_monthly_stats(self):
        """Сбросить месячную статистику (для тестирования)"""
        for usage in self.key_usage.values():
            usage.monthly_usage.clear()
        self._save_usage_stats()
        logger.info("Месячная статистика сброшена")

if not all([TOKEN, GROQ_API_KEY, ADMIN_ID_STR]):
    logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Не все переменные окружения загружены.")
    sys.exit("Ошибка: не найдены необходимые API ключи или ID в .env файле.")

try:
    ADMIN_ID = int(ADMIN_ID_STR)
    logger.info(f"ADMIN_ID установлен: {ADMIN_ID}")
except ValueError:
    logger.critical(f"Ошибка преобразования ADMIN_ID в число: {ADMIN_ID_STR}")
    sys.exit("Ошибка: ADMIN_ID должен быть числом.")

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
AUDIO_FOLDER.mkdir(exist_ok=True)

# --- КЛИЕНТЫ API ---
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Клиент Groq API успешно инициализирован.")
except Exception as e:
    logger.error(f"Ошибка инициализации клиента Groq: {e}")
    groq_client = None

# Инициализация клиента ElevenLabs
elevenlabs_client = None
async_elevenlabs_client = None

# --- Инициализация менеджера ElevenLabs ---
elevenlabs_manager = None

def initialize_elevenlabs_manager():
    """Инициализация менеджера ElevenLabs"""
    global elevenlabs_manager, elevenlabs_client, async_elevenlabs_client
    
    # Собираем все доступные ключи
    api_keys = []
    if ELEVENLABS_API_KEY:
        api_keys.append(ELEVENLABS_API_KEY)
    if ELEVENLABS_API_KEY2:
        api_keys.append(ELEVENLABS_API_KEY2)
    
    if not api_keys:
        logger.warning("Не найдены ключи ElevenLabs, сервис будет недоступен")
        return False
    
    try:
        elevenlabs_manager = ElevenLabsManager(api_keys)
        
        # Получаем первый доступный ключ
        key_index, client, async_client, message = asyncio.run(elevenlabs_manager.get_available_key())
        
        if key_index is not None:
            elevenlabs_client = client
            async_elevenlabs_client = async_client
            logger.info(f"ElevenLabs клиент инициализирован: {message}")
            return True
        else:
            logger.warning("Нет доступных ключей ElevenLabs")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка инициализации менеджера ElevenLabs: {e}")
        return False

# Инициализируем менеджер при запуске
elevenlabs_available = initialize_elevenlabs_manager()

# --- ОПТИМИЗАЦИИ PYTORCH ---
if USE_LOCAL:
    logger.info("=== ДИАГНОСТИКА PYTORCH ===")
    logger.info(f"Версия PyTorch: {torch.__version__}")
    logger.info(f"Версия CUDA: {torch.version.cuda}")
    logger.info(f"Версия cuDNN: {torch.backends.cudnn.version()}")
    logger.info(f"USE_NVIDIA_GPU: {USE_NVIDIA_GPU}")

    # Дополнительная диагностика PyTorch
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count() if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")

    if USE_NVIDIA_GPU and torch.cuda.is_available():
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability()[0] >= 8:
            logger.info("GPU Ampere+ обнаружен. Включаю поддержку TF32 для ускорения.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        logger.info("Глобальные оптимизации PyTorch для инференса применены.")
        
        # Дополнительная диагностика CUDA
        logger.info(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Не установлен')}")
        logger.info(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Не установлен')}")
        logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Не установлен')}")
    else:
        logger.warning("CUDA недоступна или отключена. Оптимизации GPU не применены.")
        
        # Диагностика причин недоступности CUDA
        logger.info("=== ДИАГНОСТИКА CUDA ===")
        logger.info(f"torch.version.cuda: {torch.version.cuda}")
        logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.info(f"USE_NVIDIA_GPU: {USE_NVIDIA_GPU}")
        
        # Проверяем переменные окружения
        cuda_home = os.environ.get('CUDA_HOME')
        cuda_path = os.environ.get('CUDA_PATH')
        logger.info(f"CUDA_HOME: {cuda_home}")
        logger.info(f"CUDA_PATH: {cuda_path}")

# --- Инициализация локальной модели Silero TTS ---
silero_model = None
silero_device = None

if USE_LOCAL:
    # Проверяем доступность torch.hub
    logger.info("=== ПРОВЕРКА TORCH.HUB ===")
    logger.info(f"torch.hub доступен: {hasattr(torch, 'hub')}")
    if hasattr(torch, 'hub'):
        logger.info(f"torch.hub.load доступен: {hasattr(torch.hub, 'load')}")
        logger.info(f"torch.hub.list доступен: {hasattr(torch.hub, 'list')}")
        logger.info(f"torch.hub.help доступен: {hasattr(torch.hub, 'help')}")

def load_silero_model_alternative():
    """Альтернативный способ загрузки модели Silero TTS"""
    try:
        logger.info("Пробуем альтернативный способ загрузки модели...")
        
        # Определяем устройство
        device = torch.device('cuda' if torch.cuda.is_available() and USE_NVIDIA_GPU else 'cpu')
        logger.info(f"Альтернативная загрузка будет использовать устройство: {device}")
        
        # Загружаем модель без указания устройства
        logger.info("Загружаем модель без указания устройства...")
        model_data = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker=SILERO_MODEL_ID
        )
        
        # Проверяем, что вернул torch.hub.load
        if isinstance(model_data, tuple) and len(model_data) == 2:
            model, _ = model_data
        else:
            model = model_data
        
        if model is None:
            raise ValueError("Модель не была загружена")
        
        logger.info(f"Альтернативная загрузка: модель получена, тип: {type(model)}")
        
        # Проверяем, что модель работает
        test_text = "Тест"
        try:
            # Пробуем сгенерировать тестовый аудио без перемещения модели
            test_file = AUDIO_FOLDER / "test_alt.wav"
            logger.info("Тестируем модель генерацией аудио...")
            
            model.save_wav(
                text=test_text,
                speaker=SILERO_SPEAKER,
                sample_rate=SILERO_SAMPLE_RATE,
                audio_path=str(test_file)
            )
            
            if test_file.exists() and test_file.stat().st_size > 0:
                logger.info("Альтернативная загрузка успешна, модель работает без перемещения")
                # Удаляем тестовый файл
                try:
                    test_file.unlink()
                except Exception:
                    pass
                return model, device
            else:
                raise ValueError("Тестовый файл не создан или пуст")
                
        except Exception as e:
            logger.error(f"Тест модели не удался: {e}")
            # Пробуем проверить модель другим способом
            try:
                if hasattr(model, 'save_wav'):
                    logger.info("Модель имеет метод save_wav, но тест не прошел")
                else:
                    logger.error("Модель не имеет метода save_wav")
                raise
            except Exception:
                raise
            
    except Exception as e:
        logger.error(f"Альтернативная загрузка не удалась: {e}")
        return None, None

def load_silero_model():
    """Загрузка модели Silero TTS с обработкой ошибок"""
    global silero_model, silero_device
    
    logger.info("=== НАЧАЛО ЗАГРУЗКИ МОДЕЛИ ===")
    
    try:
        # Проверяем доступность GPU
        if torch.cuda.is_available() and USE_NVIDIA_GPU:
            silero_device = torch.device('cuda')
            logger.info(f"Найдена CUDA: {torch.cuda.get_device_name(0)}")
            logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} ГБ")
        else:
            silero_device = torch.device('cpu')
            logger.info("CUDA не найдена или отключена, Silero TTS будет использовать CPU.")
        
        # Загружаем модель с правильными параметрами
        logger.info("Загружаем модель Silero TTS...")
        logger.info(f"Параметры загрузки: repo='snakers4/silero-models', model='silero_tts', language='ru', speaker='{SILERO_MODEL_ID}'")
        
        # Проверяем доступность torch.hub
        logger.info(f"torch.hub доступен: {hasattr(torch, 'hub')}")
        logger.info(f"torch.hub.load доступен: {hasattr(torch.hub, 'load')}")
        
        # Проверяем доступность репозитория
        try:
            logger.info("Проверяем доступность репозитория snakers4/silero-models...")
            available_models = torch.hub.list('snakers4/silero-models', verbose=False)
            logger.info(f"Доступные модели: {available_models}")
        except Exception as e:
            logger.warning(f"Не удалось получить список моделей: {e}")
        
        # Сначала пробуем основной способ загрузки
        try:
            logger.info("Загружаем модель silero_tts на нужное устройство...")
            model_data = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker=SILERO_MODEL_ID,
                device=silero_device  # Передаем устройство напрямую
            )
            logger.info("torch.hub.load выполнен успешно")
            
            # Проверяем, что вернул torch.hub.load
            if isinstance(model_data, tuple) and len(model_data) == 2:
                original_silero_model, _ = model_data
            else:
                original_silero_model = model_data
            
            logger.info(f"Результат загрузки: {type(model_data)}")
            if isinstance(model_data, tuple):
                logger.info(f"Кортеж содержит {len(model_data)} элементов")
                for i, item in enumerate(model_data):
                    logger.info(f"Элемент {i}: {type(item)} - {item is not None}")
            
            logger.info("Основной способ загрузки успешен")
            
        except Exception as e:
            logger.warning(f"Основной способ загрузки не удался: {e}")
            logger.info("Пробуем альтернативный способ...")
            original_silero_model, silero_device = load_silero_model_alternative()
            
            if original_silero_model is None:
                raise ValueError("Не удалось загрузить модель ни одним из способов")
        
        if original_silero_model is None:
            raise ValueError("Модель не была загружена (получено None)")
        
        logger.info(f"Модель загружена, тип: {type(original_silero_model)}")
        logger.info(f"Атрибуты модели: {[attr for attr in dir(original_silero_model) if not attr.startswith('_')]}")
        
        # Проверяем наличие необходимых методов
        if not hasattr(original_silero_model, 'save_wav'):
            logger.error("Модель не имеет метода 'save_wav'!")
            logger.error(f"Доступные методы: {[attr for attr in dir(original_silero_model) if not attr.startswith('_')]}")
            
            # Проверяем альтернативные методы
            if hasattr(original_silero_model, 'apply_tts'):
                logger.info("Модель имеет метод 'apply_tts', возможно это правильный метод")
            if hasattr(original_silero_model, 'apply'):
                logger.info("Модель имеет метод 'apply', возможно это правильный метод")
            if hasattr(original_silero_model, 'synthesize'):
                logger.info("Модель имеет метод 'synthesize', возможно это правильный метод")
            if hasattr(original_silero_model, 'forward'):
                logger.info("Модель имеет метод 'forward', возможно это правильный метод")
            
            raise ValueError("Модель не имеет необходимого метода 'save_wav'")
        
        logger.info("Модель имеет необходимый метод 'save_wav'")
        
        # Проверяем сигнатуру метода
        import inspect
        try:
            sig = inspect.signature(original_silero_model.save_wav)
            logger.info(f"Сигнатура метода save_wav: {sig}")
        except Exception as e:
            logger.warning(f"Не удалось получить сигнатуру метода: {e}")
        
        # Проверяем текущее устройство модели
        try:
            if hasattr(original_silero_model, 'device'):
                model_device = original_silero_model.device
                logger.info(f"Модель уже на устройстве: {model_device}")
            else:
                # Пробуем определить устройство через параметры
                try:
                    model_device = next(original_silero_model.parameters()).device
                    logger.info(f"Модель на устройстве (через параметры): {model_device}")
                except Exception:
                    logger.warning("Не удалось определить устройство модели")
                    model_device = None
        except Exception as e:
            logger.warning(f"Не удалось определить устройство модели: {e}")
            model_device = None
        
        # Проверяем, нужно ли перемещать модель
        if model_device is not None and model_device != silero_device:
            logger.info(f"Модель на устройстве {model_device}, нужно переместить на {silero_device}")
            
            # Пробуем переместить модель безопасно
            try:
                logger.info("Пробуем переместить модель на нужное устройство...")
                temp_model = original_silero_model.to(silero_device)
                if temp_model is not None:
                    original_silero_model = temp_model
                    logger.info("Модель успешно перемещена на устройство")
                else:
                    logger.warning("Модель стала None после перемещения, используем как есть")
                    # Пробуем альтернативный способ
                    try:
                        logger.info("Пробуем переместить параметры модели по отдельности...")
                        # Перемещаем параметры модели по отдельности
                        for param in original_silero_model.parameters():
                            param.data = param.data.to(silero_device)
                        logger.info("Параметры модели перемещены по отдельности")
                    except Exception as param_error:
                        logger.warning(f"Не удалось переместить параметры: {param_error}")
            except Exception as e:
                logger.warning(f"Перемещение модели не удалось: {e}")
                logger.info("Используем модель на текущем устройстве")
        else:
            logger.info("Модель уже на нужном устройстве или перемещение не требуется")
        
        # Дополнительная диагностика модели
        try:
            logger.info("=== ДИАГНОСТИКА МОДЕЛИ ===")
            logger.info(f"Тип модели: {type(original_silero_model)}")
            logger.info(f"Модель не None: {original_silero_model is not None}")
            
            if hasattr(original_silero_model, 'parameters'):
                try:
                    param_count = sum(p.numel() for p in original_silero_model.parameters())
                    logger.info(f"Количество параметров модели: {param_count:,}")
                except Exception as e:
                    logger.warning(f"Не удалось посчитать параметры: {e}")
            
            if hasattr(original_silero_model, 'device'):
                logger.info(f"Устройство модели: {original_silero_model.device}")
            
            logger.info("=== КОНЕЦ ДИАГНОСТИКИ МОДЕЛИ ===")
        except Exception as e:
            logger.warning(f"Диагностика модели не удалась: {e}")
        
        silero_model = original_silero_model
        
        logger.info(f"Модель Silero TTS успешно загружена на устройстве '{silero_device}'")
        
        # Финальная проверка работоспособности модели
        try:
            logger.info("Проверяем работоспособность модели...")
            test_text = "Тест"
            test_file = AUDIO_FOLDER / "test_final.wav"
            
            # Пробуем сгенерировать тестовый аудио
            original_silero_model.save_wav(
                text=test_text,
                speaker=SILERO_SPEAKER,
                sample_rate=SILERO_SAMPLE_RATE,
                audio_path=str(test_file)
            )
            
            if test_file.exists() and test_file.stat().st_size > 0:
                logger.info("Модель работает корректно!")
                # Удаляем тестовый файл
                try:
                    test_file.unlink()
                except Exception:
                    pass
            else:
                logger.warning("Тестовый файл не создан или пуст")
                
        except Exception as e:
            logger.error(f"Тест работоспособности модели не удался: {e}")
            logger.warning("Модель может работать некорректно")
        
        # Применяем оптимизации для GPU
        if silero_device.type == 'cuda':
            logger.info("Применяем оптимизации для GPU...")
            try:
                # Устанавливаем режим инференса
                if hasattr(original_silero_model, 'eval'):
                    original_silero_model.eval()
                    logger.info("Режим инференса установлен")
                else:
                    logger.warning("Модель не имеет метода 'eval', пропускаем установку режима инференса")
                
                # Применяем оптимизации PyTorch
                torch.set_grad_enabled(False)
                torch.backends.cudnn.benchmark = True
                
                # Если GPU Ampere+, включаем TF32
                if torch.cuda.get_device_capability()[0] >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("GPU Ampere+ обнаружен. Включаю поддержку TF32 для ускорения.")
                
                # Применяем половинную точность, если возможно
                if USE_HALF_PRECISION_TTS:
                    try:
                        if hasattr(original_silero_model, 'model') and original_silero_model.model is not None:
                            if torch.cuda.is_bf16_supported():
                                original_silero_model.model = original_silero_model.model.to(torch.bfloat16)
                                logger.info("Модель TTS переведена в BFloat16")
                            else:
                                original_silero_model.model = original_silero_model.model.to(torch.float16)
                                logger.info("Модель TTS переведена в Float16")
                    except Exception as e:
                        logger.warning(f"Не удалось применить половинную точность: {e}")
                
                logger.info("Оптимизации GPU применены успешно")
                
            except Exception as e:
                logger.error(f"Ошибка при применении оптимизаций GPU: {e}")
                logger.info("Продолжаем работу без оптимизаций GPU")
        
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить/оптимизировать модель Silero TTS: {e}", exc_info=True)
        silero_model = None
        
        # Финальная проверка, что модель не None
        if silero_model is None:
            logger.error("КРИТИЧЕСКАЯ ОШИБКА: Модель стала None после всех операций")
            logger.error("=== ЗАГРУЗКА МОДЕЛИ ЗАВЕРШЕНА С ОШИБКОЙ ===")
            return False
        
        # Дополнительная проверка модели
        try:
            logger.info(f"Финальная проверка модели:")
            logger.info(f"  - Тип модели: {type(silero_model)}")
            logger.info(f"  - Модель не None: {silero_model is not None}")
            logger.info(f"  - Имеет метод save_wav: {hasattr(silero_model, 'save_wav')}")
            if hasattr(silero_model, 'device'):
                logger.info(f"  - Атрибут device: {silero_model.device}")
            logger.info(f"  - Устройство silero_device: {silero_device}")
        except Exception as e:
            logger.warning(f"Не удалось выполнить финальную проверку модели: {e}")
        
        logger.info(f"Модель Silero TTS '{SILERO_MODEL_ID}' успешно загружена и настроена на устройстве '{silero_device}'.")
        logger.info("=== ЗАГРУЗКА МОДЕЛИ ЗАВЕРШЕНА УСПЕШНО ===")
        return True
        
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить/оптимизировать модель Silero TTS: {e}", exc_info=True)
        silero_model = None
        logger.error("=== ЗАГРУЗКА МОДЕЛИ ЗАВЕРШЕНА С ОШИБКОЙ ===")
        return False

# Загружаем модель только в локальном режиме
if USE_LOCAL:
    logger.info("=== ЗАГРУЗКА МОДЕЛИ SILERO TTS ===")
    if not load_silero_model():
        logger.error("Не удалось загрузить модель Silero TTS!")
        logger.error("Бот будет работать без TTS функциональности!")
    else:
        logger.info("=== МОДЕЛЬ SILERO TTS УСПЕШНО ЗАГРУЖЕНА ===")
else:
    logger.info("Локальная модель Silero TTS не загружается (режим API).")


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

async def test_silero_tts():
    """Тестовая функция для проверки работы Silero TTS"""
    global silero_model
    
    if silero_model is None:
        logger.error("Модель Silero TTS не загружена")
        return False
    
    try:
        logger.info("Тестируем работу Silero TTS...")
        
        # Тестовый текст
        test_text = "Привет, это тестовое сообщение для проверки работы модели."
        
        # Создаем временный файл
        test_file = AUDIO_FOLDER / "test_tts.wav"
        
        # Генерируем аудио
        await run_blocking(
            silero_model.save_wav,
            text=test_text,
            speaker=SILERO_SPEAKER,
            sample_rate=SILERO_SAMPLE_RATE,
            audio_path=str(test_file)
        )
        
        # Проверяем, что файл создан
        if test_file.exists() and test_file.stat().st_size > 0:
            logger.info(f"Тестовый аудиофайл создан успешно: {test_file.stat().st_size} байт")
            
            # Удаляем тестовый файл
            cleanup_file(test_file)
            
            return True
        else:
            logger.error("Тестовый аудиофайл не был создан или пуст")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка при тестировании Silero TTS: {e}", exc_info=True)
        return False

async def check_elevenlabs_limits():
    """Проверка доступных лимитов ElevenLabs"""
    if not elevenlabs_client:
        return False, "ElevenLabs клиент не инициализирован"
    
    try:
        user_info = await run_blocking(elevenlabs_client.user.get)
        subscription = user_info.subscription
        
        # Проверяем, есть ли доступные символы
        character_count = subscription.character_count
        character_limit = subscription.character_limit
        
        logger.info(f"ElevenLabs лимиты: использовано {character_count} из {character_limit} символов")
        
        # Если осталось меньше 1000 символов, считаем что лимит исчерпан
        available_chars = character_limit - character_count
        if available_chars < 1000:
            return False, f"Осталось мало символов: {available_chars}"
        
        return True, f"Доступно символов: {available_chars}"
        
    except Exception as e:
        logger.error(f"Ошибка при проверке лимитов ElevenLabs: {e}")
        return False, f"Ошибка проверки лимитов: {str(e)}"

def cleanup_file(filepath):
    if filepath and Path(filepath).exists():
        try:
            os.remove(filepath)
            logger.debug(f"Файл успешно удален: {filepath}")
        except OSError as e:
            logger.error(f"Ошибка при удалении файла {filepath}: {e}")
    else:
        logger.debug(f"Файл для удаления не найден: {filepath}")


async def run_blocking(func, *args, **kwargs):
    logger.debug(f"Выполняю блокирующую функцию: {func.__name__}")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, partial(func, *args, **kwargs))
    logger.debug(f"Блокирующая функция {func.__name__} завершена")
    return result


# --- ФУНКЦИИ УПРАВЛЕНИЯ ПАМЯТЬЮ GPU ---
def check_gpu_status():
    """Проверка состояния GPU"""
    logger.info("=== Статус GPU ===")
    
    # Проверяем доступность CUDA
    cuda_available = torch.cuda.is_available()
    logger.info(f"torch.cuda.is_available(): {cuda_available}")
    
    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            total_memory = torch.cuda.get_device_properties(current_device).total_memory
            allocated_memory = torch.cuda.memory_allocated(current_device)
            
            logger.info(f"Доступно GPU: {device_count}")
            logger.info(f"Текущее устройство: {current_device}")
            logger.info(f"Название GPU: {device_name}")
            logger.info(f"Всего памяти: {total_memory / 1024**3:.1f} ГБ")
            logger.info(f"Выделено памяти: {allocated_memory / 1024**3:.1f} ГБ")
            logger.info(f"Свободно памяти: {(total_memory - allocated_memory) / 1024**3:.1f} ГБ")
            
                        # Проверяем версию CUDA
            cuda_version = torch.version.cuda
            logger.info(f"Версия CUDA: {cuda_version}")
            
            # Дополнительная диагностика
            try:
                capability = torch.cuda.get_device_capability(current_device)
                logger.info(f"Возможности GPU: {capability}")
            except Exception as e:
                logger.warning(f"Не удалось получить возможности GPU: {e}")
            
            return {
                'available': True,
                'device_count': device_count,
                'current_device': current_device,
                'device_name': device_name,
                'total_memory': total_memory,
                'allocated_memory': allocated_memory,
                'cuda_version': cuda_version
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о GPU: {e}")
            return {'available': False, 'error': str(e)}
    else:
        # Проверяем возможные причины недоступности CUDA
        logger.info("GPU не доступен")
        logger.info(f"USE_NVIDIA_GPU: {USE_NVIDIA_GPU}")
        logger.info(f"torch.version.cuda: {torch.version.cuda}")
        
        # Проверяем переменные окружения
        cuda_home = os.environ.get('CUDA_HOME')
        cuda_path = os.environ.get('CUDA_PATH')
        logger.info(f"CUDA_HOME: {cuda_home}")
        logger.info(f"CUDA_PATH: {cuda_path}")
        
        # Проверяем наличие CUDA в PATH
        import shutil
        nvidia_smi_path = shutil.which('nvidia-smi')
        logger.info(f"nvidia-smi доступен: {nvidia_smi_path}")
        
        return {'available': False, 'reason': 'CUDA недоступна'}

def cleanup_gpu_memory():
    """Полная очистка памяти GPU."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Память GPU успешно очищена.")
        except Exception as e:
            logger.error(f"Ошибка при очистке памяти GPU: {e}")


def check_silero_tts_availability():
    """Проверка доступности модели Silero TTS"""
    global silero_model, silero_device
    
    logger.info("=== ПРОВЕРКА ДОСТУПНОСТИ МОДЕЛИ ===")
    
    try:
        if silero_model is None:
            logger.warning("Модель Silero TTS не загружена, пытаемся перезагрузить...")
            if not load_silero_model():
                logger.error("Не удалось перезагрузить модель Silero TTS")
                return False
        
        # Проверяем, что модель не None
        if silero_model is None:
            logger.error("Модель Silero TTS все еще None после попытки загрузки")
            return False
        
        # Проверяем, что модель работает
        if silero_device and silero_device.type == 'cuda':
            # Проверяем доступность CUDA
            if not torch.cuda.is_available():
                logger.error("CUDA недоступна, но модель загружена на GPU")
                return False
            
            # Проверяем, что модель действительно на GPU
            try:
                device_check = next(silero_model.parameters()).device
                if device_check.type != 'cuda':
                    logger.warning(f"Модель загружена на {device_check}, но должна быть на {silero_device}")
                    return False
            except Exception as e:
                logger.warning(f"Не удалось проверить устройство модели: {e}")
        
        # Дополнительная проверка модели
        try:
            logger.info(f"Дополнительная проверка модели:")
            logger.info(f"  - Тип модели: {type(silero_model)}")
            logger.info(f"  - Модель не None: {silero_model is not None}")
            logger.info(f"  - Имеет метод save_wav: {hasattr(silero_model, 'save_wav')}")
            if hasattr(silero_model, 'device'):
                logger.info(f"  - Атрибут device: {silero_model.device}")
            logger.info(f"  - Устройство silero_device: {silero_device}")
        except Exception as e:
            logger.warning(f"Не удалось выполнить дополнительную проверку модели: {e}")
        
        logger.info(f"Модель Silero TTS доступна на устройстве: {silero_device}")
        logger.info("=== ПРОВЕРКА ДОСТУПНОСТИ МОДЕЛИ ЗАВЕРШЕНА УСПЕШНО ===")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при проверке доступности модели Silero TTS: {e}")
        logger.error("=== ПРОВЕРКА ДОСТУПНОСТИ МОДЕЛИ ЗАВЕРШЕНА С ОШИБКОЙ ===")
        return False


# --- ЛОГИКА РАСПОЗНАВАНИЯ РЕЧИ ---
async def transcribe_audio_groq_with_retry(audio_path: Path, max_retries: int = 2) -> str:
    logger.info("=== НАЧАЛО TRANSCRIBE_AUDIO_GROQ_WITH_RETRY ===")
    
    if not groq_client:
        raise ConnectionError("Клиент Groq API не инициализирован.")
    
    audio_duration = AudioSegment.from_file(audio_path).duration_seconds
    logger.info(f"Длительность аудио: {audio_duration:.2f} секунд")
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Попытка {attempt + 1} из {max_retries + 1}")
            
            with open(audio_path, "rb") as audio_file:
                # Разные параметры для разных попыток
                if attempt == 0:
                    # Первая попытка - стандартные параметры
                    transcription = await run_blocking(
                        groq_client.audio.transcriptions.create,
                        file=(audio_path.name, audio_file.read()),
                        model="whisper-large-v3",
                        response_format="verbose_json",  # Используем verbose_json
                        language="ru",
                        temperature=0.0,
                        prompt="Это разговорная русская речь. Пожалуйста, распознай все слова точно.",
                        timestamp_granularities=["word", "segment"]
                    )
                elif attempt == 1:
                    # Вторая попытка - с другим промптом
                    transcription = await run_blocking(
                        groq_client.audio.transcriptions.create,
                        file=(audio_path.name, audio_file.read()),
                        model="whisper-large-v3",
                        response_format="verbose_json",  # Используем verbose_json
                        language="ru",
                        temperature=0.1,
                        prompt="Внимательно слушай и записывай каждое слово, включая паузы, междометия и повторения. Не сокращай текст.",
                        timestamp_granularities=["word", "segment"]
                    )
                else:
                    # Третья попытка - с турбо моделью для сравнения
                    transcription = await run_blocking(
                        groq_client.audio.transcriptions.create,
                        file=(audio_path.name, audio_file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",  # Используем verbose_json
                        language="ru",
                        temperature=0.0,
                        prompt="Запиши всё, что слышишь, без сокращений.",
                        timestamp_granularities=["word", "segment"]
                    )
                
                # Проверяем, что transcription - это словарь (verbose_json формат)
                if isinstance(transcription, dict):
                    # Получаем текст из поля text
                    text = transcription.get("text", "")
                    
                    # Проверяем наличие сегментов для анализа покрытия
                    segments = transcription.get("segments", [])
                    if segments:
                        total_duration = 0
                        word_count = 0
                        
                        for segment in segments:
                            if "end" in segment:
                                total_duration = max(total_duration, segment["end"])
                            if "words" in segment:
                                word_count += len(segment["words"])
                        
                        # Проверяем качество распознавания
                        coverage_ratio = total_duration / audio_duration if audio_duration > 0 else 0
                        words_per_second = word_count / total_duration if total_duration > 0 else 0
                        
                        logger.info(f"Попытка {attempt + 1}: покрытие {coverage_ratio*100:.1f}%, слов {word_count}, скорость {words_per_second:.1f} слов/сек")
                        
                        # Если покрытие хорошее и скорость нормальная, возвращаем результат
                        if coverage_ratio >= 0.85 and words_per_second >= 1.0:
                            logger.info(f"Используем результат попытки {attempt + 1}")
                            logger.info("=== TRANSCRIBE_AUDIO_GROQ_WITH_RETRY ЗАВЕРШЕН УСПЕШНО ===")
                            return text
                    else:
                        # Если сегментов нет, но текст есть, проверяем длину текста
                        if len(text) > 10:  # Минимальная длина текста
                            logger.info(f"Сегменты отсутствуют, но текст получен: {len(text)} символов")
                            logger.info("=== TRANSCRIBE_AUDIO_GROQ_WITH_RETRY ЗАВЕРШЕН УСПЕШНО ===")
                            return text
                        else:
                            logger.warning(f"Текст слишком короткий: {len(text)} символов")
                else:
                    # Если transcription - это строка (text формат)
                    text = str(transcription)
                    logger.info(f"Получен текст в строковом формате: {len(text)} символов")
                    
                    # Если текст достаточно длинный, возвращаем его
                    if len(text) > 10:
                        logger.info("=== TRANSCRIBE_AUDIO_GROQ_WITH_RETRY ЗАВЕРШЕН УСПЕШНО ===")
                        return text
                
                # Если это последняя попытка, возвращаем результат в любом случае
                if attempt == max_retries:
                    logger.warning(f"Используем результат последней попытки несмотря на низкое качество")
                    logger.info("=== TRANSCRIBE_AUDIO_GROQ_WITH_RETRY ЗАВЕРШЕН С ПРЕДУПРЕЖДЕНИЕМ ===")
                    return text if isinstance(transcription, dict) else str(transcription)
                
        except Exception as e:
            logger.error(f"Ошибка в попытке {attempt + 1}: {e}")
            if attempt == max_retries:
                raise
    
    logger.info("=== TRANSCRIBE_AUDIO_GROQ_WITH_RETRY ЗАВЕРШЕН УСПЕШНО ===")
    return ""


# --- КЛАССЫ ДЛЯ ОПТИМИЗАЦИИ ОЗВУЧКИ ---
class SmartCache:
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = asyncio.Lock()
        logger.info(f"Инициализирован кеш для TTS размером {max_size} элементов.")

    def _get_key(self, text: str, speaker: str) -> str:
        content = f"{text}|{speaker}"
        return hashlib.md5(content.encode()).hexdigest()

    async def get(self, text: str, speaker: str):
        key = self._get_key(text, speaker)
        async with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                logger.debug(f"Кеш TTS: Найден хит для ключа {key[:8]}...")
                return self.cache[key]
            else:
                logger.debug(f"Кеш TTS: Промах для ключа {key[:8]}...")
        return None

    async def put(self, text: str, speaker: str, audio_tensor):
        key = self._get_key(text, speaker)
        async with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                logger.debug(f"Кеш TTS: Удален старый элемент {oldest_key[:8]}...")

            self.cache[key] = audio_tensor
            self.access_times[key] = time.time()
            logger.debug(f"Кеш TTS: Добавлен новый элемент {key[:8]}... (размер кеша: {len(self.cache)})")


smart_tts_cache = SmartCache()


class TextPreprocessor:
    def __init__(self):
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\sа-яА-ЯёЁa-zA-Z0-9\.\,\!\?\;\:\-]')
        logger.info("Инициализирован препроцессор текста для TTS (с сохранением цифр).")

    def preprocess(self, text: str) -> str:
        original_length = len(text)
        text = self.whitespace_pattern.sub(' ', text).strip()
        text = self.special_chars_pattern.sub('', text)
        if len(text) > 1000:
            text = text[:1000]
            logger.debug(f"Текст обрезан с {original_length} до 1000 символов")
        
        logger.debug(f"Текст препроцессирован: {original_length} -> {len(text)} символов")
        return text


# --- ФУНКЦИИ ОЗВУЧКИ ТЕКСТА ---
async def text_to_speech_elevenlabs(text: str) -> bytes:
    """Генерация речи через ElevenLabs API с автоматическим переключением ключей"""
    if not elevenlabs_manager:
        return None
    
    max_retries = len(elevenlabs_manager.api_keys)
    
    for attempt in range(max_retries):
        try:
            # Получаем доступный ключ
            key_index, client, async_client, message = await elevenlabs_manager.get_available_key()
            
            if key_index is None:
                logger.error("Нет доступных ключей ElevenLabs")
                return None
            
            logger.info(f"Использую ключ {key_index + 1} для генерации речи")
            
            # Генерируем аудио
            def generate_audio():
                response = client.text_to_speech.convert(
                    voice_id=ELEVENLABS_MALE_VOICE_ID,
                    optimize_streaming_latency="0",
                    output_format=ELEVENLABS_OUTPUT_FORMAT,
                    text=text,
                    model_id=ELEVENLABS_MODEL_ID,
                    voice_settings=VoiceSettings(
                        stability=0.0,
                        similarity_boost=1.0,
                        style=0.0,
                        use_speaker_boost=True,
                    ),
                )
                
                audio_bytes = b""
                for chunk in response:
                    if chunk:
                        audio_bytes += chunk
                return audio_bytes
            
            # Запускаем синхронный код в отдельном потоке
            audio_bytes = await run_blocking(generate_audio)
            
            if audio_bytes:
                # Записываем использование
                characters_used = len(text)
                await elevenlabs_manager.record_usage(key_index, characters_used)
                
                logger.info(f"Успешно сгенерировано {len(audio_bytes)} байт аудио, использовано {characters_used} символов")
                return audio_bytes
            else:
                logger.warning(f"Ключ {key_index + 1} вернул пустой ответ, пробую следующий")
                continue
                
        except Exception as e:
            logger.error(f"Ошибка при использовании ключа {key_index + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error("Все ключи ElevenLabs недоступны")
                return None
    
    return None

async def text_to_speech_silero(text: str) -> bytes:
    """Генерация речи с приоритетом ElevenLabs, затем Silero TTS"""
    
    # Сначала пробуем ElevenLabs
    if elevenlabs_client:
        logger.info("Пробую сгенерировать речь через ElevenLabs...")
        elevenlabs_audio = await text_to_speech_elevenlabs(text)
        if elevenlabs_audio:
            logger.info("Речь успешно сгенерирована через ElevenLabs")
            return elevenlabs_audio
        else:
            logger.info("ElevenLabs недоступен, переключаюсь на Silero TTS")
    
    # Если ElevenLabs недоступен, используем Silero TTS только в локальном режиме
    if not USE_LOCAL:
        logger.warning("Локальный TTS отключен. Не удалось сгенерировать речь.")
        return None

    if not silero_model or not silero_device:
        logger.error("Модель Silero TTS не загружена, озвучка невозможна.")
        return None
    
    text_preprocessor = TextPreprocessor()
    processed_text = text_preprocessor.preprocess(text)
    
    if not processed_text:
        return None
    
    # Проверяем кеш
    cached_tensor = await smart_tts_cache.get(processed_text, SILERO_SPEAKER)
    if cached_tensor is not None:
        logger.debug(f"Текст найден в кеше, пропускаю генерацию")
        # Используем save_wav для сохранения аудио
        output_path = AUDIO_FOLDER / f"cached_audio_{hash(processed_text)}.wav"
        try:
            await run_blocking(
                silero_model.save_wav,
                text=processed_text,
                speaker=SILERO_SPEAKER,
                sample_rate=SILERO_SAMPLE_RATE,
                audio_path=str(output_path)
            )
            # Читаем сохраненный файл и конвертируем в байты
            with open(output_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            # Удаляем временный файл
            cleanup_file(output_path)
            return audio_bytes
        except Exception as e:
            logger.error(f"Ошибка при сохранении кешированного аудио: {e}")
    
    try:
        with torch.no_grad(), torch.inference_mode():
            # Генерируем аудио с помощью save_wav (как в рабочем коде)
            output_path = AUDIO_FOLDER / f"generated_audio_{hash(processed_text)}.wav"
            await run_blocking(
                silero_model.save_wav,
                text=processed_text,
                speaker=SILERO_SPEAKER,
                sample_rate=SILERO_SAMPLE_RATE,
                audio_path=str(output_path)
            )
            
            # Читаем сохраненный файл
            with open(output_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # Сохраняем в кеш (если нужно)
            try:
                audio_tensor = await run_blocking(
                    silero_model.apply_tts,
                    text=processed_text,
                    speaker=SILERO_SPEAKER,
                    sample_rate=SILERO_SAMPLE_RATE
                )
                await smart_tts_cache.put(processed_text, SILERO_SPEAKER, audio_tensor)
            except Exception as e:
                logger.warning(f"Не удалось сгенерировать тензор для кеширования: {e}")
            
            # Удаляем временный файл
            cleanup_file(output_path)
            
            return audio_bytes
    
    except Exception as e:
        logger.error(f"Ошибка при генерации речи: {e}")
        return None


# --- ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ ЗАДАЧ ---
async def process_voice_to_text(bot: Bot, voice: types.Voice, chat_id: int, message_id: int, user_id: int):
    logger.info("=== НАЧАЛО PROCESS_VOICE_TO_TEXT ===")
    files_to_clean = []
    try:
        logger.info(f"Начинаю обработку голосового сообщения от пользователя {user_id}")
        
        # Скачиваем голосовое сообщение
        voice_info = await bot.get_file(voice.file_id)
        voice_path = AUDIO_FOLDER / f"{voice_info.file_unique_id}.ogg"
        await bot.download_file(voice_info.file_path, destination=voice_path)
        files_to_clean.append(voice_path)
        
        # Конвертируем в формат, подходящий для Whisper
        wav_path = voice_path.with_suffix('.wav')
        audio = AudioSegment.from_ogg(voice_path)
        
        # Улучшаем качество аудио перед распознаванием
        audio = audio.set_frame_rate(16000)  # Устанавливаем частоту 16 кГц как рекомендовано
        audio = audio.set_channels(1)  # Моно звук
        audio.export(wav_path, format="wav")
        files_to_clean.append(wav_path)
        
        # Распознаем речь с улучшенной функцией
        recognized_text = await transcribe_audio_groq_with_retry(wav_path)
        
        if not recognized_text.strip():
            await bot.send_message(chat_id, "❌ Не удалось распознать речь в голосовом сообщении.")
            return
        
        # Отправляем распознанный текст
        await bot.send_message(
            chat_id,
            f"📝 Распознанный текст:\n{recognized_text}"
        )
        
        logger.info("=== PROCESS_VOICE_TO_TEXT ЗАВЕРШЕН УСПЕШНО ===")

    except Exception as e:
        logger.exception(f"КРИТИЧЕСКАЯ ОШИБКА в process_voice_to_text для чата {chat_id}")
        error_message = f"❌ Произошла критическая ошибка: `{str(e)[:250]}`"
        try:
            await bot.send_message(chat_id, error_message)
        except TelegramBadRequest:
            await bot.send_message(chat_id, "❌ Произошла критическая ошибка.")
        logger.error("=== PROCESS_VOICE_TO_TEXT ЗАВЕРШЕН С ОШИБКОЙ ===")
    finally:
        logger.info(f"Начинаю очистку {len(files_to_clean)} временных файлов...")
        for f in set(files_to_clean):
            cleanup_file(f)
        if USE_NVIDIA_GPU:
            logger.debug("Очистка памяти GPU после завершения задачи...")
            cleanup_gpu_memory()


async def process_text_to_voice(bot: Bot, text: str, chat_id: int, user_id: int):
    logger.info("=== НАЧАЛО PROCESS_TEXT_TO_VOICE ===")
    try:
        logger.info(f"Начинаю обработку текстового сообщения от пользователя {user_id}")

        # Проверяем доступность модели TTS
        if not check_silero_tts_availability():
            logger.error("Модель TTS недоступна в process_text_to_voice")
            await bot.send_message(chat_id, "❌ Модель TTS недоступна. Попробуйте команду /reload_tts")
            return

        # Озвучиваем текст
        voice_data = await text_to_speech_silero(text)

        if not voice_data:
            await bot.send_message(chat_id, "❌ Не удалось создать голосовое сообщение.")
            return

        # Отправляем голосовое сообщение
        await bot.send_voice(
            chat_id,
            voice=types.BufferedInputFile(voice_data, filename="voice.ogg")
        )
        
        logger.info("=== PROCESS_TEXT_TO_VOICE ЗАВЕРШЕН УСПЕШНО ===")

    except Exception as e:
        logger.exception(f"КРИТИЧЕСКАЯ ОШИБКА в process_text_to_voice для чата {chat_id}")
        error_message = f"❌ Произошла критическая ошибка: `{str(e)[:250]}`"
        try:
            await bot.send_message(chat_id, error_message)
        except TelegramBadRequest:
            await bot.send_message(chat_id, "❌ Произошла критическая ошибка.")
        logger.error("=== PROCESS_TEXT_TO_VOICE ЗАВЕРШЕН С ОШИБКОЙ ===")
    finally:
        if USE_NVIDIA_GPU:
            logger.debug("Очистка памяти GPU после завершения задачи...")
            cleanup_gpu_memory()


async def main():
    logger.info("=== НАЧАЛО РАБОТЫ БОТА ===")
    
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="Markdown"))
    dp = Dispatcher()
    
    # Проверка состояния GPU и TTS только в локальном режиме
    if USE_LOCAL:
        # Проверка состояния GPU
        logger.info("=== ПРОВЕРКА GPU ===")
        gpu_status = check_gpu_status()
        logger.info(f"Статус GPU: {gpu_status}")
        
        # Проверка доступности модели Silero TTS
        logger.info("=== ПРОВЕРКА ДОСТУПНОСТИ МОДЕЛИ SILERO TTS ===")
        if not check_silero_tts_availability():
            logger.error("КРИТИЧЕСКАЯ ОШИБКА: Модель Silero TTS не доступна!")
            await bot.send_message(ADMIN_ID, "❌ КРИТИЧЕСКАЯ ОШИБКА: Модель Silero TTS не доступна! Бот не может работать.")
            logger.error("Бот завершает работу из-за недоступности модели TTS")
            return
        else:
            logger.info("=== МОДЕЛЬ SILERO TTS ДОСТУПНА ===")
            
            # Тестируем модель перед запуском бота, только если включен debug
            if DEBUG:
                logger.info("=== ТЕСТИРОВАНИЕ МОДЕЛИ SILERO TTS ===")
                logger.info("Тестируем работу Silero TTS... -  только если включен debug")
                if not await test_silero_tts():
                    logger.error("КРИТИЧЕСКАЯ ОШИБКА: Тест модели Silero TTS не пройден!")
                    await bot.send_message(ADMIN_ID, "❌ КРИТИЧЕСКАЯ ОШИБКА: Тест модели Silero TTS не пройден! Бот не может работать.")
                    logger.error("Бот завершает работу из-за неудачного теста модели TTS")
                    return
                else:
                    logger.info("=== ТЕСТ МОДЕЛИ SILERO TTS ПРОЙДЕН УСПЕШНО ===")
    else:
        # В режиме API-only, нам не нужен GPU или локальная TTS
        gpu_status = {'available': False, 'reason': 'API-only mode'}
        logger.info("Работа в режиме API-only. Проверка GPU и локальной TTS пропущена.")

    @dp.message(Command("start", "help"))
    async def handle_start(message: types.Message):
        # Проверяем статус модели TTS
        tts_status = "✅ Доступна" if not USE_LOCAL or check_silero_tts_availability() else "❌ Недоступна"
        elevenlabs_status = "✅ Доступен" if elevenlabs_available else "❌ Недоступен"
        
        await message.reply(
            "Привет! Я бот для перевода голосовых сообщений в текст и текста в голосовые сообщения.\n\n"
            "**Как я работаю:**\n"
            "• Когда папа пишет текст — я озвучиваю его голосом\n"
            "• Когда вы или папа отправляете голосовое сообщение — я распознаю речь и отправляю текстом\n\n"
            f"Статус GPU: {'Доступен' if gpu_status['available'] else 'Не доступен'}\n"
            f"Статус TTS: {tts_status}\n"
            f"Статус ElevenLabs: {elevenlabs_status}\n"
            "**Команды:**\n"
            "• `/gpu_status` - проверить статус GPU\n"
            "• `/cleanup_gpu` - очистить память GPU\n"
            "• `/reload_tts` - перезагрузить модель TTS\n"
            "• `/force_reload_tts` - принудительно перезагрузить модель TTS\n"
            "• `/test_tts` - протестировать TTS\n"
            "• `/switch_device` - переключить GPU/CPU\n"
            "• `/system_info` - информация о системе\n"
            "• `/elevenlabs_status` - проверить статус ElevenLabs\n"
            "• `/elevenlabs_usage` - детальная статистика использования\n"
            "• `/reset_elevenlabs_stats` - сбросить статистику (админ)\n"
            "• `/versions` - версии библиотек\n"
            "• `/admin_id` - узнать свой ID\n\n"
            "Все сообщения обрабатываются на русском языке."
        )

    @dp.message(Command("admin_id"))
    async def handle_admin_id(message: types.Message):
        # Отладочная команда для проверки ID
        try:
            await message.reply(
                f"Ваш ID: {message.from_user.id}\n"
                f"ADMIN_ID: {ADMIN_ID}",
                parse_mode="Markdown"  # Указываем формат
            )
        except TelegramBadRequest:
            # Если Markdown не работает, используем обычный текст
            await message.reply(
                f"Ваш ID: {message.from_user.id}\n"
                f"ADMIN_ID: {ADMIN_ID}"
            )
        except Exception as e:
            logger.error(f"Ошибка при получении ID: {e}")
            await message.reply(f"❌ Ошибка при получении ID: {str(e)}")

    if USE_LOCAL:
        @dp.message(Command("gpu_status"))
        async def handle_gpu_status(message: types.Message):
            # Команда для проверки статуса GPU
            try:
                current_gpu_status = check_gpu_status()
                if current_gpu_status['available']:
                    status_text = (
                        f"🟢 **Статус GPU: Доступен**\n\n"
                        f"**Устройство:** {current_gpu_status['device_name']}\n"
                        f"**Всего памяти:** {current_gpu_status['total_memory'] / 1024**3:.1f} ГБ\n"
                        f"**Выделено памяти:** {current_gpu_status['allocated_memory'] / 1024**3:.1f} ГБ\n"
                        f"**Свободно памяти:** {(current_gpu_status['total_memory'] - current_gpu_status['allocated_memory']) / 1024**3:.1f} ГБ\n"
                        f"**Версия CUDA:** {current_gpu_status.get('cuda_version', 'Неизвестно')}"
                    )
                else:
                    reason = current_gpu_status.get('reason', 'Неизвестно')
                    error = current_gpu_status.get('error', '')
                    status_text = f"🔴 **Статус GPU: Не доступен**\n\n**Причина:** {reason}"
                    if error:
                        status_text += f"\n**Ошибка:** {error}"
                    status_text += "\n\nИспользуется CPU для обработки."
                
                await message.reply(status_text, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Ошибка при получении статуса GPU: {e}")
                await message.reply(f"❌ Ошибка при получении статуса GPU: {str(e)}")

        @dp.message(Command("cleanup_gpu"))
        async def handle_cleanup_gpu(message: types.Message):
            # Команда для принудительной очистки памяти GPU
            try:
                if USE_NVIDIA_GPU and silero_device and silero_device.type == 'cuda':
                    cleanup_gpu_memory()
                    await message.reply("🧹 Память GPU успешно очищена!")
                else:
                    await message.reply("ℹ️ GPU не используется, очистка не требуется.")
            except Exception as e:
                logger.error(f"Ошибка при очистке памяти GPU: {e}")
                await message.reply(f"❌ Ошибка при очистке памяти GPU: {str(e)}")

        @dp.message(Command("reload_tts"))
        async def handle_reload_tts(message: types.Message):
            # Команда для перезагрузки модели TTS
            try:
                global silero_model, silero_device
                
                await message.reply("🔄 Перезагружаю модель TTS...")
                
                # Очищаем память GPU перед перезагрузкой
                if USE_NVIDIA_GPU and silero_device and silero_device.type == 'cuda':
                    cleanup_gpu_memory()
                
                # Перезагружаем модель
                if load_silero_model():
                    logger.info("Модель TTS успешно перезагружена")
                    await message.reply("✅ Модель TTS успешно перезагружена!")
                else:
                    logger.error("Не удалось перезагрузить модель TTS")
                    await message.reply("❌ Не удалось перезагрузить модель TTS!")
                    
            except Exception as e:
                logger.error(f"Ошибка при перезагрузке модели TTS: {e}")
                await message.reply(f"❌ Ошибка при перезагрузке модели TTS: {str(e)}")

        @dp.message(Command("versions"))
        async def handle_versions(message: types.Message):
            # Команда для проверки версий библиотек
            try:
                versions_text = (
                    f"📚 **Версии библиотек:**\n\n"
                    f"**PyTorch:** {torch.__version__}\n"
                    f"**CUDA:** {torch.version.cuda}\n"
                    f"**cuDNN:** {torch.backends.cudnn.version()}\n"
                    f"**Python:** {sys.version.split()[0]}\n"
                    f"**USE_NVIDIA_GPU:** {USE_NVIDIA_GPU}\n"
                    f"**CUDA доступна:** {torch.cuda.is_available()}"
                )
                await message.reply(versions_text, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Ошибка при получении версий: {e}")
                await message.reply(f"❌ Ошибка при получении версий: {str(e)}")

        @dp.message(Command("test_tts"))
        async def handle_test_tts(message: types.Message):
            # Команда для тестирования TTS
            try:
                # Проверяем доступность модели TTS
                if not check_silero_tts_availability():
                    await message.reply("❌ Модель TTS недоступна! Попробуйте команду /reload_tts")
                    return
                
                sent_msg = await message.reply("🧪 Тестирую TTS...")
                
                # Используем нашу функцию тестирования
                if await test_silero_tts():
                    await message.reply("✅ TTS работает корректно!")
                else:
                    await message.reply("❌ TTS не работает!")
                
                # Удаляем сообщение о тестировании
                try:
                    await bot.delete_message(message.chat.id, sent_msg.message_id)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Ошибка при тестировании TTS: {e}")
                await message.reply(f"❌ Ошибка при тестировании TTS: {str(e)}")

        @dp.message(Command("switch_device"))
        async def handle_switch_device(message: types.Message):
            # Команда для переключения между GPU и CPU
            try:
                global silero_model, silero_device
                
                if silero_device and silero_device.type == 'cuda':
                    # Переключаемся на CPU
                    silero_device = torch.device('cpu')
                    if silero_model is not None:
                        silero_model = silero_model.to(silero_device)
                        logger.info("Переключились на CPU")
                        await message.reply("🔄 Переключились на CPU")
                    else:
                        await message.reply("❌ Модель TTS не загружена")
                else:
                    # Переключаемся на GPU
                    if torch.cuda.is_available():
                        silero_device = torch.device('cuda')
                        if silero_model is not None:
                            silero_model = silero_model.to(silero_device)
                            logger.info("Переключились на GPU")
                            await message.reply("🔄 Переключились на GPU")
                        else:
                            await message.reply("❌ Модель TTS не загружена")
                    else:
                        await message.reply("❌ GPU недоступен")
                        
                await message.reply("🔄 Переключились на устройство")
            except Exception as e:
                logger.error(f"Ошибка при переключении устройства: {e}")
                await message.reply(f"❌ Ошибка при переключении устройства: {str(e)}")

        @dp.message(Command("force_reload_tts"))
        async def handle_force_reload_tts(message: types.Message):
            # Команда для принудительной перезагрузки модели TTS
            try:
                global silero_model, silero_device
                
                await message.reply("🔄 Принудительно перезагружаю модель TTS...")
                
                # Очищаем память GPU
                if USE_NVIDIA_GPU and silero_device and silero_device.type == 'cuda':
                    cleanup_gpu_memory()
                
                # Сбрасываем глобальные переменные
                silero_model = None
                silero_device = None
                
                # Перезагружаем модель
                if load_silero_model():
                    logger.info("Модель TTS успешно принудительно перезагружена")
                    await message.reply("✅ Модель TTS успешно принудительно перезагружена!")
                else:
                    logger.error("Не удалось принудительно перезагрузить модель TTS")
                    await message.reply("❌ Не удалось принудительно перезагрузить модель TTS!")
                    
            except Exception as e:
                logger.error(f"Ошибка при принудительной перезагрузке модели TTS: {e}")
                await message.reply(f"❌ Ошибка при принудительной перезагрузке модели TTS: {str(e)}")

    @dp.message(Command("elevenlabs_status"))
    async def handle_elevenlabs_status(message: types.Message):
        """Команда для проверки статуса ElevenLabs"""
        try:
            if not elevenlabs_manager:
                await message.reply("❌ ElevenLabs менеджер не инициализирован")
                return
            
            status_text = "🎙️ **Статус ElevenLabs:**\n\n"
            
            # Проверяем каждый ключ
            for i, api_key in enumerate(elevenlabs_manager.api_keys):
                is_available, message_text = await elevenlabs_manager.check_key_limits(i)
                status_icon = "✅" if is_available else "❌"
                status_text += f"{status_icon} **Ключ {i+1}:** {message_text}\n"
            
            # Добавляем статистику за текущий месяц
            current_month = elevenlabs_manager.get_current_month_key()
            monthly_usage = elevenlabs_manager.get_monthly_usage(current_month)
            
            if monthly_usage:
                status_text += f"\n📊 **Использование за {current_month}:**\n"
                total_chars = sum(monthly_usage.values())
                status_text += f"Всего символов: {total_chars:,}\n"
                for key_name, chars in monthly_usage.items():
                    status_text += f"{key_name}: {chars:,} символов\n"
            
            status_text += f"\n**Текущий активный ключ:** {elevenlabs_manager.current_key_index + 1}"
            
            await message.reply(status_text, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Ошибка при проверке статуса ElevenLabs: {e}")
            await message.reply(f"❌ Ошибка при проверке статуса ElevenLabs: {str(e)}")

    @dp.message(Command("elevenlabs_usage"))
    async def handle_elevenlabs_usage(message: types.Message):
        """Команда для детальной статистики использования ElevenLabs"""
        try:
            if not elevenlabs_manager:
                await message.reply("❌ ElevenLabs менеджер не инициализирован")
                return
            
            all_stats = elevenlabs_manager.get_all_usage_stats()
            
            if not all_stats:
                await message.reply("📊 Нет статистики использования ElevenLabs")
                return
            
            usage_text = "📊 **Детальная статистика ElevenLabs:**\n\n"
            
            # Группируем по месяцам
            monthly_data = {}
            for usage in all_stats.values():
                for month, chars in usage.monthly_usage.items():
                    if month not in monthly_data:
                        monthly_data[month] = {}
                    monthly_data[month][f"Ключ {usage.key_index + 1}"] = chars
            
            # Выводим статистику по месяцам
            for month in sorted(monthly_data.keys(), reverse=True):
                usage_text += f"**📅 {month}:**\n"
                month_total = sum(monthly_data[month].values())
                usage_text += f"Всего: {month_total:,} символов\n"
                
                for key_name, chars in monthly_data[month].items():
                    usage_text += f"  {key_name}: {chars:,} символов\n"
                usage_text += "\n"
            
            await message.reply(usage_text, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Ошибка при получении статистики ElevenLabs: {e}")
            await message.reply(f"❌ Ошибка при получении статистики: {str(e)}")

    @dp.message(Command("reset_elevenlabs_stats"))
    async def handle_reset_elevenlabs_stats(message: types.Message):
        """Команда для сброса статистики (только для админа)"""
        if message.from_user.id != ADMIN_ID:
            await message.reply("❌ Эта команда доступна только администратору")
            return
        
        try:
            if not elevenlabs_manager:
                await message.reply("❌ ElevenLabs менеджер не инициализирован")
                return
            
            elevenlabs_manager.reset_monthly_stats()
            await message.reply("📊 Статистика использования ElevenLabs сброшена")
            
        except Exception as e:
            logger.error(f"Ошибка при сбросе статистики: {e}")
            await message.reply(f"❌ Ошибка при сбросе статистики: {str(e)}")

    @dp.message(Command("system_info"))
    async def handle_system_info(message: types.Message):
        # Команда для получения информации о системе
        try:
            import platform
            
            system_info = (
                f"💻 **Информация о системе:**\n\n"
                f"**ОС:** {platform.system()} {platform.release()}\n"
                f"**Архитектура:** {platform.machine()}\n"
                f"**Процессор:** {platform.processor()}\n"
                f"**Python:** {platform.python_version()}\n"
                f"**PyTorch:** {torch.__version__}\n"
                f"**CUDA доступна:** {torch.cuda.is_available()}\n"
                f"**USE_NVIDIA_GPU:** {USE_NVIDIA_GPU}\n"
                f"**Текущее устройство TTS:** {silero_device.type if silero_device else 'Не определено'}"
            )
            await message.reply(system_info, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Ошибка при получении информации о системе: {e}")
            await message.reply(f"❌ Ошибка при получении информации о системе: {str(e)}")

    @dp.message(F.voice)
    async def handle_voice(message: types.Message):
        # Обрабатываем голосовые сообщения от всех участников
        logger.info(f"Получено голосовое сообщение от пользователя {message.from_user.id}")
        sent_msg = await message.reply("🔄 Обрабатываю голосовое сообщение...")
        await process_voice_to_text(bot, message.voice, message.chat.id, sent_msg.message_id, message.from_user.id)
        # Удаляем сообщение о обработке
        try:
            await bot.delete_message(message.chat.id, sent_msg.message_id)
        except Exception as e:
            logger.warning(f"Не удалось удалить сообщение о обработке: {e}")

    @dp.message(F.text & ~F.command)
    async def handle_text(message: types.Message):
        # Добавляем логирование для отладки
        logger.info(f"Получено текстовое сообщение от пользователя {message.from_user.id}")
        logger.info(f"ADMIN_ID: {ADMIN_ID}, Сравнение: {message.from_user.id == ADMIN_ID}")

        # Озвучиваем только сообщения от папы (ADMIN_ID)
        if message.from_user.id == ADMIN_ID:
            logger.info(f"Сообщение от ADMIN_ID, начинаю озвучку")
            sent_msg = await message.reply("🔄 Озвучиваю сообщение...")
            await process_text_to_voice(bot, message.text, message.chat.id, message.from_user.id)
            # Удаляем сообщение о обработке
            try:
                await bot.delete_message(message.chat.id, sent_msg.message_id)
            except Exception as e:
                logger.warning(f"Не удалось удалить сообщение о озвучке: {e}")
        else:
            logger.info(f"Сообщение не от ADMIN_ID, игнорирую")

    logger.info("=== ЗАПУСК БОТА ДЛЯ ПЕРЕВОДА ГОЛОСОВЫХ СООБЩЕНИЙ ===")

    if USE_NVIDIA_GPU and silero_device and silero_device.type == 'cuda':
        logger.info("--- Ускорение озвучки (CUDA) ВКЛЮЧЕНО ---")
    else:
        logger.info("--- Ускорение с помощью NVIDIA GPU ВЫКЛЮЧЕНО ---")

    if not DEBUG:
        logger.info(f"Очистка рабочей папки {AUDIO_FOLDER}...")
        for f in AUDIO_FOLDER.glob('*'):
            if f.is_file(): cleanup_file(f)
    else:
        logger.warning("[DEBUG] Режим отладки включен. Временные файлы не будут удаляться при старте.")

    try:
        logger.info("=== ЗАПУСК ПОЛЛИНГА БОТА ===")
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        logger.info("Начало процедуры корректного завершения работы...")

        await bot.session.close()
        logger.info("Сессия бота закрыта.")

        if USE_NVIDIA_GPU:
            logger.info("Финальная очистка памяти GPU...")
            cleanup_gpu_memory()

        logger.info("=== ОСТАНОВКА БОТА ===")
        logger.info("=== БОТ ЗАВЕРШЕН ===")
        logger.info("=== РАБОТА БОТА ЗАВЕРШЕНА ===")
        logger.info("=== ВСЕ РЕСУРСЫ ОСВОБОЖДЕНЫ ===")


if __name__ == '__main__':
    logger.info("=== ЗАПУСК ПРОГРАММЫ ===")
    
    # Инициализация менеджера ElevenLabs перед запуском бота
    initialize_elevenlabs_manager()
    
    try:
        asyncio.run(main())
        logger.info("=== ПРОГРАММА ЗАВЕРШЕНА УСПЕШНО ===")
    except (KeyboardInterrupt, SystemExit) as e:
        logger.info(f"Завершение работы: {e}")
        logger.info("=== ПРОГРАММА ЗАВЕРШЕНА ПОЛЬЗОВАТЕЛЕМ ===")
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: {e}", exc_info=True)
        logger.error("=== ПРОГРАММА ЗАВЕРШЕНА С ОШИБКОЙ ===")