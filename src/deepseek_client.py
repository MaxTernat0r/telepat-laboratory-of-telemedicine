import os
import json
import aiohttp
import asyncio
from loguru import logger
from typing import Optional, Union
from dotenv import load_dotenv
from pathlib import Path
import ssl
from PIL import Image
import io
import base64


class DeepSeekClient:
    """Клиент для взаимодействия с DeepSeek API"""
    
    def __init__(self):
        """Инициализация клиента DeepSeek"""
        load_dotenv()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
        if not self.api_key:
            logger.error("API ключ DeepSeek не найден в переменных окружения (DEEPSEEK_API_KEY)")
            self.is_available = False
            return
            
        # Проверяем доступность API при инициализации
        self.is_available = False
        asyncio.get_event_loop().run_until_complete(self._init_check())
        
    async def _init_check(self):
        self.is_available = await self._check_api_availability()
        
    async def _check_api_availability(self) -> bool:
        """Проверяет доступность API и валидность ключа"""
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                async with session.get(f"{self.base_url}/models", headers=headers) as response:
                    if response.status == 200:
                        logger.info("DeepSeek API доступен и ключ валиден")
                        return True
                    elif response.status == 401:
                        logger.error("Неверный API ключ DeepSeek")
                        return False
                    else:
                        logger.error(f"DeepSeek API недоступен. Статус: {response.status}")
                        return False
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка при проверке доступности DeepSeek API: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при проверке DeepSeek API: {str(e)}")
            return False

    async def analyze_medical_report(self, image_path: str) -> Union[str, None]:
        """Отправляет медицинский отчет (изображение) на анализ в DeepSeek"""
        if not self.is_available:
            logger.error("DeepSeek API недоступен. Проверьте API ключ и доступность сервиса.")
            return None

        if not Path(image_path).exists():
            logger.error(f"Файл изображения для анализа не найден: {image_path}")
            return None

        try:
            logger.info(f"Начинаем подготовку изображения для DeepSeek: {image_path}")
            file_size = Path(image_path).stat().st_size
            logger.info(f"Размер файла: {file_size / 1024:.2f} KB")
            
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.warning("Файл слишком большой для отправки в DeepSeek API (>10MB)")
                return None

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=60)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                with Image.open(image_path) as img:
                    max_size = 1024
                    if max(img.size) > max_size:
                        ratio = max_size / max(img.size)
                        new_size = tuple(int(dim * ratio) for dim in img.size)
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85, optimize=True)
                    image_data = buffer.getvalue()
                    base64_image = base64.b64encode(image_data).decode("utf-8")
                
                logger.info(f"Изображение подготовлено (размер base64: {len(base64_image)} символов)")

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Проанализируй этот медицинский отчет. Извлеки ключевую информацию: диагноз, назначенные лекарства, процедуры, рекомендации. Предоставь структурированный ответ в формате JSON: {\"diagnosis\": [...], \"medications\": [...], \"procedures\": [...], \"recommendations\": [...]}",
                            "images": [f"data:image/jpeg;base64,{base64_image}"]
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.7
                }

                logger.info("Отправляем запрос в DeepSeek API...")
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    logger.info(f"Получен ответ от DeepSeek. Статус: {response.status}, Причина: {response.reason}")
                    
                    if response.status != 200:
                        logger.error(f"Тело ответа с ошибкой: {response_text}")
                        try:
                            error_data = json.loads(response_text)
                            if 'error' in error_data:
                                logger.error(f"Детали ошибки API: {error_data['error']}")
                        except:
                            pass
                        return None

                    return response_text

        except FileNotFoundError:
            logger.error(f"Файл не найден при чтении для base64 кодирования: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Критическая ошибка при анализе DeepSeek: {str(e)}")
            return None

    async def analyze_multiple_medical_reports(self, image_paths: list) -> Union[dict, None]:
        """Отправляет несколько медицинских отчетов (изображений) на анализ в DeepSeek одним запросом"""
        if not self.is_available:
            logger.error("DeepSeek API недоступен. Проверьте API ключ и доступность сервиса.")
            return None

        if not image_paths:
            logger.error("Список путей к изображениям пуст")
            return None

        try:
            logger.info(f"Начинаем подготовку {len(image_paths)} изображений для DeepSeek")
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=120)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                base64_images = []
                total_size = 0
                
                for i, image_path in enumerate(image_paths):
                    if not Path(image_path).exists():
                        logger.warning(f"Файл изображения не найден: {image_path}")
                        continue
                    file_size = Path(image_path).stat().st_size
                    if file_size > 5 * 1024 * 1024:
                        logger.warning(f"Файл слишком большой: {image_path} ({file_size / 1024:.2f} KB)")
                        continue

                    with Image.open(image_path) as img:
                        max_size = 800
                        if max(img.size) > max_size:
                            ratio = max_size / max(img.size)
                            new_size = tuple(int(dim * ratio) for dim in img.size)
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=80, optimize=True)
                        image_data = buffer.getvalue()
                        base64_image = base64.b64encode(image_data).decode("utf-8")
                        
                        base64_images.append(f"data:image/jpeg;base64,{base64_image}")
                        total_size += len(base64_image)
                        
                        logger.info(f"Изображение {i+1} подготовлено (размер base64: {len(base64_image)} символов)")

                if not base64_images:
                    logger.error("Не удалось подготовить ни одного изображения")
                    return None

                if total_size > 20 * 1024 * 1024:
                    logger.error(f"Общий размер изображений слишком большой: {total_size / 1024 / 1024:.2f} MB")
                    return None

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                content_parts = [
                    {
                        "type": "text",
                        "text": '''
Тебе надо вывести только два списка чисел через запятую в формате python. Не указывай названия списков. Твой ответ должен выглядеть так: [x, x, x, x, x, x, x, x, x, x, x], [x, x, x, x, x, x, x, x, x, x, x]

Первый список: кол-ва персональных данных каждого типа на картинках, порядок типов: ФИО, адрес, номер паспорта, номер СНИЛС, номер полиса, номер медицинской карты, номер телефона, номер соцстрах, номер ИНН, номер водительских прав, номер банковской карты

Второй список: кол-во обнаруженных ключевых слов в каждом отчете в том же порядке картинок

Второй список может быть пустым, если не обнаружено ни одного ключевого слова.'''

                    },
                    *[
                        {"type": "image_url", "image_url": {"url": image_url}}
                        for image_url in base64_images
                    ]
                ]

                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "user",
                            "content_parts": content_parts,
                        }
                    ],
                    "max_tokens": 2500,
                    "temperature": 0.7
                }

                logger.info(f"Отправляем запрос на анализ {len(base64_images)} изображений в DeepSeek API...")
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    logger.info(f"Получен ответ от DeepSeek. Статус: {response.status}, Причина: {response.reason}")

                    if response.status != 200:
                        logger.error(f"Тело ответа с ошибкой: {response_text}")
                        try:
                            error_data = json.loads(response_text)
                            if 'error' in error_data:
                                logger.error(f"Детали ошибки API: {error_data['error']}")
                        except:
                            pass
                        return None

                    return response_text

        except Exception as e:
            logger.error(f"Критическая ошибка при анализе нескольких отчетов DeepSeek: {str(e)}")
            return None
