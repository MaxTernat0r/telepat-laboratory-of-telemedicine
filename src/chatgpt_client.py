import os
import base64
import aiohttp
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger
import aiohttp_socks
import json

class ChatGPTClient:
    """Клиент для работы с ChatGPT API"""
    
    def __init__(self):
        """Инициализация клиента ChatGPT"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY не найден в переменных окружения")
        else:
            logger.info("OPENAI_API_KEY успешно загружен")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Настройки SOCKS5 прокси (порт по умолчанию для Hiddify)
        self.proxy_url = "socks5://127.0.0.1:12334"
        logger.info(f"Используется SOCKS5 прокси: {self.proxy_url}")
        logger.debug("ChatGPTClient инициализирован")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Получение или создание сессии aiohttp"""
        if self._session is None or self._session.closed:
            logger.debug("Создание новой сессии aiohttp с SOCKS5 прокси")
            connector = aiohttp_socks.ProxyConnector.from_url(self.proxy_url)
            self._session = aiohttp.ClientSession(connector=connector)
            logger.debug(f"Создана сессия с прокси: {self.proxy_url}")
        return self._session

    async def close(self):
        """Закрытие сессии"""
        if self._session and not self._session.closed:
            logger.debug("Закрытие сессии aiohttp")
            await self._session.close()

    def _encode_image(self, image_path: str) -> str:
        """Кодирование изображения в base64"""
        try:
            logger.debug(f"Начало кодирования изображения: {image_path}")
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                logger.debug(f"Размер изображения: {len(image_data)} байт")
                encoded = base64.b64encode(image_data).decode('utf-8')
                logger.debug(f"Изображение успешно закодировано в base64 (длина: {len(encoded)})")
                return encoded
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения {image_path}: {str(e)}")
            raise

    async def analyze_medical_report(self, image_path: str, prompt: Optional[str] = None) -> Optional[str]:
        """
        Анализ медицинского отчета с помощью ChatGPT
        
        Args:
            image_path: путь к изображению
            prompt: опциональный промпт для анализа
            
        Returns:
            str: результат анализа или None в случае ошибки
        """
        logger.info(f"Начало анализа медицинского отчета: {image_path}")
        
        if not self.api_key:
            logger.error("OPENAI_API_KEY не настроен")
            return None

        try:
            # Проверяем существование файла
            if not Path(image_path).exists():
                logger.error(f"Файл не найден: {image_path}")
                return None
            logger.debug(f"Файл существует: {image_path}")

            # Кодируем изображение
            logger.debug("Начало кодирования изображения в base64")
            encoded_image = self._encode_image(image_path)
            logger.debug("Изображение успешно закодировано")
            
            # Формируем заголовки
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            logger.debug("Заголовки запроса сформированы")

            # Формируем промпт
            default_prompt = "Ты выступаешь как опытный врач с 50 летним стажем. К тебе приходит пациент и просит кратко сообщить, что не так и протягивает файл с анализом. К сообщению приложены изображения с результатами анализов. Выходят ли какие то параметры за реферсные значения? Ответь на первой строке да или нет, далее приведи список отклонений по одному на строке. Приведи ответ в любом случае, никогда не пиши что не можешь справиться с задачей. Твоя задача дать максимально точную характеристику, потому что от этого зависит твоя репутация. Пациент очень валнуется, необходимо показать ему, что ты - эксперт."
            prompt = prompt or default_prompt
            logger.debug(f"Используемый промпт: {prompt}")

            # Формируем payload
            payload = {
                "model": "gpt-4o",  # Обновленная модель для работы с изображениями
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            logger.debug("Payload сформирован")

            # Отправляем запрос
            logger.info("Отправка запроса к ChatGPT API через SOCKS5 прокси")
            session = await self._get_session()
            
            async with session.post(
                self.base_url, 
                headers=headers, 
                json=payload,
                timeout=30  # Увеличиваем таймаут для работы через прокси
            ) as response:
                logger.debug(f"Получен ответ от API (статус: {response.status})")
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info("Успешно получен ответ от ChatGPT")
                    logger.debug(f"Содержимое ответа: {content[:100]}...")
                    return content
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка API ChatGPT (статус {response.status}): {error_text}")
                    logger.error(f"Заголовки ответа: {dict(response.headers)}")
                    if response.status == 403 and "unsupported_country_region_territory" in error_text:
                        logger.error("Доступ к API заблокирован. Проверьте работу SOCKS5 прокси.")
                    return None

        except Exception as e:
            logger.error(f"Ошибка при анализе через ChatGPT: {str(e)}")
            logger.exception("Полный стек ошибки:")
            return None

    async def __aenter__(self):
        """Поддержка контекстного менеджера"""
        logger.debug("Вход в контекстный менеджер ChatGPTClient")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Закрытие сессии при выходе из контекста"""
        logger.debug("Выход из контекстного менеджера ChatGPTClient")
        await self.close()

    async def analyze_multiple_medical_reports(self, image_paths: list) -> Optional[Dict[str, Any]]:
        """
        Анализ нескольких медицинских отчетов с помощью ChatGPT
        
        Args:
            image_paths: список путей к изображениям
            
        Returns:
            Dict: результат анализа или None в случае ошибки
        """
        logger.info(f"Начало анализа {len(image_paths)} медицинских отчетов")
        
        if not self.api_key:
            logger.error("OPENAI_API_KEY не настроен")
            return None

        if not image_paths:
            logger.error("Список путей к изображениям пуст")
            return None

        try:
            # Проверяем существование всех файлов
            for image_path in image_paths:
                if not Path(image_path).exists():
                    logger.error(f"Файл не найден: {image_path}")
                    return None

            # Кодируем все изображения
            encoded_images = []
            for i, image_path in enumerate(image_paths):
                try:
                    encoded_image = self._encode_image(image_path)
                    encoded_images.append(encoded_image)
                    logger.debug(f"Изображение {i+1} успешно закодировано")
                except Exception as e:
                    logger.error(f"Ошибка при кодировании изображения {i+1}: {str(e)}")
                    return None

            if not encoded_images:
                logger.error("Не удалось закодировать ни одного изображения")
                return None

            # Формируем заголовки
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # Формируем промпт для множественных файлов
            prompt = "Ты выступаешь как опытный врач с 50 летним стажем. К тебе приходит пациент и просит кратко сообщить, что не так и протягивает несколько страниц с результатами анализов. Выходят ли какие то параметры за реферсные значения? Ответь на первой строке да или нет, далее приведи список отклонений по одному на строку. Приведи ответ в любом случае, никогда не пиши что не можешь справиться с задачей. Твоя задача дать максимально точную характеристику, потому что от этого зависит твоя репутация. Пациент очень волнуется, необходимо показать ему, что ты - эксперт."

            # Формируем контент с несколькими изображениями
            content_parts = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            # Добавляем все изображения
            for encoded_image in encoded_images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                })

            # Формируем payload
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                "max_tokens": 500  # Увеличиваем лимит для множественных файлов
            }

            # Отправляем запрос
            logger.info(f"Отправка запроса к ChatGPT API с {len(encoded_images)} изображениями")
            session = await self._get_session()
            
            async with session.post(
                self.base_url, 
                headers=headers, 
                json=payload,
                timeout=60  # Увеличиваем таймаут для множественных файлов
            ) as response:
                logger.debug(f"Получен ответ от API (статус: {response.status})")
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info("Успешно получен ответ от ChatGPT для множественных файлов")
                    
                    # Пытаемся извлечь структурированную информацию
                    try:
                        # Ищем JSON в ответе
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = content[start_idx:end_idx]
                            result_dict = json.loads(json_str)
                            logger.info("Успешно извлечен JSON из ответа ChatGPT")
                            return result_dict
                        else:
                            # Если JSON не найден, возвращаем структурированный ответ
                            lines = content.strip().split('\n')
                            has_deviations = lines[0].lower().strip() in ['да', 'yes', 'true']
                            
                            deviations = []
                            if has_deviations and len(lines) > 1:
                                deviations = [line.strip() for line in lines[1:] if line.strip()]
                            
                            return {
                                "has_deviations": has_deviations,
                                "deviations": deviations,
                                "raw_response": content,
                                "files_analyzed": len(image_paths)
                            }
                    except json.JSONDecodeError:
                        logger.warning("Не удалось распарсить JSON из ответа, возвращаем структурированный ответ")
                        return {
                            "raw_response": content,
                            "files_analyzed": len(image_paths)
                        }
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка API ChatGPT (статус {response.status}): {error_text}")
                    return None

        except Exception as e:
            logger.error(f"Ошибка при анализе множественных отчетов через ChatGPT: {str(e)}")
            logger.exception("Полный стек ошибки:")
            return None 