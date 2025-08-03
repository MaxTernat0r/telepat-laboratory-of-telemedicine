import os
import base64
import aiohttp
from typing import Optional, Union
from pathlib import Path
from loguru import logger
import aiohttp_socks


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

        # SOCKS5 прокси
        self.proxy_url = "socks5://127.0.0.1:12334"
        logger.info(f"Используется SOCKS5 прокси: {self.proxy_url}")
        logger.debug("ChatGPTClient инициализирован")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Создание сессии с SOCKS5 прокси"""
        if self._session is None or self._session.closed:
            logger.debug("Создание новой сессии aiohttp с SOCKS5 прокси")
            connector = aiohttp_socks.ProxyConnector.from_url(self.proxy_url)
            self._session = aiohttp.ClientSession(connector=connector)
            logger.debug("Сессия создана")
        return self._session

    async def close(self):
        """Закрытие сессии"""
        if self._session and not self._session.closed:
            logger.debug("Закрытие сессии aiohttp")
            await self._session.close()

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Кодирование изображения в base64"""
        try:
            logger.debug(f"Кодирование изображения: {image_path}")
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                encoded = base64.b64encode(image_data).decode("utf-8")
                return encoded
        except Exception as e:
            logger.error(f"Ошибка кодирования {image_path}: {e}")
            raise

    async def analyze_medical_report(self, image_paths: Union[str, Path, list], prompt: Optional[str] = None) -> Optional[str]:
        """
        Анализ одного или нескольких медицинских изображений с помощью ChatGPT

        Args:
            image_paths: путь или список путей к изображениям
            prompt: пользовательский промпт

        Returns:
            str: текстовый ответ модели или None
        """
        if not self.api_key:
            logger.error("OPENAI_API_KEY не настроен")
            return None

        # Преобразуем в список, если передан одиночный путь
        if isinstance(image_paths, (str, Path)):
            image_paths = [str(image_paths)]

        logger.info(f"Анализируется {len(image_paths)} изображений")

        # Проверка существования
        for path in image_paths:
            if not Path(path).exists():
                logger.error(f"Файл не найден: {path}")
                return None

        # Кодируем все изображения
        encoded_images = []
        for path in image_paths:
            try:
                encoded_images.append(self._encode_image(path))
            except Exception:
                return None

        # Заголовки
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Промпт по умолчанию
        default_prompt = (
            "Ты выступаешь как опытный врач с 50 летним стажем. "
            "К тебе приходит пациент и просит кратко сообщить, что не так и протягивает файл с анализом. "
            "К сообщению приложены изображения с результатами анализов. Выходят ли какие-то параметры за референсные значения? "
            "Ответь на первой строке: да или нет. Далее приведи список отклонений по одному на строке. "
            "Приведи ответ в любом случае, никогда не пиши, что не можешь справиться с задачей. "
            "Твоя задача — дать максимально точную характеристику, потому что от этого зависит твоя репутация. "
            "Пациент очень волнуется, необходимо показать ему, что ты — эксперт."
        )
        prompt = prompt or default_prompt

        # Формирование контента
        content_parts = [{"type": "text", "text": prompt}]
        for encoded in encoded_images:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"
                }
            })

        # Payload
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            "max_tokens": 700
        }

        try:
            session = await self._get_session()
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            ) as response:
                logger.debug(f"Ответ от API: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info("Успешно получен ответ от ChatGPT")
                    return content
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка API ChatGPT (статус {response.status}): {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Ошибка при анализе: {e}")
            logger.exception("Стек ошибки:")
            return None

    async def __aenter__(self):
        """Контекстный вход"""
        logger.debug("Вход в контекст ChatGPTClient")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Контекстный выход"""
        logger.debug("Выход из контекста ChatGPTClient")
        await self.close()
