import base64
import httpx
import os
from loguru import logger

class YandexGPTClient:
    def __init__(self):
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.api_key = os.getenv("YANDEXGPT_API_KEY")# замени на свой ключ при необходимости
        self.folder_id = "ajei3u6jep2qanc8a87n"
        self.model_uri = f"gpt://{self.folder_id}/yandexgpt/latest"

    async def analyze_medical_report(self, file_path: str, prompt: str) -> str:
        try:
            # Читаем и кодируем картинку в base64
            with open(file_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")

            # Включаем название файла (или описание) в prompt,
            # т.к. YandexGPT не поддерживает картинки напрямую
            full_prompt = f"{prompt}\n\n[Изображение: {file_path.split('/')[-1]}]"

            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "modelUri": self.model_uri,
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.4,
                    "maxTokens": 1000
                },
                "messages": [
                    {"role": "system", "text": "Ты — врач-лаборант. Отвечай кратко, точно и без воды."},
                    {"role": "user", "text": full_prompt}
                ]
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)

            response.raise_for_status()
            result = response.json()
            return result["result"]["alternatives"][0]["message"]["text"]

        except Exception as e:
            logger.error(f"Ошибка при обращении к YandexGPT: {e}")
            return None
