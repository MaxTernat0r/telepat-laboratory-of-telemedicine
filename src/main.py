import os
import asyncio
import shutil
from pathlib import Path
from loguru import logger
from processor import DocumentProcessor
from deepseek_client import DeepSeekClient
from gigachat_client import GigaChatClient
from chatgpt_client import ChatGPTClient
import json

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Настройка логирования для всего приложения"""
    logger.remove()  # Удаляем стандартный обработчик
    logger.add(
        "logs/processor_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(lambda msg: print(msg), level="DEBUG")

    # Создаем директорию для логов, если она не существует
    Path("logs").mkdir(exist_ok=True)


# Настраиваем логирование при импорте модуля
setup_logging()


async def process_documents() -> None:
    """
    Обрабатывает все документы в директории input
    """
    processor = await DocumentProcessor.create()

    # Определяем статические пути
    input_dir = Path("input")
    output_dir = Path("output")

    # Создаем выходную директорию, если она не существует
    output_dir.mkdir(parents=True, exist_ok=True)

    # Получаем список всех файлов в директории
    input_files = [f for f in input_dir.iterdir() if
                   f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf')]

    if not input_files:
        logger.warning("В директории input не найдено изображений")
        return

    logger.info(f"Найдено {len(input_files)} файлов для обработки")

    # Создаем клиенты для API
    deepseek = DeepSeekClient()
    gigachat = GigaChatClient()
    chatgpt = ChatGPTClient()

    try:
        # Обрабатываем каждый файл последовательно
        for file_path in input_files:
            logger.info(f"Обработка файла: {file_path.name}")
            logger.debug(f"Полный путь к входному файлу: {file_path.absolute()}")

            try:
                # Сначала обрабатываем документ через DocumentProcessor
                output_path, extracted_data = await processor.process_document(
                    file_path=str(file_path),
                    output_dir=str(output_dir)
                )

                if output_path and Path(output_path).exists():
                    logger.info(f"Файл успешно обработан и сохранен: {output_path}")
                    logger.debug(f"Полный путь к деперсонализированному файлу: {Path(output_path).absolute()}")

                    # Проверяем размер файла
                    file_size = Path(output_path).stat().st_size
                    logger.debug(f"Размер деперсонализированного файла: {file_size} байт")

                    # Затем отправляем деперсонализированный файл на анализ через все доступные API
                    logger.info(f"Отправка деперсонализированного файла на анализ: {output_path}")

                    # Создаем директории для результатов
                    for dir_name in ['ai-result/deepseek', 'ai-result/gigachat', 'ai-result/chatgpt']:
                        Path(dir_name).mkdir(parents=True, exist_ok=True)

                    # Получаем базовое имя файла без расширения
                    base_name = Path(output_path).stem

                    # Пробуем анализ через DeepSeek
                    try:
                        logger.info(f"Начинаем анализ через DeepSeek: {output_path}")
                        deepseek_result = await deepseek.analyze_medical_report(output_path)
                        if deepseek_result:
                            output_file = Path(f"ai-result/deepseek/{base_name}_deepseek.json")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(deepseek_result, f, ensure_ascii=False, indent=2)
                            logger.info(f"Результат DeepSeek сохранен в {output_file}")
                    except Exception as e:
                        logger.error(f"Ошибка при анализе через DeepSeek: {str(e)}")

                    # Пробуем анализ через GigaChat
                    try:
                        logger.info(f"Начинаем анализ через GigaChat: {output_path}")
                        gigachat_result = await gigachat.analyze_medical_report(output_path)
                        if gigachat_result:
                            output_file = Path(f"ai-result/gigachat/{base_name}_gigachat.json")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(gigachat_result, f, ensure_ascii=False, indent=2)
                            logger.info(f"Результат GigaChat сохранен в {output_file}")
                    except Exception as e:
                        logger.error(f"Ошибка при анализе через GigaChat: {str(e)}")

                    # Пробуем анализ через ChatGPT
                    try:
                        logger.info(f"Начинаем анализ через ChatGPT: {output_path}")
                        chatgpt_result = await chatgpt.analyze_medical_report(
                            output_path,
                            prompt='''
тебе передается файл с медицинскими анализами. Если на передаваемом тебе файле удается обнаружить предмет исследования, значение и референсные значения, то ответь коротко 1, иначе 0.
'''
                        )
                        if chatgpt_result:
                            output_file = Path(f"ai-result/chatgpt/{base_name}_chatgpt.txt")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(chatgpt_result)
                            logger.info(f"Результат ChatGPT сохранен в {output_file}")
                            logger.debug(f"Содержимое ответа ChatGPT: {chatgpt_result[:200]}...")
                    except Exception as e:
                        logger.error(f"Ошибка при анализе через ChatGPT: {str(e)}")
                        logger.exception("Полный стек ошибки:")

                    # Перемещаем обработанный файл в архив
                    archive_dir = input_dir / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    shutil.move(file_path, archive_dir / file_path.name)
                    logger.info(f"Файл {file_path.name} перемещен в архив")
                else:
                    logger.error(f"Не удалось получить или найти деперсонализированный файл для {file_path.name}")

            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path.name}: {str(e)}")
                logger.exception("Полный стек ошибки:")
                continue
    finally:
        # Закрываем клиенты API, которые поддерживают закрытие
        try:
            if hasattr(gigachat, 'close'):
                await gigachat.close()
            if hasattr(chatgpt, 'close'):
                await chatgpt.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии клиентов API: {str(e)}")


async def main():
    # Проверяем существование входной директории
    if not Path("input").exists():
        logger.error("Директория input не существует")
        return

    # Инициализируем менеджер данных

    # Загружаем существующие данные, если они есть

    try:
        # Запускаем обработку документов
        await process_documents()

        # Сохраняем все данные перед завершением

    except Exception as e:
        logger.error(f"Ошибка при выполнении: {str(e)}")
        logger.exception("Полный стек ошибки:")
    finally:
        logger.info("Обработка завершена")


if __name__ == "__main__":
    asyncio.run(main())
