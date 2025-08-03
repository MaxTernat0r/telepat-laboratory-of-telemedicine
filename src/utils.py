import os
import logging

logger = logging.getLogger(__name__)

def get_poppler_path() -> str:
    """
    Получает путь к poppler
    
    Returns:
        str: путь к директории с poppler
    """
    logger.info("Начинаем поиск poppler...")
    
    # Пробуем найти poppler в разных местах
    possible_paths = [
        # Относительный путь в проекте
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'side-modules', 'poppler-24.08.0', 'Library', 'bin'),
        # Путь в системных директориях Windows
        r'C:\Program Files\poppler-24.08.0\Library\bin',
        r'C:\Program Files (x86)\poppler-24.08.0\Library\bin',
        # Путь в пользовательской директории
        os.path.expanduser('~\\poppler-24.08.0\\Library\\bin')
    ]
    
    logger.info(f"Проверяем следующие пути:")
    for path in possible_paths:
        logger.info(f"- {path}")
        if os.path.exists(path):
            logger.info(f"  Директория существует")
            if os.path.isdir(path):
                logger.info(f"  Это директория")
                # Проверяем наличие основных файлов poppler
                required_files = ['pdftoppm.exe', 'pdftotext.exe']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
                if not missing_files:
                    logger.info(f"  Найдены все необходимые файлы poppler")
                    return path
                else:
                    logger.warning(f"  Отсутствуют файлы: {', '.join(missing_files)}")
            else:
                logger.warning(f"  Это не директория")
        else:
            logger.info(f"  Директория не существует")
            
    error_msg = "Не найден poppler. Пожалуйста, установите poppler и укажите путь к нему"
    logger.error(error_msg)
    raise ValueError(error_msg)