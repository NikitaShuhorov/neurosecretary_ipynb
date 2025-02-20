import os
import warnings
import os
import logging
import warnings
import asyncio
from dataclasses import dataclass
from getpass import getpass

import openai
import whisper
import yt_dlp
import noisereduce as nr
import soundfile as sf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from pydub import AudioSegment


# Подавляем специфические предупреждения
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""# ## Конфигурация системы"""

# Конфигурация
@dataclass
class Config:
    TELEGRAM_TOKEN: str
    OPENAI_API_KEY: str
    WHISPER_MODEL: str = "base"
    GPT_MODEL: str = "gpt-4-turbo"
    AUDIO_CACHE: str = "audio_cache"
    MAX_FILE_SIZE_MB: int = 50
    TEMP_ANALYSIS: float = 0.2
    TEMP_PROTOCOL: float = 0.5
    SUPPORTED_FORMATS: tuple = ('wav', 'mp3', 'ogg', 'flac')

    def __post_init__(self):
        os.makedirs(self.AUDIO_CACHE, exist_ok=True)

"""# ## Обработка аудио с прогресс-баром

"""

# Класс для обработки аудио
class AudioProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("AudioProcessor")

    async def process_input(self, update: Update, input_source: str) -> tuple:
        """
        Определяет источник ввода: аудиофайл из Telegram или ссылка на YouTube,
        скачивает аудио и конвертирует его в WAV (если требуется).
        Возвращает кортеж (путь к аудиофайлу, тип источника).
        """
        if update.message.audio:
            try:
                # Получаем объект файла и скачиваем его
                tg_file = await update.message.audio.get_file()
                file_path = os.path.join(self.config.AUDIO_CACHE, f"{tg_file.file_id}.mp3")
                await tg_file.download_to_drive(custom_path=file_path)
                source_type = "telegram"
                self.logger.info("Аудиофайл из Telegram успешно загружен.")
            except Exception as e:
                self.logger.error(f"Ошибка загрузки файла: {e}")
                raise Exception("Ошибка при загрузке аудиофайла из Telegram.")
        elif update.message.text:
            # Если текст содержит ссылку на YouTube
            if "youtube.com" in input_source or "youtu.be" in input_source:
                source_type = "youtube"
                file_path = await self.download_youtube_audio(input_source)
            else:
                raise Exception("Неподдерживаемый формат. Отправьте аудиофайл или ссылку на YouTube.")
        else:
            raise Exception("Не удалось определить источник ввода.")

        # Если файл не в WAV, выполняется конвертация
        if not file_path.lower().endswith('.wav'):
            wav_path = await self._convert_to_wav(file_path)
            try:
                os.remove(file_path)
            except Exception:
                pass  # Если не удалось удалить, просто продолжаем
            file_path = wav_path

        return file_path, source_type

    async def _convert_to_wav(self, input_path: str) -> str:
        """Конвертация аудио в формат WAV с помощью pydub"""
        try:
            output_path = os.path.splitext(input_path)[0] + ".wav"
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            self.logger.info("Конвертация в WAV завершена.")
            return output_path
        except Exception as e:
            self.logger.error(f"Ошибка конвертации: {e}")
            raise Exception("Ошибка при конвертации аудио в WAV формат.")

    async def download_youtube_audio(self, url: str) -> str:
        """Скачивает аудио с YouTube с помощью yt_dlp"""
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.config.AUDIO_CACHE, '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)
            self.logger.info("Аудио с YouTube успешно загружено.")
            return file_path
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке аудио с YouTube: {e}")
            raise Exception("Ошибка при загрузке аудио с YouTube.")

"""# ## Обработка текста"""

class MeetingProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.whisper_model = whisper.load_model(config.WHISPER_MODEL)
        openai.api_key = config.OPENAI_API_KEY
        self.logger = logging.getLogger("MeetingProcessor")

    async def process_meeting(self, audio_path: str) -> str:
        """
        Транскрибирует аудио с помощью модели Whisper и генерирует протокол встречи с помощью GPT.
        """
        try:
            self.logger.info("Начало транскрипции аудио...")
            result = self.whisper_model.transcribe(audio_path)
            transcription = result.get("text", "")
            self.logger.info("Транскрипция завершена.")

            protocol = self.generate_protocol(transcription)
            return protocol
        except Exception as e:
            self.logger.error(f"Ошибка обработки встречи: {e}")
            raise Exception("Ошибка при обработке аудио.")

    def generate_protocol(self, transcription: str) -> str:
        """
        Отправляет транскрипцию в OpenAI GPT для генерации протокола совещания.
        """
        try:
            prompt = f"Создай протокол совещания на основе следующей транскрипции:\n\n{transcription}"
            response = openai.ChatCompletion.create(
                model=self.config.GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Ты помощник, который создает протоколы совещаний."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
            )
            protocol = response.choices[0].message.content.strip()
            return protocol
        except Exception as e:
            self.logger.error(f"Ошибка генерации протокола: {e}")
            raise Exception("Ошибка при генерации протокола.")

"""# ## Telegram Bot"""

# Основной класс Telegram-бота
class NeuroSecretaryBot:
    def __init__(self, config: Config):
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.meeting_processor = MeetingProcessor(config)
        self.app = Application.builder().token(config.TELEGRAM_TOKEN).build()

        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.AUDIO | filters.TEXT, self.handle_input))

        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )
        self.logger = logging.getLogger("Bot")

    async def start(self, update: Update, context):
        await update.message.reply_text(
            "🤖 Нейро-секретарь готов к работе!\n\n"
            "Отправьте аудиофайл (MP3/WAV/OOG) или ссылку на YouTube видео"
        )

    async def handle_input(self, update: Update, context):
        message = await update.message.reply_text("🔄 Начало обработки...")
        try:
            # Определение типа ввода
            if update.message.audio:
                input_source = update.message.audio.file_id
            elif update.message.text:
                input_source = update.message.text
            else:
                await message.edit_text("❌ Неподдерживаемый формат")
                return

            # Обработка аудио/ссылки
            await message.edit_text("📥 Загрузка данных...")
            audio_path, source_type = await self.audio_processor.process_input(update, input_source)

            # Генерация протокола встречи
            await message.edit_text("🔍 Анализ содержимого...")
            protocol = await self.meeting_processor.process_meeting(audio_path)

            # Отправка результата
            prefix = "🎥 YouTube: " if source_type == "youtube" else "✅ Готово:"
            await message.edit_text(f"{prefix}\n\n{protocol}")

            # Очистка временного файла
            if os.path.exists(audio_path):
                os.remove(audio_path)

        except Exception as e:
            await message.edit_text(f"❌ Ошибка: {str(e)}")
            self.logger.error(f"Main Error: {str(e)}")

    def run(self):
        self.logger.info("Бот запущен")
        self.app.run_polling()

"""# ## Запуск бота"""

# Точка входа в программу
if __name__ == "__main__":
    try:
        # Для корректной работы в Jupyter применяем nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()

        print("🔑 Введите данные для настройки:")
        config = Config(
            TELEGRAM_TOKEN=getpass("Telegram Token (@BotFather): "),
            OPENAI_API_KEY=getpass("OpenAI API Key (https://platform.openai.com): ")
        )

        bot = NeuroSecretaryBot(config)
        bot.run()
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")