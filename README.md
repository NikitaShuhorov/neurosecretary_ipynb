# neurosecretary_ipynb
 neurosecretary_ipynb

Создайте виртуальное окружение
python -m venv venv

Активируйте виртуальное окружение

source venv/bin/activate

Установите зависимости

pip install -r requirements.txt


"""# ## Установка зависимостей
# Выполните эту ячейку для установки необходимых библиотек
"""

pip install -q python-telegram-bot openai yt-dlp whisper noisereduce soundfile numpy tqdm pydub nest_asyncio

"""**Зта установка помогает избежать несовместимостей и неожиданных изменений в API, которые могут возникнуть при обновлении до более новой версии.**"""

pip install openai==0.28

sudo dnf install -y ffmpeg > /dev/null  # Для обработки аудио

pip install git+https://github.com/openai/whisper.git # Для Транскрибации аудиофайлов
