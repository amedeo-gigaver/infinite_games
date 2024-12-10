# Используем официальный образ Python 3.10 slim
FROM python:3.10-slim

# Устанавливаем переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=300 \
    PATH="/root/.local/bin:${PATH}"

# Устанавливаем рабочую директорию
WORKDIR /root/infinite_games

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем только файл зависимостей для установки пакетов
COPY requirements.txt /root/infinite_games

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копируем остальную часть приложения
COPY . /root/infinite_games


CMD ["python", "neurons/validator.py", "--netuid", "155", "--subtensor.network", "test", "--wallet.name", "${WALLET_NAME}", "--wallet.hotkey", "${WALLET_HOTKEY}"]
