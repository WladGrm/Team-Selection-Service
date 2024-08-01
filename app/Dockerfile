# Стадия сборки
FROM python:3.12.4-slim-bullseye AS builder

# Установим нужные пакеты для сборки
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    make \
    && apt-get clean

# Установим pip и зависимости Python
RUN pip3 install --upgrade pip

# Копируем файл зависимостей и устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --prefix=/install -r requirements.txt

# Копируем остальной код приложения
COPY . /app

# Финальная стадия
FROM python:3.12.4-slim-bullseye

# Установим нужные пакеты для выполнения
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    && apt-get clean

# Установим зависимость для pandas
RUN apt-get install -y libatlas-base-dev

# Скопируем зависимости из builder
COPY --from=builder /install /usr/local

# Копируем файлы приложения из builder
COPY --from=builder /app /app
WORKDIR /app

# Скрипт для запуска приложения
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
