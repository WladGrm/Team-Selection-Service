Сервис подбора проектных групп (Bauman Deep Analytics, Приоритет 2030)
----------------------
----------------------

Установка и запуск без Docker (Windows/Linux/MacOS)
-----------------------------------------------------
1. Склонировать репозиторий с помощью команды `git clone https://gitlab.com/vladgrom1905/team-selection-service.git` и перейти в папку app с помощью `cd team-selection-service/app`.
2. Создать виртуальное окружение `python -m venv venv`.
3. Активировать виртуальное окружение. Для Windows - `venv\Scripts\activate.bat`. Для Linux/MacOS - `source venv/bin/activate`.
4. Установить необходимые библиотеки с помощью `pip install -r requirements.txt`.
5. Запустить приложение с помощью `python __main__.py`.
6. Результат работы программы будет выводиться в консоль, а также будет создан файл Output.xlsx в папке app.


Установка и запуск с помощью Docker (Протестировано на Ubuntu)
--------------------------------------------------------------
1. Убедиться, что установлен Docker и настроены права Х-сервера. Один из вариантов настройки: `xhost +local:root`.
2. Склонировать репозиторий и перейти в папку app
2. Собрать образ с помощью `docker build -t gui-app .`
4. Перейти в корень проекта `cd ..` и запустить Docker Compose `docker compose up -d`
5. В случае запуска с помощью Docker результат работы программы будет сохраняться в виде двух файлов: Output.xlsx и output.txt в папке app. (В среднем решение появится в течение минуты после запуска программы).


 
