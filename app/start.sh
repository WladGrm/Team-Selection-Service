#!/bin/bash

# Запуск вашего приложения
export DISPLAY=${DISPLAY}
python3 __main__.py 2>&1 > ./output.txt
