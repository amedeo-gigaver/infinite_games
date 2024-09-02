#!/bin/bash

PM2_PROCESS_NAME=$1
VENV_DIR="venv"  # Укажите путь к вашему виртуальному окружению

# Активируем виртуальное окружение
source $VENV_DIR/bin/activate

while true; do
  sleep 600

  VERSION=$(git rev-parse HEAD)

  git pull --rebase --autostash

  NEW_VERSION=$(git rev-parse HEAD)

  if [ $VERSION != $NEW_VERSION ]; then
    pip install -r requirements.txt
    pm2 restart $PM2_PROCESS_NAME
  fi
done
