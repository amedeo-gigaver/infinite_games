#!/bin/bash
set -e

cd /root/infinite_games || exit 1

if [ -z "$1" ]; then
  echo "Usage: $0 <PM2_PROCESS_NAME>"
  exit 1
fi
PM2_PROCESS_NAME=$1
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment directory $VENV_DIR does not exist."
  exit 1
fi

source $VENV_DIR/bin/activate

VERSION=$(git rev-parse HEAD)

git pull --rebase --autostash

NEW_VERSION=$(git rev-parse HEAD)

if [ "$VERSION" != "$NEW_VERSION" ]; then
  pip install -r requirements.txt
  pm2 restart "$PM2_PROCESS_NAME" --update-env
fi
### use this command to add script in crontab
### (crontab -l 2>/dev/null; echo "*/10 * * * * /root/infinite_games/checker.sh validator >> /root/infinite_games/checker.log 2>&1") | crontab -
