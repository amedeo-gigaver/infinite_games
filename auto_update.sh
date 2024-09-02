#!/bin/bash

PM2_PROCESS_NAME=$1
VENV_DIR="venv"

source $VENV_DIR/bin/activate

while true; do

  sleep 500

  VERSION=$(git rev-parse HEAD)

  git pull --rebase --autostash

  NEW_VERSION=$(git rev-parse HEAD)

  if [ "$VERSION" != "$NEW_VERSION" ]; then
    pip install -r requirements.txt
    pm2 restart "$PM2_PROCESS_NAME"
  fi

  pm2_status=$(pm2 show "$PM2_PROCESS_NAME" | grep -i "status" | awk '{print $4}')

  if [ "$pm2_status" = "online" ]; then
    /usr/bin/curl -fsS -m 10 --retry 2 "https://hc-ping.com/ca08ce8d-e25f-47f2-9852-b5a99b6dffad"
  else
    /usr/bin/curl -fsS -m 10 --retry 2 "https://hc-ping.com/ca08ce8d-e25f-47f2-9852-b5a99b6dffad/fail"
    pm2 start neurons/validator.py --name validator --interpreter python3 -- --netuid 6 --subtensor.network finney --wallet.name nkey --wallet.hotkey hkey --logging.debug --logging.trace
  fi
done
