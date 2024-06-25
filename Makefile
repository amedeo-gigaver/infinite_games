SOURCE_VENV := source .venv/bin/activate
PYTHON_INTERPRETER := python3
SOURCE_VENV := source .venv/bin/activate
NETWORK = test
NETUID = 155
WALLET_NAME = foreign_wallet_1
WALLET_HKEY = to_hotkey_to_foreign

.PHONY: clean venv run_miner delete_cache_and_run_miner run_miner_active delete_cache_and_run_miner_active

clean:
	rm -rf .venv

venv:
	${PYTHON_INTERPRETER} -m venv .venv
	${SOURCE_VENV} && pip install -r requirements.txt

run_miner: venv
	${SOURCE_VENV} && pm2 start neurons/miner.py --interpreter ${PYTHON_INTERPRETER} --name miner -- --wallet.name ${WALLET_NAME} --netuid ${NETUID} --wallet.hotkey ${WALLET_HKEY} --subtensor.network ${NETWORK} --logging.debug

delete_cache_and_run_miner: venv
	rm .miner-cache.json
	${SOURCE_VENV} && pm2 start neurons/miner.py --interpreter ${PYTHON_INTERPRETER} --name miner -- --wallet.name ${WALLET_NAME} --netuid ${NETUID} --wallet.hotkey ${WALLET_HKEY} --subtensor.network ${NETWORK} --logging.debug

run_miner_active: venv
	${SOURCE_VENV} && ${PYTHON_INTERPRETER} neurons/miner.py --netuid ${NETUID} --subtensor.network ${NETWORK} --wallet.name ${WALLET_NAME} --wallet.hotkey ${WALLET_HKEY}

delete_cache_and_run_miner_active: venv
	rm .miner-cache.json
	${SOURCE_VENV} && ${PYTHON_INTERPRETER} neurons/miner.py --netuid ${NETUID} --subtensor.network ${NETWORK} --wallet.name ${WALLET_NAME} --wallet.hotkey ${WALLET_HKEY}

