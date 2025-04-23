
## Wallet setup

This step creates local coldkey and hotkey pairs for your miner or validator. Both of these items can be shared (for example you have to share you coldkey to get faucet tokens and these will appear on [Taostats](https://taostats.io/)). Do **not** share your mnemonic.

Create a coldkey and hotkey for your miner/validator wallet.

```bash
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

You can also regenerate an existing wallet using the following:

```bash
btcli w regen_coldkey --wallet-path ~/.bittensor/wallets/ \
    --wallet-name miner --mnemonic "${MNEMONIC_COLDKEY}"

btcli w regen_hotkey --wallet-name miner --wallet-hotkey default \
    --mnemonic "${MNEMONIC_HOTKEY}"
```

Two important commands are `btcli wallet list` and `btcli wallet overview`. 

The first enables one to vizualize their wallets in a tree-like manner as well as which hotkeys are associated with a given coldkey. Importantly, one can see the wallet name that is associated with each pair of coldkey and hotkey. This is important since the names of the coldkey and the hotkey are used rather than the keys themselves when running bittensor commands related to wallets.  

The second enables one to see the subnets in which their wallet is registered as well as the wallet's balance. 
