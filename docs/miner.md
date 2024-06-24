
# Miner

As a miner you should implement your prediction strategy in the `forward` function of `neurons/miner.py`. You will be receiving data in the form defined in the `infinite_games/protocol.py` and will only need to complete the `probability` entry. 

A key point for the first iteration of the subnet is that you will be receiving data from validators every minute. **You do not have to respond every time**. You only have to send at least one response before the cutoff date of the event. This is particularly important if you use a base model like GPT4 for your mining strategy.

## Base Miner

We are currently providing two base mining models that we will be upgrading in the following weeks.

1. Copying Miner
This miner pulls the latest price from Polymarket, which corresponds to the current aggregate prediction on Polymarket, and sends it to the validators. Similarly, it pull the latest odds for a given Azuro event and sends the inverse (which is the corresponding probability) to validators.

code: WIP

2. Base LLM integration
We integrate the LLM prompting [pipeline](https://github.com/dannyallover/llm_forecasting.git) with news retrieval developed by the authors of the forecasting LLM paper quoted in the readme. As is it only uses [Google News](https://news.google.com/home?hl=en-US&gl=US&ceid=US:en) for news retrieval (the original model from the article used 4 other different sources). It is also based on a single GPT4 base model.You will hence need to add your OpenAI key to your .env file.

code: WIP

## Cutoff

**Polymarket** : 24 hours before the resolution date.

**Azuro** : the start date of the sporting event.

# Miner strategy 

A reference providing a **baseline miner** strategy is the article ["Approaching Human Level Forecasting with Language Models"](https://arxiv.org/html/2402.18563v1?s=35) ([1]). The authors fine-tune an LLM to generate predictions on binary events (including the ones listed on Polymarket) which nears the performance of human forecasters when submitting a forecast for each prediction, and which beats human forecasters in a setting where the LLM can choose to give a prediction or not based on its confidence.

The authors released a version of the code they used to build the LLM they describe in the paper. You can find it [here](https://github.com/dannyallover/llm_forecasting.git).


# System requirements

The computational requirements for a miner will depend significantly on the frequency at which they need to send predictions to be competitive since the base model we consider consists of a set of LLM sub-modules that each need to perform computations. Miners will also need to eventually continually provide new data to their models. Initially with 20-30 events settling per day we estimate miners to require between 16 and 36GB of RAM. We are in the process of testing the computational requirements further. 

A significant cost in the case of the repo referenced above is the cost of using e.g the OpenAI API as well as the cost of retrieving news data. Every time a prediction is produced, a base model is used to generate various prompts. A miner could circumvent that by using a local model or by using the output of the [subnet 18](https://github.com/corcel-api/cortex.t.git). 
News retrieval is done through news provider like https://www.newscatcherapi.com/. They are most often behind a paywall.


# Getting Started



First clone this repo by running `git clone `. Then to run a miner:


Clone repository

```bash
git clone https://github.com/amedeo-gigaver/infinite_games.git
```

Change directory

```bash
cd infinite_games
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate a Virtual Environment

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```
Enable `torch`:

```bash
export USE_TORCH=1
```

The venv should be active whenever the neurons are run.

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your miner.

The miner will be registered to the subnet specified. This ensures that the miner can run the respective miner scripts.

Create a coldkey and hotkey for your miner wallet.

```bash
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

## 2a. (Optional) Getting faucet tokens

Faucet is disabled on the testnet. Hence, if you don't have sufficient faucet tokens, ask the Bittensor Discord community for faucet tokens. Go to the help-forum on the Discord and post your request in the "Requests for Testnet TAO" thread.

## 3. Register keys

This step registers your subnet miner keys to the subnet.

```bash
btcli subnet register --wallet.name miner --wallet.hotkey default
```

To register your miner on the testnet add the `--subtensor.network test` flag.

Follow the below prompts:

```bash
>> Enter netuid (0): # Enter the appropriate netuid for your environment
Your balance is: # Your wallet balance will be shown
The cost to register by recycle is τ0.000000001 # Current registration costs
>> Do you want to continue? [y/n] (n): # Enter y to continue
>> Enter password to unlock key: # Enter your wallet password
>> Recycle τ0.000000001 to register on subnet:8? [y/n]: # Enter y to register
📡 Checking Balance...
Balance:
  τ5.000000000 ➡ τ4.999999999
✅ Registered
```

## 4. Check that your keys have been registered

This step returns information about your registered keys.

Check that your miner key has been registered:

```bash
btcli wallet overview --wallet.name miner
```

To check your miner on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: TBD # or 155 on testnet
COLDKEY    HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
miner  default  197    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                56  none  5GKkQKmDLfsKaumnkD479RBoD5CsbN2yRbMpY88J8YeC5DT4
1          1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                                Wallet balance: τ0.000999999
```

## Running a Miner

### Direct Run 

Run the following command inside the `infinite_games` directory:

`python neurons/miner.py --netuid 155 --subtensor.network test --wallet.name miner --wallet.hotkey default`

### PM2 Installation

Install and run pm2 commands to keep your miner online at all times.


`sudo apt update`

`sudo apt install npm` 

`sudo npm install pm2 -g`

`Confirm pm2 is installed and running correctly`

`pm2 ls`

Use the following example command to run the miner:

`pm2 start neurons/miner.py --interpreter /usr/bin/python3  --name miner -- --wallet.name miner --netuid 155 --wallet.hotkey default --subtensor.network test --logging.debug`


#### Variables Explanation

--wallet.name: Provide the name of your wallet.
--wallet.hotkey: Enter your wallet's hotkey.
--netuid: Use 155 for testnet.
--subtensor.network: Specify the network you want to use (finney, test, local, etc).
--logging.debug: Adjust the logging level according to your preference.
--axon.port: Specify the port number you want to use.


Use the following commands to monitor the status and logs:

`pm2 status`
`pm2 logs 0`

#### Useful PM2 Commands

The following commands will be useful for the management of your miner:

`pm2 list` # lists all pm2 processes

`pm2 logs <pid>` # replace pid with your process ID to view logs

`pm2 restart <pid>` # restart this pic

`pm2 stop <pid>` # stops your pid

`pm2 del <pid>` # deletes your pid

`pm2 describe <pid>` # prints out metadata on the process

## 5. Stopping your miner

To stop your miner, press CTRL + C in the terminal where the miner is running.

