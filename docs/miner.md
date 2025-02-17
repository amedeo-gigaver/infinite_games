# Miner

You will be receiving data in the form defined in `neurons/protocol.py` and will only need to complete the `probability` entry.

You will be receiving requests from validators every 3-5 minute. You do not have to recalculate your prediction every time. Our baseline miner is implementing a caching system that allows for a recalculation schedule. This is particularly important if you use a base model like GPT4 that might be expensive.

## General

Before registering your miner on the main network you can register it on testnet.
Our testnet netuid is 155. You will also need faucet TAO. You can ask for them [here](https://discord.com/channels/799672011265015819/1190048018184011867).

To register on mainnet you will need real TAO. You can get TAO by swapping them against USDC on an exchange like [MEXC](https://www.mexc.com/). Other exchanges supporting TAO are listed [here](https://taostats.io/). We give steps below on creating your own wallet from [btcli](https://docs.bittensor.com/btcli). The registration cost fluctuates depending on the current demand to register on the network but it's minimum price is 0.7 TAO. After you register there is an immunity period during which you cannot be excluded, if you are the lowest miner at the end of it you are deregistered. 

Here are entries in the Bittensor documentation for [miners](https://docs.bittensor.com/miners/) and [coldkeys](https://docs.bittensor.com/getting-started/wallets). 

## Key insights into the scoring system

- You have to predict all events, you get the worst score if you don't sent a prediction for a given time interval (current each interval lasts 4h) on a given event
- You are mostly rewarded for your early predictions
- In the new peer score system if you are a new miner we give you a score of $0$ on all the intervals for which you could not submit a prediction but where the associated question is taken into account in your score (in the legacy system we assume you sent 0.5)

## Baseline Miner

We integrate an LLM pipeline based on the [forecasting-tools](https://github.com/CodexVeritas/forecasting-tools/) repository. Please refer to the *Getting Started* section below for setup instructions.

# Miner strategy

A reference in the literature providing insights on building a forecasting LLM is the article ["Approaching Human Level Forecasting with Language Models"](https://arxiv.org/html/2402.18563v1?s=35) ([1]). The authors fine-tune an LLM to generate predictions on binary events (including the ones listed on Polymarket) which nears the performance of human forecasters when submitting a forecast for each prediction, and which beats human forecasters in a setting where the LLM can choose to give a prediction or not based on its confidence.

The authors released a version of the code they used to build the LLM they describe in the paper. You can find it [here](https://github.com/dannyallover/llm_forecasting.git). 


# System requirements

The computational requirements for a miner will depend significantly on the frequency at which they need to send predictions to be competitive since the base model we consider consists of a set of LLM sub-modules that each need to perform computations. Miners will also need to eventually continually provide new data to their models.

A significant cost in the case of the repo referenced above is the cost of using e.g the OpenAI API as well as the cost of retrieving news data. Every time a prediction is produced, a base model is used to generate various prompts. A miner could circumvent that by using a local model or by using the output of the [subnet 18](https://github.com/corcel-api/cortex.t.git).
News retrieval is often done through news provider like https://www.newscatcherapi.com/. They are most often behind a paywall.


# Background information on event types

## ACLED

We join [here](docs/ACLED-data/protest-data-3days.json) and [here](docs/ACLED-data/Protests-events-7-days.json) two dataset of events that stretches over 6 months (from January 2024 to June 2024) which represents the type of events that will be sent to miners from [ACLED](https://acleddata.com/) data.

This first set of events will have the following structure:
"Will the amount of protests in [country] during [3 days window or 1 week window] be above [30 day moving average] ?"

**Key details**:
- the countries will initially be EU countries (with non degenerate protest rates) and the US. This is the exact list: ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Poland', 'Romania', 'Netherlands', 'Belgium', 'Ireland', 'Sweden', 'Czech Republic', 'Greece', 'Portugal', 'Hungary', 'Austria', 'Serbia', 'Bulgaria', 'Denmark', 'United States']
- the cutoff will be set at the start of the time window
- events will be generated daily but they will only resolve on Tuesday-Wednesday due to the schedule ACLED follows to update its data.

Miners will have to retrieve protest data relevant to each of the countries listed above in order to improve their predictions.



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

The venv should be active whenever the neurons are run.

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your miner. Both of these items can be shared (for example you have to share you coldkey to get faucet tokens and these will appear on [Taostats](https://taostats.io/)). Do **not** share your mnemonic.

Create a coldkey and hotkey for your miner wallet.

```bash
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

## 2a. (Optional) Getting faucet tokens

Faucet is disabled on the testnet. Hence, if you don't have sufficient faucet tokens, ask the Bittensor Discord community for faucet tokens. Go to the help-forum on the Discord and post your request in the "Requests for Testnet TAO" thread.

## 3. Register keys

This step registers your subnet miner keys to the subnet. This ensures that the miner can run the respective miner scripts.

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

`pm2 start neurons/miner.py --interpreter python3  --name miner -- --wallet.name miner --netuid 155 --wallet.hotkey default --subtensor.network test --logging.debug`


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

### Useful PM2 Commands

The following commands will be useful for the management of your miner:

`pm2 list` # lists all pm2 processes

`pm2 logs <pid>` # replace pid with your process ID to view logs

`pm2 restart <pid>` # restart this pic

`pm2 reload <pid>` # reload this pic

`pm2 stop <pid>` # stops your pid

`pm2 del <pid>` # deletes your pid

`pm2 describe <pid>` # prints out metadata on the process

## 5. Stopping your miner

To stop your miner, press CTRL + C in the terminal where the miner is running.

## 6. Using the Built-in LLM forecaster

The repository includes a pre-built LLM (Language Model) forecaster at [`neurons/miner/forecasters/llm_forecaster.py`](../neurons/miner/forecasters/llm_forecaster.py) that you can use for making predictions. To use it:

1. Get the required API keys:
   - Perplexity API key
   - OpenAI API key

2. Set your API keys as environment variables:
```bash
export PERPLEXITY_API_KEY=<your_perplexity_api_key>
export OPENAI_API_KEY=<your_openai_api_key>
```

3. Open [`neurons/miner.py#L31`](../neurons/miner.py#L31) and locate the `assign_forecaster` function
4. Replace the placeholder forecaster with `LLMForecaster`
5. Start your miner - it will now use LLM models to forecast events

## 7. Creating Your Own forecaster

You can create your own custom forecaster to implement different prediction strategies:

1. Create a new forecaster class that inherits from the base forecaster ([`neurons/miner/forecasters/base.py`](../neurons/miner/forecasters/base.py))
2. Implement the `_run` method in your forecaster
   - This method must return a prediction value between 0 and 1
   - You can access event information through the `MinerEvent` class ([`neurons/miner/models/event.py`](../neurons/miner/models/event.py))

3. Update the `assign_forecaster` function in [`neurons/miner.py`](../neurons/miner.py) to use your forecaster

The `assign_forecaster` function lets you use different forecasters for different types of events. You can examine the event information and choose the most appropriate forecaster for each case.

