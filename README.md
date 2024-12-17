<div align="center">

<img src="docs/infinite-games.jpeg" alt="Project Logo" width="200"/>

# **Infinite Games** 


[Discord](https://discord.gg/BWdj7SGQ) • [Dashboard](https://app.hex.tech/1644b22a-abe5-4113-9d5f-3ad05e4a8de7/app/5f1e0e62-6072-4440-9646-6d2b60cd1674/latest) •
[Website](https://www.infinitegam.es/) • [Twitter](https://twitter.com/Playinfgames) •  [Network](https://taostats.io/) 

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>



<!-- update with discord invite link -->

</div>

  <!--
  <a href="">Twitter</a> -->
    
  <!-- <a href="">Bittensor</a> -->


# Forecasting of Future Events

##  Introduction


We incentivize the prediction of future events. The prediction space is based on binary future events such as the one listed on [Polymarket](https://polymarket.com/) and on [Azuro](https://azuro.org/). We are always actively expanding to new markets and providers. We are focused on *judgemental forecasting* rather than *statistical forecasting*. We hence expect the models used by miners to be LLMs. 

### High-level mechanism

Miners submit their predictions to validators. Each prediction has to be done early enough before the event underlying the prediction settles. Once the event settles, the validators that received the prediction score the miner.

## LLMs open new predictive capabilities

Making predictions is a hard task that requires cross-domain knowledge and intuition. It is often limited in explanatory reasoning and domain-specific (the expert in predicting election results will differ from the one predicting the progress in rocket-engine technology) ([1]). At the same time it is fundamental to human society, from geopolitics to economics. 

<!-- The COVID-19 measures for example were based on epidemiological forecasts. Science is another area where prediction is crucial ([2]) to determine new designs or to predict the outcome of experiments (executing one experiment is costly). Such predictions rely on the knowledge of thousands papers and on multidisciplinary and multidimensional analysis (*can a study replicate ? should one use a molecular or behavioral approach?*). -->

LLMs approach or surpass human forecasting abilities. They near on average the crowd prediction on prediction market events ([1]), and surpass humans in predicting neuroscience results ([2]). They are also shown to be calibrated with their predictions i.e confident when right. Through their generalization capabilities and unbounded information processing, LLMs have the potential to automate the prediction process or complement humans. 


### Real-world applications

The value of the subnet first relies in the improvement of the efficiency of prediction markets. This value can be extracted by validators through arbitrage. The validators may obtain a better knowledge of the probability of an event settling and communicate this information to a prediction market by opening a position. 

The first applications built on top of our subnet could be related to prediction markets. A trader could query our market to obtain the most up to date and relevant predictions to their portfolio based on the current news landscape (LLMs would be constantly aggregating the most up to date and relevant news articles). They could then readjust their positions accordingly or trade directly on this information. 

In the long term, a validator could provide paid economic forecasts or more generally the output of any forward-looking task addressed to an LLM ([2]). A customer might then provide a series of paid sub-queries related to the information they aim at retrieving.

<!-- It could also be used by scientists to design their experiment and frame their ideas. For example, the value of a paper often resides in the way the results are presented and cross-analysed. One way resulting in poor conclusions while the other giving good results. An LLM might help detect the adequate framework. -->


## Miners 

Miners compete by sending to the validators a dictionary identifying an event $E$ with their estimation of the probability $p$ that $E$ is realized. For example, $E$ could be *SBF is sentenced to life*. In the case of Polymarket, an event identifier is a [Polymarket condition id](https://docs.polymarket.com/#overview-8). 


### Miner strategy 

A reference providing a **baseline miner** strategy is the article ["Approaching Human Level Forecasting with Langage Models"](https://arxiv.org/html/2402.18563v1?s=35) ([1]). The authors fine-tune an LLM to generate predictions on binary events (including the ones listed on Polymarket) which nears the performance of human forecasters when submitting a forecast for each prediction, and which beats human forecasters in a setting where the LLM can choose to give a prediction or not based on its confidence.

According to the article, the performance of LLMs likely depends significantly on the amount of data they can retrieve for a given prediction. In the study, this performance was likely limited by the finite amount of data one can extract from prediction markets. If our subnet is able to continually produce new synthetic data miners could be able to beat the SoA (average Brier score of 0.179).


## Validators

Validators record the miners' predictions and score them once the events settle. At each event settlement, a score is added to the moving average of the miner's score. This simple model ensures that all validators score the miners at roughly the same time. Importantly, we implement a **cutoff** for the submission time of a prediction. The cutoff is currently set at 24 hours for Polymarket events and at the start of the relevant sporting event on Azuro (think kick-off of a soccer match). The cutoff is needed since as the event nears resolution the probability of the true outcome tends to one.

## Scoring rule
*We are currently using model 2*

Denote by $S(p_i, o_i)$ the quadratic scoring rule (the Brier score) for a prediction $p_i$ of a binary event $E_i$ and where $o_i$ is $1$ if $E_i$ is realized and $0$ otherwise. With a renormalization we have that $S(p_i, 1)= 1- (1-p_i)^2$ if $o_i$ is $1$ and $S(p_i,0)=1-p_i^2$ if $o_i$ is $0$. A quadratic scoring rule is strictly proper i.e it strictly incentivizes miners to report their true prediction. 

### model 1

The validators directly use a **quadratic scoring rule** on the miners' predictions. If the miner predicted that $E_i$ be realized with probability $p_i$, upon settlement of the outcome the validator scores the miner by adding $S(p_i, o_i)$ to their moving average of the miner's score.

We give miners a score of $0$ on the events for which they did not submit a prediction.

### model 2

The validator stores **the time series of the miner's predictions** and computes the Brier score of each element of the time series. It hence obtains a new time series of Brier scores. A number $n$ of intervals is set between the issues date and the resolution date. The validator then computes a **weighted average of the Brier scores**, where the weight is exponentially decreasing with time, in interval $k$ it has value $exp(-\frac{n}{k} +1)$ where $k$ starts at $n$ and decreases to $1$.

The final score is a linear combination of the weighted average and of a linear component that depends on how good is the miner compared to other miners.

This is described in details [here](https://hackmd.io/@nielsma/S1sB8xO_C).


### model 3


We implement a **sequentially shared quadratic scoring rule**. This allows us crucially to aggregate information as well as to score $0$ miners that do not bring new information to the market.
The scoring rule functions by scoring each miner relatively to the previous one. The score of the miner $j$ is then $S_j = S(p_j, o_i) - S(p_{j-1}, o_i)$ where $p_{j-1}$ is the submission of the previous miner. Importantly this payoff can be negative, therefore in practice when aggregating the scores of a miner we add a $\max(-,0)$ operation. 

The aggregated score of a miner that a validator sends to the blockchain is the following:

$$\frac{1}{N} \sum_j S_j$$

where $N$ is the number of events that the validator registered as settled during the tempo.

A simpler version of this model is, instead of paying the miner for their delta to the previous prediction, pay them for their delta to the Polymarket probability at the submission time i.e $S(p_j, o_i) - S(\text{price on polymarket at t}, o_i)$ where $p_j$ is submitted at $t$.

We also want to incorporate a progress or stability component in the scoring rule, as well as not introduce a latency game among miners to submit their predictions (as incentivized by the sequential scoring rule). 

<!--
## Incentive compability

See [here](docs/mechanism.md) for a discussion of our mechanism. -->

## Roadmap

- Scoring with exponentially decreasing weights until settlement date and linear differentiation mechanism - July 25th 
- Synthetic event generation with central resolution using ACLED data - early August
- Scoring with exponential differentiation mechanism, new entropy scoring component and new improvement rate scoring component - August/September
- Comprehensive and granular analytics - September
- Synthetic event generation from news data using an LLM - September
- Synthetic event generation with central resolution with various API modules: elections API, court rulings - data, space flights 
- Mining competition in partnership with Crunch DAO
- Synthetic event generation with UMA resolution - human verifiers resolve our events through the OOv2 
- Aggregation of miners’ predictions - through simple cutoff for benchmark events 
- Synthetic event generation with trustless resolution using UMA - we use the UMA Data Asserter framework for our event resolutions that then go through a challenge period
- More advanced aggregation mechanism based on sequential scoring 

Other items on our roadmap involve:
- commit-reveal on the miners' predictions
- make the prediction framework more LLM specific and create mechanisms that explicitely generate data for the fine-tuning of prediction focused LLMs
- consider other prediction markets such as Metaculus and Manifold (mostly as benchmark events)
- using Reuters or WSJ headlines for event generation

<!-- We first aim at adjusting the scoring rule by updating to a variation of the *model 2* described above. We will likely implement several other updates in order to make the mechanism more robust. One of them could be a commit-reveal step for the predictions submitted by miners. Some updates may be due to experimental data.

We would also possibly like to make the prediction framework more LLM specific and create mechanisms that explicitely generate data for the fine-tuning of prediction focused LLMs.

We plan to extend the set of predicted events to other prediction markets and event providers (Manifold, Metacalculus, Metacalculus). Our main goal is to obtain a continuous feed of organic events by using e.g WSJ headlines or Reuters' API. -->

<!-- In the limit, miners could update the weights of an entire LLM.-->

## Running a miner or validator

Regarding instructions and requirements, see [here](docs/validator.md) for validators and [here](docs/miner.md) for miners.

## Setting up a bittensor wallet
A detailed explanation of how to set up a wallet can be found [here](https://docs.bittensor.com/getting-started/wallets). 
We also provide some indications [here](docs/wallet-setup.md).


## References

| Reference ID | Author(s) | Year | Title |
|--------------|-----------|------|-------|
| 1 | Halawi and al. | 2024| [Approaching Human Level Forecasting with Langage Models](https://arxiv.org/html/2402.18563v1?s=35)|
| 2 | Luo and al. | 2024 | [LLM surpass human experts in predicting neuroscience results](https://arxiv.org/pdf/2403.03230.pdf)|


---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
