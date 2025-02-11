## Definitions


There are miners $m\in M$ and questions $q \in Q$ with respective cutoff times $t_q$. For any $(m,q)$ a miner $m$ submits a time series $(p_{m,q,t})_{t\leq t_q}$ of forecasts ranging over $t \in T_q$ a list of time points depending on the question and preceding the cutoff $t_q$. If a miner does not submit a forecast we denote his submission by $p_{\emptyset}$. 

Let $S(p_{m,q,t},o_q)$ be the score of a given prediction if the question resolved to $o_E \in \{0,1\}$ with value $1$ if the underlying event occurred and $0$ otherwise.

## Peer score

For a set of $n$ miners the peer score on a question $q$ and time step $t$ takes the following form:

$S(p_{m,q,t},o_q)=\frac{1}{n}\sum_{j \neq m} (\log(p_{m,q,t}) - \log(p_{j,q,t})) = log(p_{m,q,t}) - log(\text{GM}(p_{j,q,t})_{j \neq m})$

where $\text{GM}(p_{j,q,t})_{j \neq m})$ is the geometric mean of the predictions of all the miners except $m$ for the time step $t$.

In order to prevent instabilities we clip the miners predictions. Indeed otherwise one confident and wrong prediction, e.g sending $1$ while the question resolves to $0$, would result in a score of $-\infty$ de facto eliminating the miner. We clip the predictions to $(0.1, 0.99)$.

In order to incentivise full coverage we have the following rule:
$S(p_{\emptyset},o_q) = \text{worst possible score on a given question}$


## Weights

We associate a weight $w_{q, t}$ to each depending on the time of the submission $t \in T_q$. 


We choose exponentially decreasing weights along the intuition that predicting gets exponentially harder as one goes back in time. Denote $T_q = [A_q, B_q ]$.


We divide the time segment $[A_q,B_q]$ into $n$ intervals $[t_j, t_{j+1}]$ of equal length (currently 4 hours). Then for the interval $[t_j, t_{j+1}]$ we set the weight $w_{q,t_j} =  e^{-\frac{n}{n-j}+1}$ where $t_0 = A_q$ and $t_{n} = B_q$ and where $j$ increases from $0$ to $n-1$.

## Averaging per window

Each $p_{m,q,t}$ is in fact the arithmetic average of the miner's predictions in a given time window $[t_j, t_{j+1}]$ i.e

$p_j = \frac{\sum_{t' \in [t_j, t_{j+1}]} p_{t'}}{\sum_{t' \in [t_j, t_{j+1}]} 1}$



## Weighted average 


Given our weights ($w_{q,t}$) and the miner's time series $(p_{m,q,t})$ we compute the following time weighted average for each miner:

$S_{m,q} = \frac{\sum_t w_{q,t}S(p_{m,q,t}, o_q)}{\sum_t w_{q,t}}$

## Moving average and extremisation

We build a moving average $L_{m,q} = \sum_q S_{m,q}$ where $q$ ranges over the last $N$ questions. The parameter $N$ is chosen to be the average number of questions a miner would get during the immunity period.

We finally extremise this moving average to reward more better predictors and filter our miners which on average do not contribute any signal.

$R_{m,q}= \max(L_{m,q}, 0)^2$

The weight of the miner is then the normalised value:

$W_{m,q}= \frac{R_{m,q}}{\sum_{m' \neq m}R_{m',q}}$


## New miners

When a miner registers at a time $t$ on the subnet they will send predictions for questions $q$ that already opened at a time $t_{q,0} < t$. When this happens we give the new miner a score of $0$ which corresponds to the baseline, i.e when a miner is neither bringing new information compared to the aggregate nor is penalized. We will also give them a score of $0$ on the questions in the moving average which resolved before the miner registered.


