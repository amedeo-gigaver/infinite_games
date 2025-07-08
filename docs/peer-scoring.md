## Definitions


There are miners $m\in M$ and questions $q \in Q$ with respective cutoff times $t_q$. For any $(m,q)$ a miner $m$ submits a time series 

$$
(p_{m,q,t})_{t \leq t_q}
$$

of forecasts ranging over $t \in T_q$ a list of time points depending on the question and preceding the cutoff $t_q$. If a miner does not submit a forecast for a given time step we denote his submission by $0_{m,q,t}$. 

Let $S(p_{m,q,t},o_q)$ be the score of a given prediction if the question resolved to $o_E \in \{0,1\}$ with value $1$ if the underlying event occurred and $0$ otherwise.

## Peer Score

For a given question $q$ at time step $t$, each miner's peer score measures how their prediction compares to those of all the other miners. With $n$ miners in total, in case the question resolves to $o_E$ the peer score for miner $m$ is computed as:

$$S(p_{m,q,t}, o_E) = \frac{1}{n} \sum_{j \neq m} \Bigl( \log(|o_E - p_{m,q,t}|) - \log(|o_E - p_{j,q,t}|) \Bigr)
= \log(|o_E - p_{m,q,t}|) - \frac{1}{n}\sum_{j \neq m} \log(|o_E - p_{j,q,t}|)$$.

In other words, this score reflects the difference between the logarithm of miner $m$'s prediction and the average logarithm of the predictions from all other miners.

To avoid problems with extreme predictions such as a miner confidently predicting $1$ when the outcome is $0$, which would yield a score of $-\infty$, we clip all predictions to the range $(0.01, 0.99)$.

## New miners

When a miner registers at a time $t$ on the subnet they will send predictions for questions $q$ that already opened at a time $t_{q,0} < t$. 

When this happens we give the new miner a score of $0$ which corresponds to the baseline, i.e when a miner is not bringing any information to the aggregate. We also give them a score of $0$ on the questions in the moving average which resolved before the miner registered.

## Penalty for not submitting a forecast

Unresponsive miners are penalised by imputing missing predictions as the value at 1/3 of the distance between the worst prediction and the average prediction. We denote the average prediction by $p_{q,t}^M$, the worst prediction by $p_{q,t}^W$, and the imputed prediction by $p_{q,t}^I$. We then have:

$$S(0_{m,q,t}, o_E) = S(p_{q,t}^I, o_E)$$ 

where $p_{q,t}^I = (p_{q,t}^M + \frac{1}{3}\times (p_{q,t}^W - p_{q,t}^M)$.


## Weights

We associate a weight $w_{q, t}$ to each prediction depending on the time of the submission $t \in T_q$. 


We choose exponentially decreasing weights along the intuition that predicting gets exponentially harder as one goes back in time. Denote $T_q = [A_q, B_q ]$.


We divide the time segment $[A_q,B_q]$ into $n$ intervals $[t_j, t_{j+1}]$ of equal length (currently 4 hours). Then for the interval $[t_j, t_{j+1}]$ we set the weight $w_{q,t_j} =  e^{-\frac{n}{n-j}+1}$ where $t_0 = A_q$ and $t_{n} = B_q$ and where $j$ increases from $0$ to $n-1$.

## Averaging per window

Each $p_{m,q,t}$ is in fact the arithmetic average of the miner's predictions in a given time window $[t_j, t_{j+1}]$ i.e

$$p_{m,q,t} = \frac{\sum_{t' \in [t_j, t_{j+1}]} p_{m,q,t'}}{\sum_{t' \in [t_j, t_{j+1}]} 1}$$



## Weighted average 


Given our weights ($w_{q,t}$) and the miner's time series $(p_{m,q,t})$ we compute the following time weighted average for each miner:

$$S_{m,q} = \frac{\sum_t w_{q,t}S(p_{m,q,t}, o_q)}{\sum_t w_{q,t}}$$

## Class Balanced Weights

We build a weighted moving average 

$$L_m = \frac{1}{\sum_{q \in W} w_{q}} \sum_{q \in W} S_{m,q}\,w_{q}$$ 

where $q$ ranges over the set $W$ of the last $N$ questions. The parameter $N$ is chosen to be proportional to the number of questions generated and resolved during an immunity period. 

The weights are class balanced weights whereby if $k = \sum_{q \in W} 1_{\{\text{outcome}(q)=1\}}$ is the number of events which resolved positively, we define 
$w_{1} = \frac{1}{q_Y}$ and $w_{0} = \frac{1}{1 - q_Y}$

where $q_Y = \frac{k + 1}{N + 2}$.

We then have $w_q = w_1$ if the outcome of $q$ is $1$ and vice versa.


## Extremisation

Next, to better distinguish consistently strong predictors from those who do not contribute meaningful signal, we transform this moving average using an extremisation step:

$$R_m = \max\bigl(L_m, 0\bigr)^2$$.

This step rewards miners with positive performance by squaring their average score, while effectively filtering out miners with non-positive averages.

Finally, each miner's weight is determined by normalizing these extremised scores:

$$W_m = \frac{R_m}{\sum_{m' \neq m} R_{m'}}$$.

