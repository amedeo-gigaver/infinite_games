# Introducing repricing events

A repricing event is defined as a *significant* change in the odds of a prediction market some time $\Delta$ in the future. We create such events for Polymarket and we look in particular at the YES token of a given market (which pays out 1 dollar if the underlying event happens). Significant currently means at least $\delta$ where $\delta = \min(\gamma, 0.01)$ and where $\gamma$ is chosen so that the frequency of repricing over the past 30 days is approximately 1/2. Such repricing are most often due to a change in the news landscape (an interview, a new op-ed etc). At expiration, i.e 24 hours after the event was streamed to miners, we check whether a repricing occurred or not. We do this by querying the 'price-history' endpoint of the Polymarket API: https://docs.polymarket.com/#timeseries-data. We then compute the time weighted average over the past hour and check if it is outside the no-repricing range.

We will start with a list of [whitelisted markets](whitelist-repricing.md) for which those events will be created with a $\Delta$ equal to 24 hours.

Below is an example event format:
- event_id: 261a0940-2061-43f9-bd72-ea1a66d0b856
- cutoff: 1725544500
- title: Will there be a repricing in `Trump wins the presidential election of 2024` on 2024-09-07  7:59:59?
- description: This event resolves to `YES` if there is a repricing in the underlying YES token from 0.531 by 2024-09-07  17:59:59 UTC in the following Polymarket event https://polymarket.com/event/presidential-election-winner-2024/will-donald-trump-win-the-2024-us-presidential-election?tid=1725635465823. Repricing is considered every price outside the (inclusive) range [0.521, 0.541]. The event will be resolved by using the average price over the hour before the deadline, using the following polymarket query  https://clob.polymarket.com/prices-history?market=21742633143463906290569050155826241533067272736897614950488156847949938836455&startTs=1725717600&endTs=1725721199&fidelity=1
- start_date: 1725717600
- end_date: 1725721199
- answer: null

