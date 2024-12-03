import os
from typing import Any, AsyncGenerator, List, Union

import bittensor as bt
from bittensor import AxonInfo

from infinite_games.protocol import EventPredictionSynapse


def query_miners(
    dendrite: bt.dendrite, axons: List[AxonInfo], synapse: EventPredictionSynapse
) -> List[Union[AsyncGenerator[Any, Any], EventPredictionSynapse]]:
    """Function that sends a query to miners and gets response"""
    timeout = float(os.environ.get("QUERY_TIMEOUT_SEC", "120"))
    bt.logging.debug(f"Query Timeout {timeout} seconds")
    responses = dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=axons,
        # Pass the synapse to the miner.
        synapse=synapse,
        # Do not deserialize the response so that we have access to the raw response.
        deserialize=False,
        timeout=timeout,
    )

    return responses
