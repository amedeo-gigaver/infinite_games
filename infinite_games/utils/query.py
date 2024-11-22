import os
from typing import Any, AsyncGenerator, List, Union
import bittensor as bt
from bittensor import AxonInfo

from infinite_games.protocol import EventPredictionSynapse


def query_miners(
        dendrite: bt.dendrite, axons: List[AxonInfo],
        synapse: EventPredictionSynapse
) -> List[
    Union[AsyncGenerator[Any, Any], EventPredictionSynapse]
]:
    """Function that sends a query to miners and gets response"""
    bt.logging.debug(f"Query Timeout {float(os.environ.get('QUERY_TIMEOUT_SEC', '60'))}")
    responses = dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=axons,
        # Pass the synapse to the miner.
        synapse=synapse,
        # Do not deserialize the response so that we have access to the raw response.
        deserialize=False,
        timeout=float(os.environ.get('QUERY_TIMEOUT_SEC', '60'))
    )
    return responses
