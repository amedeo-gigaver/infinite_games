# Release Notes

## [1.3.9] - 2025-02-19
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.0.1
- **Scoring**: Updated the scoring methodology by replacing the Brier scoring approach with Peer scoring.
- **Database**: Change database auto vacuum to incremental and add migration to vacuum & reclaim empty space

## [1.3.8] - 2025-02-11
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.0.0
- **Scoring**: Peer Scores are computed and stored in a dedicated table while continuing to use Brier scoring.
- **Validator API**: Introduced an optional API for validators, which can be enabled by setting the appropriate environment variable.
- **Maintenance**: Improved the sanitization of scores exported to the database used for the Mainnet dashboard.
- **Miner Update**: Added a new LLM forecaster based on [forecasting-tools](https://github.com/Metaculus/forecasting-tools)

## [1.3.7] - 2025-02-03
- **Events Resolution**: Events resolution and deletion upgraded to batched requests to improve overhead and reduce resolution time.

## [1.3.6] - 2025-01-27
- **Bittensor Upgrade**: Upgraded to Bittensor version 8.5.2
- **Requirements Update**: Removed unused pip requirements and switched the torch requirement to the CPU version. This significantly reduces the size of the required Python environment.
- **Maintenance**:
    - Added a task to gradually and regularly delete old records from the local database, preventing uncontrolled database growth.
    - In a future update, the database will be vacuumed to reclaim storage from deleted records.
- **Database Enhancements**: Integrated Alembic for managing all changes to the local database schema.
- **Data Exporting**: Increased the payload batch size for data export to improve efficiency.

## [1.3.5] - 2025-01-20
- **Scoring**: After scoring data analysis and simulations, we reduced the pre-normalization exponential factor from 30 to 5. This will prevent outlier miners to get disproportionate gains for some events. Additionally, it improves the chances of new miners to catchup with the existing miners.
- **Maintenance**: Removed deprecated validator code, reorganize existing miner code, cleanup dead code.

## [1.3.4] - 2025-01-13
- **Validator Architecture**: Validator architecture re-implemented to handle scaling events.

## [1.3.3] - 2024-12-17
- **Bittensor Upgrade**: Upgraded to Bittensor version 8.5.1
- **Default `netuid` for Validators**: The `--netuid` argument now defaults to 6, ensuring validators register on the intended subnet when the argument is not explicitly provided.
- **Logging Improvements**: Enhanced logging and error reporting for better debugging and issue tracking
- **Resilient Score Exporting**: Re-exporting functionality added to ensure scores from failed or missed past intervals are included in the current interval.

## [1.3.2] - 2024-12-10
- **Bittensor Upgrade**: Upgraded to Bittensor version 7.4.0.
- **Bug Fix**: Patched a recent common `SSLEOFError` encountered during block number retrieval.
- **Query Optimization**: Miners are now queried at a maximum frequency of once every 5 minutes to reduce unnecessary overhead.
- **Database Enhancements**: Improved data insertion logic for the local database, ensuring better reliability and performance.
- **Logging Improvements**: Enhanced logging and error reporting for better debugging and issue tracking.
- **Code Appearance**: Applied formatting and organization improvements using `black` and `isort`.