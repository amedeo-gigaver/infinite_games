# Release Notes

## [1.3.2] - 2024-12-10
- **Bittensor Upgrade**: Upgraded to Bittensor version 7.4.0.
- **Bug Fix**: Patched a recent common `SSLEOFError` encountered during block number retrieval.
- **Query Optimization**: Miners are now queried at a maximum frequency of once every 5 minutes to reduce unnecessary overhead.
- **Database Enhancements**: Improved data insertion logic for the local database, ensuring better reliability and performance.
- **Logging Improvements**: Enhanced logging and error reporting for better debugging and issue tracking.
- **Code Appearance**: Applied formatting and organization improvements using `black` and `isort`.