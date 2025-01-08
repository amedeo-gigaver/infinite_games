import asyncio

from neurons.validator.main import main
from neurons.validator.utils.logger.logger import logger

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit(0)
    except Exception:
        logger.exception("Unexpected error")
        exit(1)
