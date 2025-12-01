"""Main entry point for itmo-epml."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main function."""
    logger.info("Starting itmo-epml...")
    # Your code here


if __name__ == "__main__":
    main()
