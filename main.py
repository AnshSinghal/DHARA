from logger import DharaLogger

logger = DharaLogger(__name__).get_logger()

try:
    logger.info("main.py started.")
    print("Hello from dhara-new!")
    logger.info("main.py finished successfully.")
except Exception as e:
    logger.error(f"Error in main.py: {e}")
    logger.exception(e)


if __name__ == "__main__":
    main()
