import logging

def get_logger(name):
    logger = logging.getLogger(name)
    return logger
#has to be done first
def init_root_logger(filename):
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    )
    print("Initialized root logger")