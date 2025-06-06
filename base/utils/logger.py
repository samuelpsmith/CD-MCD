#
#Logging module for custom logging
#
import logging

#Params: String name
#Returns: logger instance
def get_logger(name):
    logger = logging.getLogger(name)
    return logger
#has to be done first
#Params: String filename - logging file name
#Returns: Void
#Does: Applies settings for root logger
def init_root_logger(filename):
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    )
    print("Initialized root logger")