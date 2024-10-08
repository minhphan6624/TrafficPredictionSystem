# System Imports
import sys

import pandas as pd

sys.dont_write_bytecode = True

import signal

# Project Imports
import gui.window as window
import utilities.logger as logger

# Global Variables
VERSION = "1.0.0"


def main():
    logger.log(f"Launching TPS GUI - Version {VERSION}")

    # Register Ctrl+C signal handler
    signal.signal(signal.SIGINT, signal_handler)

    window.run()


def signal_handler(sig, frame):
    logger.log("Exiting TPS GUI.")
    sys.exit(0)


if __name__ == "__main__":
    main()
