import os
import argparse
from pathlib import Path
import bittensor as bt
from loguru import logger
from enum import auto, Enum
from dotenv import load_dotenv

load_dotenv()


class NeuronType(Enum):
    MINER = auto()
    VALIDATOR = auto()


def check_config(config: bt.config):
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )

    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)


def add_args(neuron_type: NeuronType, parser):
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="Epoch length in blocks",
        default=100
    )

    if neuron_type == NeuronType.MINER:
        parser.add_argument(
            "--data.source",
            type=str,
            help="Data source path",
            default=""
        )

        parser.add_argument(
            "--analytics.batch_size",
            type=int,
            help="Batch size for processing",
            default=1000
        )

        parser.add_argument(
            "--neuron.database_name",
            type=str,
            help="Database name",
            default="analytics.sqlite"
        )

        parser.add_argument(
            "--offline",
            action="store_true",
            help="Run in offline mode",
            default=False
        )


def create_config(neuron_type: NeuronType):
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    add_args(neuron_type, parser)

    return bt.config(parser)