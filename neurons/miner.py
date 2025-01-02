import bittensor as bt
import threading
import time
from core.data_fetcher import DataFetcher
from core.data_analyzer import DataAnalyzer
from neurons.config import check_config, create_config, NeuronType
from typing import Dict, Tuple


class Miner:
    def __init__(self, config=None):
        self.config = config or create_config(NeuronType.MINER)
        check_config(self.config)

        # Initialize components
        self.fetcher = DataFetcher(self.config)
        self.analyzer = DataAnalyzer()

        # Initialize state
        self.lock = threading.Lock()
        self.should_exit = False
        self.step = 0
        self.last_sync_timestamp = None

        if not self.config.offline:
            self.setup_subtensor()
            self.setup_axon()

    def setup_subtensor(self):
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

    def setup_axon(self):
        bt.logging.info("Setting up axon...")
        self.axon = bt.axon(wallet=self.wallet)
        self.axon.attach(
            forward_fn=self.process_dataset_request,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority
        ).start()
        bt.logging.info(f"Axon created: {self.axon}")

    def sync(self):
        if not self.config.offline:
            self.check_registered()
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            bt.logging.info(f"Synced metagraph: {self.metagraph}")

    def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
                netuid=self.config.netuid,
                hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(f"Wallet: {self.wallet} not registered.")
            exit(1)

    async def process_dataset_request(self, synapse) -> Dict:
        try:
            df = self.fetcher.fetch_data(
                source=synapse.source,
                filename=getattr(synapse, 'filename', None)
            )

            if df is None:
                return {"error": "Failed to fetch dataset"}

            analysis_results = {}
            for col in synapse.columns:
                if col in df.columns:
                    analysis_results[col] = self.analyzer.analyze_text_column(df, col)

            return {
                "source_info": self.fetcher.get_source_info(synapse.source),
                "analysis": analysis_results
            }

        except Exception as e:
            bt.logging.error(f"Error processing request: {str(e)}")
            return {"error": str(e)}

    async def blacklist(self, synapse) -> Tuple[bool, str]:
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        not_validator = not self.metagraph.validator_permit[caller_uid]
        return not_validator, "Caller not a validator" if not_validator else ""

    async def priority(self, synapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    def run(self):
        bt.logging.info(f"Starting miner on network: {self.config.subtensor.chain_endpoint}")

        if not self.config.offline:
            self.sync()
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        while not self.should_exit:
            time.sleep(60)
            if not self.config.offline:
                self.sync()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.should_exit = True


if __name__ == "__main__":
    with Miner() as miner:
        miner.run()