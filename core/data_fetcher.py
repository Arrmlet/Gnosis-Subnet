import bittensor as bt
from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger

class DataFetcher:
    def __init__(self):
        self.hf_api = HfApi()

    def get_source_info(self, repo_id: str) -> dict:
        """Get minimal required dataset info."""
        try:
            info = self.hf_api.dataset_info(repo_id=repo_id)
            return {
                'files': [s.rfilename for s in info.siblings if s.rfilename.endswith('.parquet')],
                'is_private': info.private,
                'tasks': info.card_data.get('task_ids', []) if info.card_data else []
            }
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            return None

    def fetch_data(self, source: str, filename: str):
        """Fetch specific parquet file."""
        try:
            dataset = load_dataset(
                source,
                data_files={'train': filename},
                split='train'
            )
            return dataset.to_pandas()
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None


if __name__ == '__main__':
    fetcher = DataFetcher()
    info = fetcher.get_source_info("arrmlet/reddit_dataset_44")
    print(info)