import pandas as pd
from datasets import load_dataset
import numpy as np
from loguru import logger
from transformers import pipeline


class DataProcessor:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            max_length=512,
            truncation=True
        )

    @staticmethod
    def batch_texts(texts, batch_size=32):
        """Helper to batch texts for processing"""
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

    def analyze_sentiment(self, texts):
        results = []
        for batch in self.batch_texts(texts):
            batch_results = self.sentiment_analyzer(batch)
            results.extend(batch_results)

        sentiment_counts = {
            'POS': sum(1 for r in results if r['label'] == 'POS'),
            'NEU': sum(1 for r in results if r['label'] == 'NEU'),
            'NEG': sum(1 for r in results if r['label'] == 'NEG')
        }

        return {
            "sentiment_distribution": sentiment_counts,
            "positive_ratio": sentiment_counts['POS'] / len(results),
            "negative_ratio": sentiment_counts['NEG'] / len(results),
            "neutral_ratio": sentiment_counts['NEU'] / len(results)
        }

    def analyze_dataset(self, source: str, filename: str, columns: list, sample_size=1000):
        try:
            dataset = load_dataset(
                source,
                data_files={'train': filename},
                split='train'
            )

            df = dataset.to_pandas()
            logger.info(f"Available columns: {df.columns.tolist()}")

            results = {}
            for col in columns:
                if col not in df.columns:
                    continue

                if 'text' in col.lower():
                    # Basic text analytics
                    basic_stats = self.process_text(df[col])

                    # Sentiment analysis on sample
                    sample_texts = df[col].sample(n=min(sample_size, len(df))).tolist()
                    sentiment_stats = self.analyze_sentiment(sample_texts)

                    results[col] = {**basic_stats, "sentiment": sentiment_stats}

                elif 'date' in col.lower() or 'time' in col.lower():
                    results[col] = self.process_datetime(df[col])

            return {
                "analysis": results,
                "metadata": {
                    "total_rows": len(df),
                    "columns": df.columns.tolist(),
                    "memory_usage": df.memory_usage(deep=True).sum() / 1024 ** 2
                }
            }

        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return {"error": str(e)}

    @staticmethod
    def process_text(series):
        return {
            "avg_length": float(series.str.len().mean()),
            "max_length": int(series.str.len().max()),
            "min_length": int(series.str.len().min()),
            "null_count": int(series.isnull().sum()),
            "unique_count": int(series.nunique())
        }

    @staticmethod
    def process_datetime(series):
        series = pd.to_datetime(series)
        return {
            "min_date": series.min().isoformat(),
            "max_date": series.max().isoformat(),
            "null_count": int(series.isnull().sum()),
            "unique_count": int(series.nunique())
        }

def test_processor():
    import json
    processor = DataProcessor()
    results = processor.analyze_dataset(
        "arrmlet/reddit_dataset_44",
        "data/train-DataEntity_chunk_0.parquet",
        ["text", "datetime"],
        sample_size=10_000  # Smaller sample for testing
    )

    print("\nProcessor results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    test_processor()