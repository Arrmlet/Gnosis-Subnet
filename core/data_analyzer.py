import pandas as pd
from typing import List, Dict, Any
import bittensor as bt


class DataAnalyzer:
    def __init__(self):
        pass

    def analyze_text_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze text column statistics."""
        if column not in df.columns:
            return {"error": f"Column {column} not found"}

        try:
            stats = {
                "total_rows": len(df),
                "null_count": df[column].isnull().sum(),
                "unique_count": df[column].nunique(),
                "avg_length": df[column].str.len().mean(),
                "min_length": df[column].str.len().min(),
                "max_length": df[column].str.len().max(),
            }
            return stats
        except Exception as e:
            bt.logging.error(f"Error analyzing column {column}: {str(e)}")
            return {"error": str(e)}

    def get_column_summaries(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get basic summaries for all columns."""
        summaries = {}
        for col in df.columns:
            summaries[col] = {
                "dtype": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "unique_count": df[col].nunique()
            }
        return summaries

    def process_batch(self, df: pd.DataFrame, batch_size: int = 1000):
        """Process dataframe in batches for memory efficiency."""
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            yield df.iloc[start_idx:end_idx]