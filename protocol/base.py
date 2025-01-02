from typing import Dict, List, Optional, Tuple
import bittensor as bt
from pydantic import Field, ConfigDict

class BaseProtocol(bt.Synapse):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    version: Optional[int] = Field(default=None)

class DataAnalyticsRequest(BaseProtocol):
    """Protocol for data analytics requests"""
    source: str = Field(
        description="Data source path or identifier"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Specific file to analyze"
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Columns to analyze"
    )
    batch_size: Optional[int] = Field(
        default=1000,
        description="Batch size for processing"
    )

class DataAnalyticsResponse(BaseProtocol):
    """Protocol for data analytics responses"""
    source_info: Dict = Field(
        default_factory=dict,
        description="Source metadata"
    )
    analysis_results: Dict = Field(
        default_factory=dict,
        description="Analysis results by column"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if request failed"
    )

# Request limits per validation period
REQUEST_LIMITS = {
    DataAnalyticsRequest: 5,
    DataAnalyticsResponse: 5
}