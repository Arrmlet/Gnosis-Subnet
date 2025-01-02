import unittest
import json
from protocol.base import DataAnalyticsRequest, DataAnalyticsResponse


def serialize_like_dendrite(synapse) -> str:
    return json.dumps(synapse.model_dump())  # Updated to use model_dump


def deserialize(json_str: str, cls):
    return cls(**json.loads(json_str))


class TestAnalyticsProtocol(unittest.TestCase):
    def test_request_serialization(self):
        # Create request
        request = DataAnalyticsRequest(
            source="arrmlet/reddit_dataset_44",
            filename="data/train-DataEntity_chunk_0.parquet",
            columns=["text", "datetime"]
        )

        # Test serialization
        serialized = serialize_like_dendrite(request)
        print("\nSerialized request:", serialized)

        # Test deserialization
        deserialized = deserialize(serialized, DataAnalyticsRequest)
        print("\nDeserialized request:", deserialized)

        # Assertions
        self.assertEqual(request.source, deserialized.source)
        self.assertEqual(request.filename, deserialized.filename)
        self.assertEqual(request.columns, deserialized.columns)

        print("\nTest passed successfully!")


if __name__ == "__main__":
    unittest.main(verbosity=2)