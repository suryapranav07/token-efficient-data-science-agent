from .eda_engine import EDAEngine
from .memory import Memory
from .compressor import SchemaCompressor
from .importance import ColumnImportance


class DataScienceAgent:
    def __init__(self, df):
        self.df = df
        self.eda = EDAEngine(df)
        self.memory = Memory()

    def run(self):
        schema = self.eda.summarize_schema()
        outliers = self.eda.detect_outliers()
        correlations = self.eda.correlation_summary()

        importance_scores = ColumnImportance.compute(self.df)

        self.memory.update("schema", schema)
        self.memory.update("outliers", outliers)
        self.memory.update("correlations", correlations)

        compressed_schema = SchemaCompressor.compress(
            schema,
            importance_scores,
            top_k=3
        )

        compressed_memory = self.memory.compress()

        return compressed_schema, compressed_memory, importance_scores

