class Memory:
    def __init__(self):
        self.store = {}

    def update(self, key, value):
        self.store[key] = value

    def compress(self):
        return {
            "columns": len(self.store.get("schema", {})),
            "outliers": {
                k: v for k, v in self.store.get("outliers", {}).items() if v > 0
            },
            "top_correlations": self.store.get("correlations", [])[:3]
        }
