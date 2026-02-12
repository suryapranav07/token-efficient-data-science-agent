import pandas as pd
import numpy as np

class EDAEngine:
    def __init__(self, df):
        self.df = df

    def summarize_schema(self):
        summary = {}
        for col in self.df.columns:
            col_data = self.df[col]

            summary[col] = {
                "dtype": str(col_data.dtype),
                "missing_%": round(col_data.isnull().mean() * 100, 2),
                "unique": int(col_data.nunique())
            }

            if np.issubdtype(col_data.dtype, np.number):
                summary[col].update({
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std())
                })

        return summary
    
    def detect_outliers(self):
        outliers = {}

        for col in self.df.select_dtypes(include=np.number):
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            outliers[col] = int(count)

        return outliers
   
    def correlation_summary(self, threshold=0.6):
        corr = self.df.corr(numeric_only=True)
        strong = []

        for i, col in enumerate(corr.columns):
            for j in range(i + 1, len(corr.columns)):
                value = corr.iloc[i, j]
                if abs(value) > threshold:
                    strong.append((corr.columns[i], corr.columns[j], float(value)))

        return strong
