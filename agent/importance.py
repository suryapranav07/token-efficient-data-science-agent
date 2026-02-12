import numpy as np

class ColumnImportance:

    @staticmethod
    def compute(df):
        scores = {}

        numeric_cols = df.select_dtypes(include=np.number).columns

        # Compute correlation matrix once
        corr = df.corr(numeric_only=True)

        for col in df.columns:
            missing_ratio = df[col].isnull().mean()

            score = 0

            # Numeric scoring
            if col in numeric_cols:
                variance = df[col].var()

                # Normalize variance (avoid extreme scaling)
                norm_variance = np.log1p(variance)

                # Max correlation magnitude with other columns
                max_corr = 0
                if col in corr.columns:
                    max_corr = max(
                        abs(corr[col].drop(col))
                    ) if len(corr[col].drop(col)) > 0 else 0

                score = norm_variance + max_corr

            else:
                # Categorical scoring
                unique_ratio = df[col].nunique() / len(df)
                score = unique_ratio

            # Penalize missing values
            score -= missing_ratio

            scores[col] = float(score)

        return scores
