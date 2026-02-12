class SchemaCompressor:

    @staticmethod
    def compress(schema, importance_scores, top_k=3):
        # Sort columns by importance
        sorted_cols = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_columns = [col for col, _ in sorted_cols[:top_k]]

        lines = []

        for col, stats in schema.items():

            if col in top_columns:
                # Detailed summary
                line = f"{col} ({stats['dtype']}), missing={stats['missing_%']}%, unique={stats['unique']}"

                if "mean" in stats:
                    line += f", mean={round(stats['mean'],2)}, std={round(stats['std'],2)}"

            else:
                # Light summary
                line = f"{col} ({stats['dtype']}), missing={stats['missing_%']}%"

            lines.append(line)

        return "\n".join(lines)
