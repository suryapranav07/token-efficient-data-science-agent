import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Token-Efficient Data Science Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Token-Efficient Data Science Agent")
st.markdown(
    "An adaptive data analysis system that compresses dataset schemas "
    "and reduces LLM token costs intelligently."
)
st.divider()

# ------------------------------
# Helper Functions
# ------------------------------

def estimate_tokens(text):
    return max(1, len(text) // 4)


def summarize_schema(df):
    summary = {}
    for col in df.columns:
        col_data = df[col]

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


def detect_outliers(df):
    outliers = {}
    for col in df.select_dtypes(include=np.number):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outliers[col] = int(count)
    return outliers


def correlation_summary(df, threshold=0.6):
    corr = df.corr(numeric_only=True)
    strong = []

    for i, col in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]
            if abs(value) > threshold:
                strong.append((corr.columns[i], corr.columns[j], float(value)))

    return strong


def compute_importance(df):
    scores = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df.corr(numeric_only=True)

    for col in df.columns:
        missing_ratio = df[col].isnull().mean()
        score = 0

        if col in numeric_cols:
            variance = df[col].var()
            norm_variance = np.log1p(variance)

            max_corr = 0
            if col in corr.columns:
                vals = corr[col].drop(col)
                if len(vals) > 0:
                    max_corr = max(abs(vals))

            score = norm_variance + max_corr
        else:
            unique_ratio = df[col].nunique() / len(df)
            score = unique_ratio

        score -= missing_ratio
        scores[col] = float(score)

    return scores


def adaptive_compress(schema, importance_scores, top_k=3):
    sorted_cols = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_columns = [col for col, _ in sorted_cols[:top_k]]

    lines = []

    for col, stats in schema.items():
        if col in top_columns:
            line = f"{col} ({stats['dtype']}), missing={stats['missing_%']}%, unique={stats['unique']}"
            if "mean" in stats:
                line += f", mean={round(stats['mean'],2)}, std={round(stats['std'],2)}"
        else:
            line = f"{col} ({stats['dtype']}), missing={stats['missing_%']}%"

        lines.append(line)

    return "\n".join(lines)


# ------------------------------
# Upload Section
# ------------------------------

st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file to analyze and compress",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean empty columns
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # ------------------------------
    # Core Processing
    # ------------------------------

    schema = summarize_schema(df)
    outliers = detect_outliers(df)
    correlations = correlation_summary(df)
    importance_scores = compute_importance(df)

    compressed_schema = adaptive_compress(schema, importance_scores)

    raw_text = df.to_csv(index=False)
    compressed_text = compressed_schema + str(outliers) + str(correlations)

    raw_tokens = estimate_tokens(raw_text)
    compressed_tokens = estimate_tokens(compressed_text)

    reduction = round((1 - compressed_tokens/raw_tokens) * 100, 2)
    compression_ratio = round(raw_tokens / compressed_tokens, 2)

    # ------------------------------
    # Dashboard Overview
    # ------------------------------

    st.header("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))

    with col2:
        st.metric("Raw Token Estimate", raw_tokens)
        st.metric("Compressed Token Estimate", compressed_tokens)

    st.markdown("### 🚀 Compression Performance")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Token Reduction", f"{reduction}%")

    with col4:
        st.metric("Compression Ratio", f"{compression_ratio}x")

    # ------------------------------
    # Importance Ranking
    # ------------------------------

    st.header("🔍 Column Importance Ranking")

    importance_df = pd.DataFrame(
        sorted(importance_scores.items(), key=lambda x: x[1], reverse=True),
        columns=["Column", "Importance Score"]
    )

    importance_df["Importance Score"] = importance_df["Importance Score"].round(3)

    st.dataframe(importance_df, use_container_width=True)

    # ------------------------------
    # Token Visualization
    # ------------------------------

    st.header("📉 Token Comparison")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["Raw", "Compressed"], [raw_tokens, compressed_tokens])
    ax.set_ylabel("Token Count")
    ax.set_title("Raw vs Compressed Token Usage")

    st.pyplot(fig)

    # ------------------------------
    # Cost Simulation (INR)
    # ------------------------------

    st.header("💰 LLM Cost Simulation (INR)")

    price_per_1k_tokens_usd = 0.03
    usd_to_inr = 83

    raw_cost_inr = (raw_tokens / 1000) * price_per_1k_tokens_usd * usd_to_inr
    compressed_cost_inr = (compressed_tokens / 1000) * price_per_1k_tokens_usd * usd_to_inr
    cost_savings_inr = raw_cost_inr - compressed_cost_inr

    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric("Raw Cost (₹)", round(raw_cost_inr, 2))

    with col6:
        st.metric("Compressed Cost (₹)", round(compressed_cost_inr, 2))

    with col7:
        st.metric("Savings (₹)", round(cost_savings_inr, 2))

    # ------------------------------
    # Footer
    # ------------------------------

    st.divider()
    st.caption(
        "Built as part of an AI optimization challenge. "
        "Demonstrates adaptive schema compression and token-efficient analysis."
    )
