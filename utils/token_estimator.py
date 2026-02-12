def estimate_tokens(text):
    # Rough GPT-style approximation
    # 1 token ≈ 4 characters (English average)
    return max(1, len(text) // 4)
