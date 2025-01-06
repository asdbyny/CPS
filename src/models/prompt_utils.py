def load_messages(model_name, prompt):
    templates = {
        # Models via API calls
        "claude-3-opus": [
            {"role": "user", "content": prompt},
        ],
        "claude-3-5-sonnet": [
            {"role": "user", "content": prompt},
        ],
        "deepseek-v2": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        "gemini-1.5-pro": prompt,  # Gemini uses prompt (string) instead of message list.
        "gpt-4": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "gpt-4o": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "gpt-4o-mini": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        # Models run locally
        "Deepseek-math-7b-rl": [
            {"role": "user", "content": prompt},
        ],
        "Internlm2-math-20b": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "Llama-3-70B": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "Mixtral-8x22B": [{"role": "user", "content": prompt}],
        "Qwen1.5-72B": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "Yi-1.5-34B": [{"role": "user", "content": prompt}],
    }
    return templates.get(model_name)
