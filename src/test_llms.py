from models import ModelWrapper


def main():
    # Models via API calls
    # model_name = "claude-3-opus"
    # model_name = "deepseek-v2"
    # model_name = "gemini-1.5-pro"
    # model_name = "gpt-4"
    model_name = "gpt-4o"

    # Models run locally
    # model_name = "Deepseek-math-7b-rl"
    # model_name = "Internlm2-math-20b"
    # model_name = "Llama-3-70B"
    # model_name = "Mixtral-8x22B"
    # model_name = "Qwen1.5-72B"
    # model_name = "Yi-1.5-34B"

    model = ModelWrapper(model_name)
    prompt = "What is 2+2?"
    response = model.generate_response(prompt)
    print(response)


if __name__ == "__main__":
    main()
