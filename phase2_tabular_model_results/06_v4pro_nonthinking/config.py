from harness import ModelConfig, disable_thinking_deepseek

CONFIG = ModelConfig(
    name="v4pro_nonthinking",
    model="deepseek-v4-pro",
    provider="deepseek",
    api_key_env="DEEPSEEK_API_KEY",
    price_in=0.435,
    price_out=0.87,
    thinking_disable=disable_thinking_deepseek,
    json_mode=False,
    default_subset="subsample_v4pro_nonthinking.json",
)
