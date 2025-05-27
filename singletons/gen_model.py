from custom.StreamableOpenRouterLLM import StreamableOpenRouterLLM
from dotenv import load_dotenv
import os
import random
from singletons.logger import get_logger


load_dotenv(override=True)

logger = get_logger()
_gen_model = None
# Flag to avoid retrying a persistently failing model too often
_primary_model_init_failed = False

GENERATIVE_MODEL_TEMPERATURE = 0.5


def get_openrouter_model_name():
    return "mistralai/mistral-7b-instruct:free"  # Example


# Add excepted_keys if you use it
def get_openrouter_api_key(excepted_keys=None):
    combined_keys = os.getenv("OPENROUTER_API_KEY")
    keys = combined_keys.split(",") if combined_keys else []
    logger.debug(f"Available OpenRouter API keys: {keys}")
    if len(keys) == 0:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    if excepted_keys:
        keys = [key for key in keys if key not in excepted_keys]
        if len(keys) == 0:
            raise ValueError(
                "No valid OPENROUTER_API_KEY available after filtering.")
    return random.choice(keys) if keys else None


def get_gen_model():
    global _gen_model, _primary_model_init_failed

    if _gen_model is not None:
        return _gen_model

    if _primary_model_init_failed:
        # If it failed before and we decided it's a persistent issue
        logger.error(
            "Primary LLM initialization previously failed and is marked as unavailable.")
        raise RuntimeError(
            "Primary LLM service is unavailable due to previous initialization failure.")

    try:
        logger.info("Attempting to initialize primary OpenRouter LLM...")
        api_key = get_openrouter_api_key()  # This will raise ValueError if key is missing

        _gen_model = StreamableOpenRouterLLM(
            api_key=api_key,
            temperature=GENERATIVE_MODEL_TEMPERATURE,
            model=get_openrouter_model_name(),
            stream=True,
        )
        logger.info(f"Successfully initialized primary OpenRouter LLM with model: "
                    f"{getattr(_gen_model, 'model', 'N/A')} and temperature: {getattr(_gen_model, 'temperature', 'N/A')}")
        _primary_model_init_failed = False  # Reset flag on success
        return _gen_model
    except ValueError as ve:  # Catch config errors like missing API key
        logger.error(
            f"Configuration error during LLM initialization: {ve}", exc_info=True)
        _primary_model_init_failed = True  # Mark as a persistent failure
        raise RuntimeError(f"LLM configuration error: {ve}") from ve
    except Exception as e:
        logger.error(
            f"Failed to initialize StreamableOpenRouterLLM: {e}", exc_info=True)
        # For other exceptions, you might not set _primary_model_init_failed=True
        # if you want to allow retries on subsequent calls, but for now, let's be strict.
        # _primary_model_init_failed = True
        raise RuntimeError(f"Failed to initialize LLM service: {e}") from e


def get_fallback_gen_model():
    global _gen_model  # To access the primary model's details if needed for fallback logic
    logger.info("Attempting to initialize fallback OpenRouter LLM...")
    try:
        # Implement your fallback logic, e.g., different model or API key
        # For simplicity, let's assume it tries to use a different (or same) API key
        # and potentially a different model if the primary one is known.

        # This is a simplified example. Your actual fallback might be more complex.
        api_key = get_openrouter_api_key(excepted_keys=[getattr(
            _gen_model, 'api_key', None)] if _gen_model else None)

        fallback_model_name = "openai/gpt-3.5-turbo"  # Example fallback model

        fallback_llm = StreamableOpenRouterLLM(
            api_key=api_key,
            temperature=GENERATIVE_MODEL_TEMPERATURE,
            model=fallback_model_name,
            stream=True,
        )
        logger.info(
            f"Successfully initialized fallback OpenRouter LLM with model: {fallback_model_name}")
        return fallback_llm
    except ValueError as ve:
        logger.error(
            f"Configuration error during fallback LLM initialization: {ve}", exc_info=True)
        raise RuntimeError(f"Fallback LLM configuration error: {ve}") from ve
    except Exception as e:
        logger.error(
            f"Failed to initialize fallback StreamableOpenRouterLLM: {e}", exc_info=True)
        raise RuntimeError(
            f"Failed to initialize fallback LLM service: {e}") from e
