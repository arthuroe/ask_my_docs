import os
import logging
import json
import requests
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterFreeAdapter:
    """Adapter for accessing only free LLMs through OpenRouter.ai API"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize the OpenRouter adapter for free models only.

        Args:
            api_key: OpenRouter API key. If None, will try to load from environment.
            base_url: Base URL for the OpenRouter API.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning(
                "No OpenRouter API key provided. Using limited free access.")

        self.base_url = base_url
        self.app_url = ""

        # Get app info for better tracking
        self.app_name = os.getenv("APP_NAME", "AskMyDocs")

        self.update_best_free_model()

    def update_best_free_model(self) -> bool:
        """
        Find and set the best available free model.

        Returns:
            Boolean indicating success.
        """
        free_models = self.list_free_models()

        if not free_models:
            # If API call fails, use fallback list of known free models
            logger.warning(
                "Could not retrieve free models list. Using fallback models.")
            self.model = self._get_fallback_model()
            return False

        # Sort models by preference:
        # 1. Llama 4 models (highest priority)
        # 2. Gemini models
        # 3. Mistral models
        # 4. DeepSeek models
        # 5. Others
        ranked_models = self._rank_free_models(free_models)

        if ranked_models:
            self.model = ranked_models[0]["id"]
            logger.info(f"Selected free model: {self.model}")
            return True
        else:
            self.model = self._get_fallback_model()
            logger.warning(
                f"No suitable free models found. Using fallback: {self.model}")
            return False

    def _rank_free_models(self, free_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank free models by preference for document QA tasks.

        Args:
            free_models: List of free model dictionaries.

        Returns:
            Sorted list of models by preference.
        """
        # Define preference tiers
        tier_1_patterns = ["llama-4", "llama4"]
        tier_2_patterns = ["gemini", "claude"]
        tier_3_patterns = ["mistral", "mixtral"]
        tier_4_patterns = ["deepseek"]

        # Helper function to determine tier
        def get_model_tier(model_id: str) -> int:
            model_id_lower = model_id.lower()

            # Check for free tag/suffix
            is_free = ":free" in model_id_lower or "-free" in model_id_lower
            if not is_free:
                return 99  # Deprioritize non-free models

            # Check pattern matches
            for pattern in tier_1_patterns:
                if pattern in model_id_lower:
                    return 1

            for pattern in tier_2_patterns:
                if pattern in model_id_lower:
                    return 2

            for pattern in tier_3_patterns:
                if pattern in model_id_lower:
                    return 3

            for pattern in tier_4_patterns:
                if pattern in model_id_lower:
                    return 4

            return 5  # Other free models

        # Sort by tier, then by context length (longer is better)
        ranked_models = sorted(
            free_models,
            key=lambda m: (
                get_model_tier(m["id"]),
                # Negative to sort in descending order
                -m.get("context_length", 0)
            )
        )

        return ranked_models

    def _get_fallback_model(self) -> str:
        """
        Get a fallback model if API calls fail.

        Returns:
            Model ID string for a known free model.
        """
        # List of known free models, ordered by preference
        fallback_models = [
            "meta-llama/llama-4-scout:free",
            "google/gemini-2.5-pro-exp-03-25:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "deepseek/deepseek-v3-base:free",
            "nousresearch/deephermes-3-llama-3-8b-preview:free",
            "huggingfaceh4/zephyr-7b-beta"  # Always fallback to this older but reliable one
        ]

        return fallback_models[0]

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for OpenRouter API requests.

        Returns:
            Dictionary of headers.
        """
        headers = {
            "Content-Type": "application/json"
        }

        # Add API key if available
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        headers["HTTP-Referer"] = self.app_url
        headers["X-Title"] = self.app_name

        return headers

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on OpenRouter.

        Returns:
            List of model information dictionaries.
        """
        try:
            headers = self._get_headers()

            response = requests.get(
                f"{self.base_url}/models",
                headers=headers
            )

            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                logger.error(
                    f"Error listing models: {response.status_code} - {response.text}"
                )
                return []

        except Exception as e:
            logger.error(f"Exception listing models: {str(e)}")
            return []

    def list_free_models(self) -> List[Dict[str, Any]]:
        """
        List models that are free to use on OpenRouter.

        Returns:
            List of free model information dictionaries.
        """
        # Get all models
        models = self.list_models()

        # Filter for free models - looking for multiple indicators
        free_models = []
        for model in models:
            model_id = model.get("id", "").lower()
            pricing = model.get("pricing", {})

            # Check various indicators that a model is free
            is_free = False

            # Check for explicit free tag in model ID
            if ":free" in model_id or "-free" in model_id:
                is_free = True

            # Check for zero pricing
            elif (pricing.get("prompt") == 0 and pricing.get("completion") == 0):
                is_free = True

            # Check for free_tier indicator if present
            elif model.get("free_tier", False):
                is_free = True

            if is_free:
                free_models.append(model)

        # Log the number of free models found
        logger.info(f"Found {len(free_models)} free models on OpenRouter")

        return free_models

    def _handle_streaming_response(self, response):
        """
        Handle streaming response from OpenRouter API.

        Args:
            response: Response object from requests.

        Returns:
            Combined text from streaming response.
        """
        result = ""

        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')

                # Remove the "data: " prefix
                if line_text.startswith("data: "):
                    line_text = line_text[6:]

                # Skip keep-alive lines
                if line_text.strip() == "[DONE]":
                    break

                try:
                    # Parse the JSON
                    json_data = json.loads(line_text)

                    # Extract the text
                    if "choices" in json_data and json_data["choices"]:
                        delta = json_data["choices"][0].get("delta", {})
                        if "content" in delta:
                            result += delta["content"]
                except json.JSONDecodeError:
                    pass

        return result

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """
        Generate text using OpenRouter API with a free model.

        Args:
            prompt: The prompt to send to the model.
            temperature: Controls randomness. Lower is more deterministic.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.

        Returns:
            Generated text from the model.
        """
        # Ensure we have a model selected
        if not self.model:
            self.update_best_free_model()

        # If still no model, return error
        if not self.model:
            return "Error: No free models available on OpenRouter."

        try:
            headers = self._get_headers()

            # Use OpenAI-compatible format for the request
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response
                    return self._handle_streaming_response(response)
                else:
                    # Handle regular response
                    content = response.json(
                    )["choices"][0]["message"]["content"]
                    # Log model usage for tracking
                    usage = response.json().get("usage", {})
                    logger.info(
                        f"Used model {self.model} - Input: {usage.get('prompt_tokens', 0)}, Output: {usage.get('completion_tokens', 0)}")
                    return content
            else:
                error_info = f"Error {response.status_code}"
                try:
                    error_detail = response.json()
                    error_message = error_detail.get(
                        "error", {}).get("message", "Unknown error")
                    error_info = f"{error_info}: {error_message}"
                except:
                    error_info = f"{error_info}: {response.text}"

                logger.error(f"Error generating text: {error_info}")

                # Check for specific error cases
                if "rate limit" in error_info.lower():
                    return "Error: Rate limit exceeded for this free model. Please try again later or try a different model."

                # If there's an issue with the model, try to get a different one
                if "model" in error_info.lower() or "no endpoints" in error_info.lower():
                    prev_model = self.model
                    if self.update_best_free_model() and self.model != prev_model:
                        logger.info(
                            f"Retrying with different free model: {self.model}")
                        return self.generate(prompt, temperature, max_tokens, stream)

                return f"Error: Failed to generate response. {error_info}"

        except Exception as e:
            logger.error(f"Exception during text generation: {str(e)}")
            return f"Error: {str(e)}"


class OpenRouterFreeChain:
    """Chain for handling Q&A with OpenRouter free LLMs"""

    def __init__(self, adapter: OpenRouterFreeAdapter):
        """
        Initialize the OpenRouter free chain.

        Args:
            adapter: An initialized OpenRouterFreeAdapter.
        """
        self.adapter = adapter

    def create_prompt(self, query: str, context: List[str]) -> str:
        """
        Create a prompt for the LLM based on the query and context.

        Args:
            query: The user's question.
            context: List of document contents to provide as context.

        Returns:
            Formatted prompt string.
        """
        context_str = "\n\n".join(
            [f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])

        prompt = f"""You are an AI assistant answering questions based on the provided documents.

Context information:
{context_str}

Based on the above context, please answer the following question:
{query}

If the information to answer the question is not contained in the provided documents, respond with: "I don't have enough information in the provided documents to answer this question."

Answer:"""

        return prompt

    def run(self, query: str, context: List[str]) -> str:
        """
        Run the chain to get an answer.

        Args:
            query: The user's question.
            context: List of document contents to provide as context.

        Returns:
            Answer from the model.
        """
        prompt = self.create_prompt(query, context)
        return self.adapter.generate(prompt)


def get_best_free_model() -> str:
    """
    Get the best available free model from OpenRouter.

    Returns:
        Model ID string for the recommended free model.
    """
    adapter = OpenRouterFreeAdapter()
    adapter.update_best_free_model()
    return adapter.model
