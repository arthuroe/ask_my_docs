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

        # Get app info for better tracking
        self.app_name = os.getenv("APP_NAME", "AskMyDocs")

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
