import requests

from config.settings import get_settings
from utils.logger_exceptions import get_logger
from src.prompts_layer import get_system_prompt

logger = get_logger(__name__)

settings = get_settings()


class OpenRouterService:

    def __init__(self):

        self.api_key = (
            settings.openrouter_api_key
        )

        self.model = (
            settings.llm_model
        )

        self.base_url = (
            settings.openrouter_base_url
        )

        logger.info(
            "OpenRouter Service Initialized"        
        )

    def generate_answer(
        self,
        question: str,
        context: str,
        department: str,
    ) -> str:

        try:

            system_prompt = (
                get_system_prompt(
                    department
                )
            )

            user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "temperature": 0,
                "max_tokens": 1000,
            }

            headers = {
                "Authorization":
                    f"Bearer {self.api_key}",
                "Content-Type":
                    "application/json",
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            response.raise_for_status()

            data = response.json()

            answer = (
                data["choices"][0]
                ["message"]["content"]
            )

            logger.info(
                "Answer generated successfully"
            )

            return answer

        except Exception:

            logger.exception(
                "OpenRouter request failed"
            )

            raise

    def health_check(
        self,
    ) -> bool:

        try:

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ],
                "max_tokens": 5,
            }

            headers = {
                "Authorization":
                    f"Bearer {self.api_key}",
                "Content-Type":
                    "application/json",
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30,
            )

            return (
                response.status_code == 200
            )

        except Exception:

            logger.exception(
                "Health check failed"
            )

            return False