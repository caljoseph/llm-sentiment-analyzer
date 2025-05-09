import litellm
import base64
import os
import mimetypes
from typing import Dict, Optional, Any, Tuple, Union
from pydantic import BaseModel

# Configure litellm to drop unsupported parameters
litellm.drop_params = True


class ModelTooLargeError(Exception):
    pass


class UnsupportedMediaTypeError(Exception):
    pass


class ModelDoesNotSupportVisionError(Exception):
    pass


class ModelResponse(BaseModel):
    raw_output: Any
    extracted_label: Optional[int] = None
    logprobs: Optional[Dict[str, Any]] = None


def encode_image_file(image_path: str) -> str:
    """
    Encode an image file to base64 format for API requests.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded image with appropriate prefix
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Get file extension
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()

    # Map common extensions to mime types
    mime_type = "image/jpeg"  # Default
    if ext in [".png"]:
        mime_type = "image/png"
    elif ext in [".gif"]:
        mime_type = "image/gif"
    elif ext in [".webp"]:
        mime_type = "image/webp"

    # Format with data URI scheme
    return f"data:{mime_type};base64,{encoded_image}"


class ModelManager:
    def __init__(self, model_name: str, api_base: Optional[str] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.api_base = api_base
        self.model_kwargs = model_kwargs or {}
        self.supports_vision = litellm.supports_vision(model=model_name)

    def _extract_label_from_output(self, output: str) -> int:
        # Look for a label 1-5 in the output
        for char in output:
            if char in "12345":
                return int(char)
        return 0  # Invalid or not found

    def _extract_logprobs(self, response: Any) -> Optional[Dict[str, float]]:
        # Extract token logprobs from the response
        try:
            logprobs_dict = {}

            # Access the logprobs from the first choice
            if (hasattr(response, 'choices') and
                    len(response.choices) > 0 and
                    hasattr(response.choices[0], 'logprobs') and
                    response.choices[0].logprobs is not None and
                    hasattr(response.choices[0].logprobs, 'content') and
                    response.choices[0].logprobs.content is not None):

                # Iterate through content items
                for content_item in response.choices[0].logprobs.content:
                    # Get top logprobs for each token
                    if hasattr(content_item, 'top_logprobs'):
                        for logprob_item in content_item.top_logprobs:
                            # Store token and its logprob value
                            if hasattr(logprob_item, 'token') and hasattr(logprob_item, 'logprob'):
                                logprobs_dict[logprob_item.token] = logprob_item.logprob

            return logprobs_dict if logprobs_dict else None

        except Exception as e:
            print(f"Error extracting logprobs: {str(e)}")
            return None

    async def generate_sentiment_rating(self,
                                        content: Optional[str] = None,
                                        content_path: Optional[str] = None,
                                        system_prompt: str = "",
                                        user_prompt: str = "",
                                        temperature: float = 1.0,
                                        max_tokens: int = 50,
                                        logprobs: bool = True,
                                        top_logprobs: int = 20) -> ModelResponse:
        
        # Check if both content and content_path are provided or both are None
        if (content is None and content_path is None) or (content is not None and content_path is not None):
            raise ValueError("You must provide either content OR content_path, not both or neither")

        # If content_path is provided, check if the model supports vision
        if content_path is not None and not self.supports_vision:
            raise ModelDoesNotSupportVisionError(
                f"Model {self.model_name} does not support vision inputs. Use a vision-capable model like gpt-4-vision-preview."
            )

        # Prepare message content appropriately
        if content_path is not None:
            # Vision prompt with image
            try:
                encoded_image = encode_image_file(content_path)
                message_content = [
                    {"type": "text", "text": user_prompt.format(content="")},
                    {
                        "type": "image_url",
                        "image_url": {"url": encoded_image}
                    }
                ]
            except Exception as e:
                raise Exception(f"Error processing image at {content_path}: {str(e)}")
        else:
            # Standard text prompt
            user_prompt = user_prompt.format(content=content)
            message_content = user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message_content}
        ]
        
        kwargs = {
            "model": self.model_name,
            "api_base": self.api_base,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs
        }
            
        # Add any additional model-specific kwargs
        kwargs.update(self.model_kwargs)
        
        try:
            response = await litellm.acompletion(**kwargs)
            output_text = response.choices[0].message.content
            
            return ModelResponse(
                raw_output=response,
                extracted_label=self._extract_label_from_output(output_text),
                logprobs=self._extract_logprobs(response)
            )
        except Exception as e:
            # Handle rate limits with exponential backoff
            if "rate_limit" in str(e).lower() or "429" in str(e):
                # This would be replaced with actual retry logic in production
                raise Exception(f"Rate limit exceeded: {str(e)}. Implement retry logic.")
            
            # Handle missing API keys with clear error messages
            if "api key" in str(e).lower() or "apikey" in str(e).lower():
                model_provider = self.model_name.split("/")[0].split("-")[0]
                env_var_name = f"{model_provider.upper()}_API_KEY"
                raise Exception(f"Missing API key for {model_provider}. "
                               f"Set the {env_var_name} environment variable.")
            
            # Re-raise other exceptions
            raise