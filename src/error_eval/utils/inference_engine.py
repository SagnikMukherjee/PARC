import openai
import os
import json
import time
import concurrent.futures
import google.generativeai as genai

from dotenv import load_dotenv
from typing import List, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from loguru import logger

load_dotenv()

# Cost tracking variables
total_prompt_tokens = 0
total_completion_tokens = 0
total_cost = 0

class CostTracker:
    def __init__(self):
        # Cost per 1K tokens for different models
        self.cost_rates = {
            "o1-mini": {
                "prompt": 0.0033,     
                "completion": 0.0132    
            },
            "o1-preview": {
                "prompt": 0.0165,     
                "completion": 0.066    
            },
            "gpt4o": {
                "prompt": 0.0050,     
                "completion": 0.0150    
            }
            # Add other models' rates as needed
        }
        
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> dict:
        if model not in self.cost_rates:
            logger.warning(f"No cost rates defined for model {model}")
            return {
                "prompt_cost": 0,
                "completion_cost": 0,
                "total_cost": 0
            }
            
        rates = self.cost_rates[model]
        prompt_cost = (prompt_tokens / 1000) * rates["prompt"]
        completion_cost = (completion_tokens / 1000) * rates["completion"]
        total_cost = prompt_cost + completion_cost
        
        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost
        }

class Hyperparameters:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 1)

class InferenceEngine:
    def __init__(
        self, inference_strategy: str, connection_details: dict, model_name: str = None
    ):
        print('here')
        self.model_name = model_name
        self.api_kind = inference_strategy
        self.cost_tracker = CostTracker()
        if inference_strategy == "openai":
            self._initialize_openai_client(connection_details)
        elif inference_strategy == "azure_openai":
            self._initialize_azure_openai_client(connection_details)
        elif inference_strategy == "gemini":
            self._initialize_gemini_client(connection_details, model_name)
        else:
            raise ValueError(f"Invalid inference strategy: {inference_strategy}")

    def _initialize_openai_client(self, connection_details: dict):
        if "api_key" not in connection_details:
            connection_details["api_key"] = os.getenv("OPENAI_API_KEY")

        if "base_url" not in connection_details:
            connection_details["base_url"] = connection_details.get(
                "base_url", "https://api.openai.com/v1"
            )

        self.client = openai.OpenAI(
            api_key=connection_details["api_key"],
            base_url=connection_details["base_url"],
        )

    def _initialize_azure_openai_client(self, connection_details: dict):
        if "api_key" not in connection_details:
            connection_details["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")

        if "base_url" not in connection_details:
            connection_details["base_url"] = os.getenv("AZURE_OPENAI_BASE_URL")

        if "api_version" not in connection_details:
            connection_details["api_version"] = "2024-02-01"

        self.client = openai.AzureOpenAI(
            api_key=connection_details["api_key"],
            azure_endpoint=connection_details["base_url"],
            api_version=connection_details["api_version"],
        )

    def _initialize_gemini_client(self, connection_details: dict, model_name: str):
        if "api_key" not in connection_details:
            connection_details["api_key"] = os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=connection_details["api_key"])
        self.client = genai.GenerativeModel(model_name)

    def _log_usage(self, response, model_name: str):
        global total_prompt_tokens, total_completion_tokens, total_cost
        
        try:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Calculate costs
            costs = self.cost_tracker.calculate_cost(model_name, prompt_tokens, completion_tokens)
            
            # Update global counters
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_cost += costs["total_cost"]
            
            # Log the usage and costs
            logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            logger.info(f"Cost - Prompt: ${costs['prompt_cost']:.4f}, Completion: ${costs['completion_cost']:.4f}, Total: ${costs['total_cost']:.4f}")
            
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "costs": costs
            }
            
        except Exception as e:
            logger.error(f"Error logging usage: {str(e)}")
            return None

    def _openai_single_message_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> str:
        logger.debug(f"Running inference for {self.model_name}")
        
        if self.model_name == "o1-mini" or self.model_name == "o1-preview":
            # For O1 models, combine system and user messages into a single user message
            if len(messages) > 1:
                messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                # Log usage and costs
                usage_stats = self._log_usage(response, self.model_name)
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}. Model: {self.model_name}")
                return str(response)
        else:
            # Standard behavior for other models
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **hyperparameters.__dict__
            )
            # Log usage and costs
            usage_stats = self._log_usage(response, self.model_name)
            return response.choices[0].message.content

    def _gemini_single_message_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> str:
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        combined_prompt = f"<system>{system_prompt}</system><user>{user_prompt}</user>"
        response = self.client.generate_content(combined_prompt)
        return response.text

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(2),
    )
    def _retry_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> str:
        if self.api_kind == "openai" or self.api_kind == "azure_openai":
            return self._openai_single_message_inference(messages, hyperparameters)
        elif self.api_kind == "gemini":
            return self._gemini_single_message_inference(messages, hyperparameters)
        else:
            raise ValueError(f"Invalid inference strategy: {self.api_kind}")

    def single_message_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> str:
        try:
            return self._retry_inference(messages, hyperparameters)
        except RetryError:
            logger.error("All retry attempts failed")
            return ""

    def _openai_parallel_messages_inference(
        self,
        messages: List[List[Dict[str, str]]],
        hyperparameters: Hyperparameters,
        max_workers: int = 32,
    ) -> List[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.single_message_inference, msgs, hyperparameters)
                for msgs in messages
            ]
            responses = []
            for future in futures:
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Inference task failed: {e}")
                    responses.append(None)
        return responses

    def _gemini_parallel_messages_inference(
        self,
        messages: List[List[Dict[str, str]]],
        hyperparameters: Hyperparameters,
        max_workers: int = 32,
    ) -> List[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.single_message_inference, msgs, hyperparameters)
                for msgs in messages
            ]
            responses = []
            for future in futures:
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Inference task failed: {e}")
                    responses.append(None)
        return responses

    def parallel_messages_inference(
        self, messages: List[List[Dict[str, str]]], hyperparameters: Hyperparameters
    ) -> List[str]:
        if self.api_kind == "openai" or self.api_kind == "azure_openai":
            return self._openai_parallel_messages_inference(messages, hyperparameters)
        elif self.api_kind == "gemini":
            return self._gemini_parallel_messages_inference(messages, hyperparameters)
        else:
            raise ValueError(f"Invalid inference strategy: {self.api_kind}")

    def _create_batch_file(self, requests: List[Dict]) -> str:
        file_name = "batch_requests.jsonl"
        with open(file_name, "w") as file:
            for request in requests:
                json.dump(request, file)
                file.write("\n")
        return file_name

    def _upload_batch_file(self, file_name: str):
        with open(file_name, "rb") as file:
            response = self.client.files.create(file=file, purpose="batch")
        return response.id

    def create_batch(
        self, requests: List[Dict], completion_window: str = "24h"
    ) -> Dict:
        file_name = self._create_batch_file(requests)
        file_id = self._upload_batch_file(file_name)

        batch_job = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )
        return batch_job

    def check_batch_status(self, batch_id: str) -> Dict:
        return self.client.batches.retrieve(batch_id)

    def get_batch_results(self, batch_job: Dict) -> List[Dict]:
        while batch_job["status"] != "completed":
            time.sleep(60)  # Wait for 1 minute before checking again
            batch_job = self.check_batch_status(batch_job["id"])

        result_file_id = batch_job["output_file_id"]
        result = self.client.files.content(result_file_id).content

        results = []
        for line in result.decode().split("\n"):
            if line:
                results.append(json.loads(line))
        return results

    def _openai_batch_inference(
        self, messages: List[List[Dict[str, str]]], hyperparameters: Hyperparameters
    ) -> List[str]:
        requests = []
        for i, message_list in enumerate(messages):
            request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": message_list,
                    **hyperparameters.__dict__,
                },
            }
            requests.append(request)

        batch_job = self.create_batch(requests)
        results = self.get_batch_results(batch_job)

        return [
            result["response"]["body"]["choices"][0]["message"]["content"]
            for result in results
        ]

    def batch_inference(
        self, messages: List[List[Dict[str, str]]], hyperparameters: Hyperparameters
    ) -> List[str]:
        if self.api_kind == "openai" or self.api_kind == "azure_openai":
            return self._openai_batch_inference(messages, hyperparameters)
        else:
            raise ValueError(f"Invalid inference strategy: {self.api_kind}")

    def __str__(self):
        return f"InferenceEngine(api_kind={self.api_kind}, model_name={self.model_name})"


def get_total_usage():
    """Return the current total usage statistics"""
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost": total_cost
    }


if __name__ == "__main__":
    inference_engine = InferenceEngine(
        inference_strategy="openai", connection_details={}, model_name="gpt-4"
    )
    print(inference_engine)