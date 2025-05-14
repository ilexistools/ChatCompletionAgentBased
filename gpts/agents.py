import os, logging
from typing import Any, Dict, Optional, List, Type
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import tiktoken
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class GPTAgent:
    def __init__(
        self,
        *,
        max_tokens: int = 1000,
        model: Optional[str] = None,
        temperature: float = 0.2,
        output_schema: Optional[Type[BaseModel]] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output_schema = output_schema

        self.role: str = kwargs.get("role", "")
        self.goal: str = kwargs.get("goal", "")
        self.backstory: str = kwargs.get("backstory", "")
        self.knowledge: str = kwargs.get("knowledge", "")
        self.messages: List[Dict[str, str]] = []

        self._load_env()
        self.gpt_model = model or self.api_model
        self._init_client()

    def _load_env(self) -> None:
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url_base = os.getenv("OPENAI_API_BASE")
        self.api_model = os.getenv("OPENAI_MODEL_NAME")
        if not all([self.api_key, self.api_url_base, self.api_model]):
            raise EnvironmentError("OPENAI_API_KEY, OPENAI_API_BASE, and OPENAI_MODEL_NAME must be set.")

    def _init_client(self) -> None:
        self.client = OpenAI(base_url=self.api_url_base, api_key=self.api_key)

    def _fill_placeholders(self, template: str, inputs: Dict[str, Any]) -> str:
        for k, v in inputs.items():
            template = template.replace(f"{{{k}}}", str(v))
        return template

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def clear_messages(self) -> None:
        self.messages.clear()

    def _calculate_tokens(self, text: str) -> int:
        try:
            enc = tiktoken.encoding_for_model(self.gpt_model)
            return len(enc.encode(text))
        except Exception:
            if self.verbose:
                logger.debug("Token calculation failed", exc_info=True)
            return 0

    def get_description(self) -> str:
        return f"Role: {self.role}\nGoal: {self.goal}\nBackstory: {self.backstory}\nKnowledge: {self.knowledge}"

    def _build_tool(self):
        if not self.output_schema:
            return None

        schema = self.output_schema.schema()
        return [{
            "type": "function",
            "function": {
                "name": "structured_output",
                "description": "Structured output based on user intent.",
                "parameters": schema
            }
        }]

    def run(self, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        self.verbose = kwargs.get("verbose", self.verbose)
        self.max_tokens = kwargs.get("max_tokens", self.max_tokens)
        self.gpt_model = kwargs.get("model", self.gpt_model)
        self.temperature = kwargs.get("temperature", self.temperature)
        self.output_schema = kwargs.get("output_schema", self.output_schema)

        role = self._fill_placeholders(self.role, inputs or {})
        goal = self._fill_placeholders(self.goal, inputs or {})
        backstory = self._fill_placeholders(self.backstory, inputs or {})
        knowledge = self._fill_placeholders(self.knowledge, inputs or {})

        sys_role = "assistant" if self.gpt_model in ["o1", "o1-mini", "o3-mini"] else "system"

        system_parts = []
        for part in ("role", "goal", "knowledge"):
            text = {"role": role, "goal": goal, "knowledge": knowledge}.get(part)
            if text:
                system_parts.append(text)

        self.clear_messages()
        sys_content = "\n".join(part.strip() for part in system_parts if part)
        if sys_content:
            self.add_message(sys_role, sys_content)

        self.add_message("user", backstory)

        if self.verbose:
            print("\nPrompt:\n")
            for message in self.messages:
                print(f"{message['content']}")

        tools = self._build_tool()
        tool_choice = {"type": "function", "function": {"name": "structured_output"}} if tools else None

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=self.messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=kwargs.get("stream", False),
                    tools=tools,
                    tool_choice=tool_choice
                )

                if tools:
                    arguments = response.choices[0].message.tool_calls[0].function.arguments
                    if self.verbose:
                        print("\nStructured response (JSON):\n", arguments)
                    return self.output_schema.parse_raw(arguments)

                content = getattr(response.choices[0].message, "content", "")
                if not content:
                    raise RuntimeError("Empty response from API.")
                if self.verbose:
                    print("\nResponse:\n", content)
                return content

            except Exception as e:
                logger.error("API error (attempt %d): %s", attempt + 1, e)

        raise RuntimeError("Failed to get a response after 3 retries.")
