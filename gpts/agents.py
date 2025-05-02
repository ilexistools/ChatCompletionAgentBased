import os, json, logging
from typing import Any, Dict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import tiktoken

logger = logging.getLogger(__name__)

class GPTAgent:
    def __init__(
        self,
        *,
        max_tokens: int = 1000,
        model: Optional[str] = None,
        temperature: float = 0.2,
        json_format: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        """Initialize GPTAgent, load env vars, set defaults, and configure client."""
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.gpt_model = model
        self.temperature = temperature
        self.json_format = json_format

        # agent context
        self.role: str      = kwargs.get("role", "")
        self.goal: str      = kwargs.get("goal", "")
        self.backstory: str = kwargs.get("backstory", "")
        self.knowledge: str = kwargs.get("knowledge", "")
        self.messages: List[Dict[str, str]] = []
        
        self._load_env()
        self._init_client()

    def _load_env(self) -> None:
        """Load OpenAI credentials from .env."""
        load_dotenv(find_dotenv())
        self.api_key      = os.getenv("OPENAI_API_KEY")
        self.api_url_base = os.getenv("OPENAI_API_BASE")
        self.api_model    = os.getenv("OPENAI_MODEL_NAME")
        if not all([self.api_key, self.api_url_base, self.api_model]):
            raise EnvironmentError("OPENAI_API_KEY, API_BASE and MODEL must be set.")

        if self.gpt_model is None:
            self.gpt_model = self.api_model

    def _init_client(self) -> None:
        """Instantiate OpenAI client."""
        self.client = OpenAI(base_url=self.api_url_base, api_key=self.api_key)

    def _fill_placeholders(self, template: str, inputs: Dict[str, Any]) -> str:
        for k, v in inputs.items():
            template = template.replace(f"{{{k}}}", str(v))
        return template

    def add_message(self, role: str, content: str) -> None:
        """Append a new message to the conversation buffer."""
        self.messages.append({"role": role, "content": content})

    def clear_messages(self) -> None:
        """Reset the message history."""
        self.messages.clear()

    def _calculate_tokens(self, text: str) -> int:
        """Return approximate token count, or 0 on failure."""
        try:
            enc = tiktoken.encoding_for_model(self.gpt_model)
            return len(enc.encode(text))
        except Exception:
            if self.verbose:
                logger.debug("Token calculation failed", exc_info=True)
            return 0
    
    def get_description(self) -> str:
        """Return a description of the agent."""
        description = f"Role: {self.role}\nGoal: {self.goal}\nBackstory: {self.backstory}\nKnowledge: {self.knowledge}"
        return description

    def run(self, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Build messages, call the OpenAI API, and return the response."""
        # update dynamic params
        self.verbose     = kwargs.get("verbose", self.verbose)
        self.max_tokens  = kwargs.get("max_tokens", self.max_tokens)
        self.gpt_model   = kwargs.get("model", self.gpt_model)
        self.temperature = kwargs.get("temperature", self.temperature)

        # fill templates
        role      = self._fill_placeholders(self.role, inputs or {})
        goal      = self._fill_placeholders(self.goal, inputs or {})
        backstory = self._fill_placeholders(self.backstory, inputs or {})
        knowledge = self._fill_placeholders(self.knowledge, inputs or {})

        # determine system role
        sys_role = "assistant" if self.gpt_model in ["o1","o1-mini","o3-mini"] else "system"

        # assemble messages
        self.clear_messages()
        for part in ("role", "goal", "knowledge", "json_format"):
            text = locals().get(part)
            if text:
                self.add_message(sys_role, f"{part.capitalize()}: {text}")
        self.add_message("user", backstory)

        if self.verbose:
            logger.debug("Messages to send: %s", self.messages)

        # API call w/ simple retry
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=self.messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=kwargs.get("stream", False),
                    response_format={"type":"json_object"} if self.json_format else None
                )
                content = resp.choices[0].message.content
                return json.loads(content) if self.json_format else content
            except Exception as e:
                logger.error("API error (attempt %d): %s", attempt+1, e)
        raise RuntimeError("Failed to get a response after 3 retries.")
