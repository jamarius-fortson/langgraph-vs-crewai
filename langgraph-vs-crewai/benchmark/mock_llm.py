import time
from typing import Any, List, Dict, Optional, Union
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

class MockChatOpenAI(BaseChatModel):
    """A mock LangChain-compatible LLM for benchmark development."""
    
    model_name: str = "gpt-4o-mock"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        time.sleep(0.1)
        input_str = str(messages)
        content = "2" if "1" in input_str and "+" in input_str else "I processed your request."
        
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def bind_tools(self, tools: List[Any], **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-chat-openai"
