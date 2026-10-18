"""Bounded LangChain chat model configuration."""

from langchain_openai import ChatOpenAI

from enterprise_rag_chatbot.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise ValueError("Set OPENAI_API_KEY in .env to generate answers")
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        timeout=45,
        max_retries=2,
        max_tokens=settings.chat_max_tokens,
    )
