import os
import json
import random
import sys
import time
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from tools import TOOL_DEFINITIONS, TOOL_HANDLERS


SYSTEM_PROMPT = """
You are an ecommerce customer service agent MVP.

Your job is to help customers with:
- Returns and exchanges: explain eligibility, required order information,
  item condition expectations, refund or replacement options, and next steps.
- Shipping and logistics: answer questions about tracking, delivery estimates,
  lost packages, delayed shipments, and address updates.
- Product specifications: clarify size, color, material, compatibility,
  warranty, bundle contents, and usage details.

Guidelines:
- Be concise, friendly, and professional.
- Ask for the order number, email, SKU, tracking number, or other missing
  details when they are needed.
- Do not invent order status, tracking events, stock levels, policies, or
  product facts. If information is unavailable, say so and explain what the
  customer should provide next.
- When the customer provides an order number or tracking number, use the
  matching tool before answering with status or logistics details.
- For any product specification, sizing, material, compatibility, care,
  warranty, FAQ, or product-specific return/exchange policy question, you must
  call search_product_knowledge first, even when the customer only provides a
  product name or a natural-language question.
- Do not answer product facts from memory. Use the knowledge tool result as the
  source of truth and clearly say when no matching product knowledge is found.
- Treat tool results as the source of truth. If a tool reports that no record
  was found, ask the customer to verify the identifier instead of guessing.
- Guardrail: for fraud/scam complaints, legal threats, explicit requests for a
  human agent, refund or payment disputes, safety incidents, or complaints that
  remain unresolved or are outside the knowledge base, call
  create_human_ticket. Do not attempt to debate, promise a resolution, or make
  up facts in these cases.
- When creating a ticket, use a concise factual summary and the matching issue
  type. If no authenticated user ID is available in this CLI, pass
  `anonymous`; never invent a personal identifier.
- After a successful ticket tool call, tell the customer the returned ticket ID
  and that the issue has been routed to human support.
- Keep the conversation focused on ecommerce support.
""".strip()

MAX_TOOL_ROUNDS = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


def build_client() -> OpenAI:
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
        )

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    # Retries are handled explicitly below so the CLI uses one backoff policy
    # for normal replies and tool-call follow-up requests.
    kwargs["max_retries"] = 0
    return OpenAI(**kwargs)


def _read_non_negative_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _read_non_negative_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after")
    try:
        return max(0.0, float(retry_after)) if retry_after is not None else None
    except (TypeError, ValueError):
        return None


def _create_completion_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    max_retries = _read_non_negative_int(
        "OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES
    )
    base_delay = _read_non_negative_float(
        "OPENAI_RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY_SECONDS
    )

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt >= max_retries:
                raise

            server_delay = _retry_after_seconds(exc)
            backoff = base_delay * (2**attempt)
            jitter = random.uniform(0, min(0.25, backoff / 4))
            delay = server_delay if server_delay is not None else backoff + jitter
            print(
                f"Temporary API error; retrying in {delay:.1f}s "
                f"({attempt + 1}/{max_retries})...",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError("API request failed without an exception")


def _assistant_message_payload(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant"}
    if message.content is not None:
        payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _tool_error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _execute_tool_call(tool_call: Any) -> str:
    function_name = tool_call.function.name
    handler = TOOL_HANDLERS.get(function_name)
    if handler is None:
        return _tool_error(f"未知工具: {function_name}")

    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return _tool_error(f"工具参数不是有效 JSON: {function_name}")

    if not isinstance(arguments, dict):
        return _tool_error(f"工具参数必须是 JSON 对象: {function_name}")

    try:
        return handler(**arguments)
    except TypeError as exc:
        return _tool_error(f"工具参数错误: {exc}")


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def ask_agent(client: OpenAI, model: str, messages: list[dict[str, Any]]) -> str:
    # Keep partial tool-call messages out of the live conversation if a later
    # network request fails; the caller can then safely remove the user input.
    working_messages = list(messages)

    for _ in range(MAX_TOOL_ROUNDS):
        response = _create_completion_with_retry(
            client,
            model=model,
            messages=working_messages,
            temperature=0.3,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []
        working_messages.append(_assistant_message_payload(assistant_message))

        if not tool_calls:
            messages[:] = working_messages
            content = assistant_message.content
            return content.strip() if content else ""

        for tool_call in tool_calls:
            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": _execute_tool_call(tool_call),
                }
            )

    raise RuntimeError("Tool call loop exceeded the maximum number of rounds")


def main() -> None:
    _configure_console_encoding()
    client = build_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("Ecommerce Customer Service Agent MVP")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            answer = ask_agent(client, model, messages)
        except Exception as exc:
            print(f"Agent error: {exc}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
