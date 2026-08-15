# Ecommerce Customer Service Agent MVP

A minimal Python-based ecommerce customer service agent that connects to an OpenAI-compatible API endpoint. The agent is designed for an MVP support workflow covering returns and exchanges, shipping and logistics, and product specification questions.

## Features

- OpenAI SDK integration using the standard Python client.
- Environment-based configuration for API key, base URL, and model.
- OpenAI-compatible Function Calling tools for mock order and logistics lookup.
- Lightweight product knowledge retrieval from `products.json` for specs, sizing, materials, FAQs, and after-sales policies.
- Guardrail-based human support ticket fallback for sensitive disputes and unresolved cases.
- Exponential backoff retries for transient API connection, timeout, rate-limit,
  and server errors.
- Focused system prompt for ecommerce support scenarios.
- Continuous command-line chat loop for quick local testing.
- Safe fallback behavior when customer order, tracking, or product details are missing.

## Architecture

```text
Customer CLI Input
        |
        v
Agent Loop + Conversation Memory + System Guardrails
        |
        v
OpenAI-compatible Chat Completion API
        |
        +---------------------------+
        |                           |
        v                           v
Direct Agent Reply             Function Calling
                                    |
                 +------------------+------------------+
                 |                  |                 |
                 v                  v                 v
          Order Status       Logistics Lookup    Product Knowledge
          query_order_status query_logistics     search_product_knowledge
                                                       |
                                                       v
                                                products.json (local RAG)
                 |                  |                 |
                 +------------------+-----------------+
                                    |
                                    v
                         Tool JSON Results -> Model -> Customer Reply

Sensitive dispute / unresolved complaint / out-of-scope request
                                    |
                                    v
                         create_human_ticket
                                    |
                                    v
                         Ticket JSON -> Model -> Human Support ID
```

The three core modules are:

- **Tools call chain**: the model selects a function, the agent executes it locally, then sends the standard `role=tool` JSON result back to the model for the final reply.
- **Local RAG knowledge base**: `search_product_knowledge` retrieves product specifications, sizing, materials, FAQs, warranty, and after-sales rules from `products.json`; the model must use this result for product-specific facts.
- **Human ticket fallback**: `create_human_ticket` is the Guardrail for fraud complaints, legal threats, refund disputes, explicit human-agent requests, safety incidents, and cases the knowledge base cannot resolve.

The project intentionally keeps the architecture small and easy to extend:

- `agent.py` contains the command-line interface, OpenAI client setup, system prompt, and message loop.
- `tools.py` contains the mock business functions and their JSON schemas for Function Calling.
- `products.json` is the lightweight external product knowledge base used by `search_product_knowledge`.
- `requirements.txt` defines the minimal runtime dependencies.
- `.env` can be used locally for secrets and endpoint configuration.

## Requirements

- Python 3.10 or later
- An OpenAI-compatible API key

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
# Optional retry settings for transient API failures.
OPENAI_MAX_RETRIES=3
OPENAI_RETRY_DELAY_SECONDS=1
```

`OPENAI_BASE_URL` is optional. Leave it unset when using the default OpenAI endpoint.
For a relay or other OpenAI-compatible service, set it to that service's `/v1`
endpoint and keep `OPENAI_API_KEY` in the same `.env` file.

## Usage

Run the agent:

```bash
python agent.py
```

Example interaction:

```text
Customer: I want to exchange a jacket for a larger size.
Agent: I can help with that. Please share your order number, the email used for the purchase, and the size you would like instead.
```

Type `exit` or `quit` to end the session.

When a customer provides a supported mock order number such as `ORD-1001` or a
tracking number such as `SF1234567890`, the agent calls the corresponding tool
and uses its JSON result in the reply.

For product questions such as the material or sizing of the `Retro Runner`, the
agent first searches `products.json` and answers from the matched knowledge
result instead of relying on memory.

## Extension Ideas

- Connect the agent to a real order management system.
- Add structured tool calls for return authorization, tracking lookup, and inventory checks.
- Persist conversation history to a database.
- Add automated tests for prompt behavior and error handling.

## License

MIT
