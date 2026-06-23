"""
Research Assistant CLI — uses Claude Haiku with three tools:
  1. web_search        — live web results via Tavily
  2. arxiv_search       — academic papers via the arxiv package
  3. wikipedia_lookup   — established encyclopedic background via the wikipedia package
"""

import json
import os
import arxiv
import anthropic
import requests
import wikipedia
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Wikipedia's API rejects requests with no/blank User-Agent, returning an HTML
# error page instead of JSON (which crashes the wikipedia package's JSON parser).
wikipedia.set_user_agent("research-assistant/1.0 (contact: example@example.com)")

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Tool definitions (sent to Claude so it knows what it can call)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the live web for recent news, documentation, tutorials, "
            "or any topic that benefits from up-to-date information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "arxiv_search",
        "description": (
            "Search arXiv for academic papers on scientific or technical topics. "
            "Returns titles, authors, abstracts, and links."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrases to search for on arXiv.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "wikipedia_lookup",
        "description": (
            "Look up established encyclopedic background on a person, place, or "
            "concept using Wikipedia. Use this for well-known, settled facts (e.g. "
            "biography, history, definitions) — NOT for breaking news or recent "
            "events (use web_search instead) and NOT for academic/scientific "
            "papers (use arxiv_search instead)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic, person, place, or concept to look up on Wikipedia.",
                }
            },
            "required": ["query"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def run_web_search(query: str) -> str:
    response = tavily.search(query=query, max_results=5)
    results = response.get("results", [])
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"**{r['title']}**\n{r['url']}\n{r.get('content', '')}\n")
    return "\n---\n".join(lines)


def run_arxiv_search(query: str, max_results: int = 5) -> str:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = list(client.results(search))
    if not papers:
        return "No papers found."
    lines = []
    for p in papers:
        authors = ", ".join(a.name for a in p.authors[:3])
        if len(p.authors) > 3:
            authors += " et al."
        lines.append(
            f"**{p.title}**\n"
            f"Authors: {authors}\n"
            f"Published: {p.published.date()}\n"
            f"URL: {p.entry_id}\n"
            f"Abstract: {p.summary[:300]}...\n"
        )
    return "\n---\n".join(lines)


def run_wikipedia_lookup(query: str) -> str:
    try:
        page = wikipedia.page(query, auto_suggest=False)
        summary = wikipedia.summary(query, sentences=5)
        return f"**{page.title}**\n{page.url}\n\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        options = ", ".join(e.options[:10])
        return (
            f"'{query}' is ambiguous on Wikipedia. Possible matches: {options}. "
            "Try again with a more specific query."
        )
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."
    except (requests.exceptions.RequestException, requests.exceptions.JSONDecodeError):
        return (
            f"Wikipedia lookup for '{query}' failed due to a network or API error. "
            "Try again."
        )


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "web_search":
        return run_web_search(inputs["query"])
    if name == "arxiv_search":
        return run_arxiv_search(
            inputs["query"], inputs.get("max_results", 5)
        )
    if name == "wikipedia_lookup":
        return run_wikipedia_lookup(inputs["query"])
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_turn(messages: list) -> str:
    """Send messages to Claude, execute any tool calls, and return the final text."""
    while True:
        response = claude.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant message to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Extract the final text reply
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Execute each tool call and collect results
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\n[tool: {block.name}({json.dumps(block.input)})]")
            result = dispatch_tool(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        # Feed tool results back for the next iteration
        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

def main():
    print("Research Assistant  (type 'quit' or 'exit' to stop)\n")
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        reply = run_turn(messages)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
