"""CAUSA NOSTRA: evidence-grounded reasoning over the supplied CAMO graph.

Layers (dependencies point downward only):

    cli
     └── pipeline      orchestration of Understand -> Retrieve -> Reason -> Verify -> Respond
          ├── answers  output contract (Pydantic), deterministic output checks, rubric
          ├── llm      LLM endpoint client (BRINE_URL / BRINE_MODEL / BRINE_API_KEY)
          ├── retrieval  readable evidence records and BM25 search (Haystack)
          └── camo     CAMO interpretation rules and graph invariants
               └── graph  Pydantic models, JSON loading, indexed lookups
"""

__version__ = "0.1.0"
