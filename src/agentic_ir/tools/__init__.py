"""Tool registry exposed to the Retrieval agent.

Tools are plain callables with JSON-schema descriptions, so the same
registry serves both LLM tool-calling and the deterministic heuristic
shortcut path.
"""
