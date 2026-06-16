import re

class QueryRouter:
    def __init__(self):
        # Define keywords triggering specific retrieval modes
        self.rag_keywords = [
            r"document", r"pdf", r"file", r"upload", r"indexed", 
            r"my notebook", r"paper", r"in the text", r"context"
        ]
        self.search_keywords = [
            r"weather", r"news", r"today", r"latest", r"current", 
            r"now", r"recent", r"price of", r"stock", r"scores"
        ]

    def route(self, query):
        query_lower = query.lower()
        
        has_rag = any(re.search(kw, query_lower) for kw in self.rag_keywords)
        has_search = any(re.search(kw, query_lower) for kw in self.search_keywords)
        
        if has_rag and has_search:
            return "HYBRID" # RAG + Web Search
        elif has_rag:
            return "RAG"
        elif has_search:
            return "SEARCH"
        else:
            return "DIRECT"
