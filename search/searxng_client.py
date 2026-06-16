import os
import requests

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")

class SearXNGClient:
    def __init__(self, base_url=SEARXNG_URL):
        self.base_url = base_url

    def search(self, query, top_k=3):
        print(f"Querying SearXNG: {query}")
        
        # We will attempt to connect to the local SearXNG instance
        try:
            url = f"{self.base_url}/search"
            params = {
                "q": query,
                "format": "json",
                "engines": "google,bing,duckduckgo",
                "language": "en"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                parsed_results = []
                for r in results[:top_k]:
                    parsed_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "") or r.get("snippet", "")
                    })
                return parsed_results
        except Exception as e:
            print(f"[WARNING] SearXNG connection to {self.base_url} failed: {e}")
            
        # Fallback 1: Try public instance
        public_instances = [
            "https://searx.be",
            "https://searxng.site",
            "https://search.privacyredirect.com"
        ]
        
        for instance in public_instances:
            try:
                print(f"Attempting public SearXNG instance: {instance}")
                url = f"{instance}/search"
                params = {
                    "q": query,
                    "format": "json",
                    "language": "en"
                }
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    parsed_results = []
                    for r in results[:top_k]:
                        parsed_results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("content", "") or r.get("snippet", "")
                        })
                    return parsed_results
            except:
                continue
                
        # Fallback 2: Mock results for development / demonstration mode
        print("[INFO] Using mock search results (Offline fallback).")
        return [
            {
                "title": f"Information on '{query}'",
                "url": "https://en.wikipedia.org/wiki/Special:Search",
                "snippet": f"This is an offline mock snippet for the query: {query}. To enable real web search, please run a local SearXNG container at http://localhost:8080."
            }
        ]

    def format_context(self, search_results):
        if not search_results:
            return "No web search results found."
            
        context_lines = []
        for i, res in enumerate(search_results, 1):
            context_lines.append(
                f"[{i}] Title: {res['title']}\n"
                f"    URL: {res['url']}\n"
                f"    Snippet: {res['snippet']}\n"
            )
        return "\n".join(context_lines)

if __name__ == "__main__":
    client = SearXNGClient()
    res = client.search("Python 3.13 features", top_k=2)
    print("\nFormatted Search Context:")
    print(client.format_context(res))
