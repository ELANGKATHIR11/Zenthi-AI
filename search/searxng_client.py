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
                
        # Fallback 2: DuckDuckGo HTML search (from odysseus-dev)
        print("SearXNG failed. Attempting DuckDuckGo HTML search fallback...")
        ddg_results = self._duckduckgo_fallback(query, top_k=top_k)
        if ddg_results:
            return ddg_results
                
        # Fallback 3: Mock results for development / demonstration mode
        print("[INFO] Using mock search results (Offline fallback).")
        return [
            {
                "title": f"Information on '{query}'",
                "url": "https://en.wikipedia.org/wiki/Special:Search",
                "snippet": f"This is an offline mock snippet for the query: {query}. To enable real web search, please run a local SearXNG container at http://localhost:8080."
            }
        ]

    def _duckduckgo_fallback(self, query, top_k=3):
        try:
            import httpx
            from bs4 import BeautifulSoup
            response = httpx.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "kp": "1"},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
                timeout=10,
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                parsed = []
                for result in soup.select(".result")[:top_k]:
                    link = result.select_one(".result__a")
                    if not link:
                        continue
                    url = link.get("href", "")
                    if "uddg=" in url:
                        from urllib.parse import urlparse, parse_qs
                        try:
                            parsed_url = urlparse(url)
                            qs = parse_qs(parsed_url.query)
                            if "uddg" in qs:
                                url = qs["uddg"][0]
                        except:
                            pass
                    snippet_el = result.select_one(".result__snippet")
                    parsed.append({
                        "title": link.get_text(" ", strip=True),
                        "url": url,
                        "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
                    })
                return parsed
        except Exception as e:
            print(f"[WARNING] DuckDuckGo HTML fallback search failed: {e}")
        return []

    def scrape_url_content(self, url, timeout=10):
        """Scrape webpage content using BeautifulSoup (inspired by Odysseus content fetcher)"""
        try:
            import httpx
            from bs4 import BeautifulSoup
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text(separator="\n")
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = "\n".join(chunk for chunk in chunks if chunk)
                return {
                    "title": soup.title.string.strip() if soup.title else "No Title",
                    "content": clean_text[:5000],
                    "success": True
                }
        except Exception as e:
            return {"error": str(e), "success": False}
        return {"error": "Unknown error", "success": False}

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
    import sys
    # Avoid unicode printing crash on Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    client = SearXNGClient()
    res = client.search("Python 3.13 features", top_k=2)
    print("\nFormatted Search Context:")
    print(client.format_context(res))
