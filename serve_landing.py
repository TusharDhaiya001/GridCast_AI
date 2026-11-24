#!/usr/bin/env python3
import http.server
import socketserver
import os
import urllib.parse
import json
import sys

# Import RAG engine directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from rag_engine import GridCastRAG
    rag_engine_instance = GridCastRAG()
except Exception as e:
    rag_engine_instance = None
    print(f"Warning: could not import rag_engine: {e}")

PORT = 5500
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landing")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def guess_type(self, path):
        if str(path).endswith(".ipynb"):
            return "application/x-ipynb+json"
        return super().guess_type(path)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # API endpoint for RAG queries
        if path == "/api/rag":
            query_params = urllib.parse.parse_qs(parsed.query)
            q = query_params.get("q", [""])[0]
            top_k = int(query_params.get("k", [3])[0])

            if rag_engine_instance and q:
                res = rag_engine_instance.generate_response(q, top_k=top_k)
            elif rag_engine_instance:
                res = {
                    "query": "",
                    "synthesized_answer": "Please provide a query parameter (?q=...).",
                    "citations": [],
                    "retrieved_chunks": []
                }
            else:
                res = {"error": "RAG engine not initialized"}

            payload = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        # Handle notebook routing for browser vs download
        if path.endswith("Electricity_Demand_and_Price_forecasting_India.ipynb"):
            query = urllib.parse.parse_qs(parsed.query)
            accept = self.headers.get("Accept", "")
            # If user explicitly requested download or raw, or not an HTML browser request
            if "download" in query or "raw" in query or "text/html" not in accept:
                return super().do_GET()
            else:
                # Redirect browser visit to rich interactive viewer
                self.send_response(302)
                self.send_header("Location", "/notebook-viewer.html")
                self.end_headers()
                return
        return super().do_GET()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Serving GridCast landing page at http://localhost:{PORT}")
        httpd.serve_forever()

