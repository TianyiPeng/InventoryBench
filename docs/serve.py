"""Serve the benchmark website locally. Run: python serve.py"""
import http.server
import socketserver
import os

PORT = 8765
DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)

with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print("Serving at: http://127.0.0.1:{}/".format(PORT))
    print("Open in browser: http://127.0.0.1:{}/index.html".format(PORT))
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()
