#!/usr/bin/env python3
"""
Runner script for NAVIS SIH26168 Interactive Map Dashboard.
"""
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav.dashboard import start_dashboard

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NAVIS Interactive Map Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve dashboard on")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()
    start_dashboard(args.port, open_browser=not args.no_browser)
