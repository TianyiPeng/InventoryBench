# InventoryBench Website

This directory contains the static HTML website for InventoryBench.

## Structure

```
leaderboard_website/
├── index.html              # Homepage
├── leaderboard.html        # Full leaderboard page
├── dataset.html            # Dataset documentation
├── build_stats.py          # Script to generate leaderboard data
├── generate_detail_pages.py # Script to generate detail pages
├── serve.py                # Local development server
├── assets/
│   └── css/
│       └── main.css       # Stylesheet
├── data/
│   └── leaderboard.json   # Leaderboard data (generated)
└── leaderboard/
    ├── gemini-3-flash/
    ├── gpt-5-mini/
    └── grok-4.1-fast/
        └── *.html          # Detail pages for each LLM+Method combination
```

## Usage

### 1. Generate Leaderboard Data

Run the build script to aggregate benchmark results from `results/`:

```bash
python leaderboard_website/build_stats.py
```

This will:
- Scan all `*_bench` directories in `results/`
- Aggregate performance metrics from `benchmark_results.json` files
- Generate `leaderboard_website/data/leaderboard.json`

### 2. Generate Detail Pages

After updating the leaderboard data, regenerate detail pages:

```bash
python leaderboard_website/generate_detail_pages.py
```

This creates HTML detail pages for each LLM+Method combination in the `leaderboard/` directory.

### 3. Serve the Website

```bash
cd leaderboard_website
python serve.py
```

Then open `http://127.0.0.1:8765/index.html` in your browser.

## Notes

- All paths in HTML files are relative, so the site works when served from this directory
- The website is pure HTML/CSS/JavaScript with no build step required
- Leaderboard data is embedded inline in the HTML pages
