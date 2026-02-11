"""Generate HTML detail pages for each LLM+Method combination."""
import json
from pathlib import Path

WEBSITE_DIR = Path(__file__).resolve().parent  # benchmark_website directory
DATA_FILE = WEBSITE_DIR / "data" / "leaderboard.json"
LEADERBOARD_DIR = WEBSITE_DIR / "leaderboard"


def get_score_class(score):
    """Determine CSS class for score badge based on value."""
    if score >= 0.5:
        return "high"
    elif score >= 0.3:
        return "medium"
    else:
        return "low"


def format_family_name(family):
    """Format family name for display."""
    return family.replace("_", " ").title()


def format_lead_name(lead):
    """Format lead time name for display."""
    if lead == "lead_time_0":
        return "Immediate (0)"
    elif lead == "lead_time_4":
        return "Fixed Delay (4)"
    elif lead == "lead_time_stochastic":
        return "Stochastic"
    return lead.replace("_", " ").title()


def generate_detail_page(method_data):
    """Generate HTML for a single detail page."""
    llm_id = method_data["llm_id"]
    method_id = method_data["method_id"]
    llm_label = method_data["llm_label"]
    method_label = method_data["method_label"]
    model_name = method_data.get("model_name", "")
    mean_ratio = method_data["mean_ratio"]
    by_family = method_data.get("by_family_mean_ratio", {})
    by_lead = method_data.get("by_lead_mean_ratio", {})
    
    score_class = get_score_class(mean_ratio)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{llm_label} · {method_label} · Leaderboard</title>
  <meta name="description" content="Performance details for {llm_label} with {method_label} method on the AI Agents for Inventory Control Benchmark.">
  <link rel="stylesheet" href="../../assets/css/main.css">
</head>
<body>
  <header class="site-header">
    <div class="wrapper header-inner">
      <div class="site-branding">
        <a class="site-title" href="../../index.html">AI Agents for Inventory Control</a>
        <p class="site-tagline">Benchmark for OR- and LLM-based inventory agents</p>
      </div>
      <nav class="site-nav">
        <a href="../../index.html">Home</a>
        <a href="../../dataset.html">Dataset</a>
        <a href="../../leaderboard.html">Leaderboard</a>
      </nav>
    </div>
  </header>

  <main class="page-content">
    <div class="wrapper">
      <article class="page">
        <div class="page-content">
          <div class="detail-header">
            <h1 class="detail-title">{llm_label} · {method_label}</h1>
            <p class="detail-subtitle">{model_name}</p>
            <div class="detail-score-section">
              <p style="margin: 0 0 0.5rem 0; color: #4b5563;">Average Normalized Reward (all benchmark instances)</p>
              <div class="score-badge {score_class}">{mean_ratio:.4f}</div>
            </div>
          </div>

          <div class="breakdown-section">
            <div class="breakdown-card">
              <h3>Performance by Dataset Family</h3>
              <table class="detail-table">
                <thead>
                  <tr>
                    <th>Family</th>
                    <th>Avg Normalized Reward</th>
                  </tr>
                </thead>
                <tbody>
"""
    
    # Add family breakdown rows
    if by_family:
        for family, score in sorted(by_family.items()):
            fam_class = get_score_class(score)
            html += f"""                  <tr>
                    <td>{format_family_name(family)}</td>
                    <td><span class="detail-score-value score-badge {fam_class}" style="display: inline-block; padding: 0.2rem 0.5rem; font-size: 0.9rem;">{score:.4f}</span></td>
                  </tr>
"""
    else:
        html += """                  <tr>
                    <td colspan="2" style="text-align: center; color: #9ca3af;">No data available</td>
                  </tr>
"""
    
    html += """                </tbody>
              </table>
            </div>

            <div class="breakdown-card">
              <h3>Performance by Lead-Time Setting</h3>
              <table class="detail-table">
                <thead>
                  <tr>
                    <th>Lead Time</th>
                    <th>Avg Normalized Reward</th>
                  </tr>
                </thead>
                <tbody>
"""
    
    # Add lead time breakdown rows
    if by_lead:
        # Sort lead times: lead_time_0, lead_time_4, lead_time_stochastic
        lead_order = ["lead_time_0", "lead_time_4", "lead_time_stochastic"]
        sorted_leads = sorted(by_lead.items(), key=lambda x: lead_order.index(x[0]) if x[0] in lead_order else 999)
        
        for lead, score in sorted_leads:
            lead_class = get_score_class(score)
            html += f"""                  <tr>
                    <td>{format_lead_name(lead)}</td>
                    <td><span class="detail-score-value score-badge {lead_class}" style="display: inline-block; padding: 0.2rem 0.5rem; font-size: 0.9rem;">{score:.4f}</span></td>
                  </tr>
"""
    else:
        html += """                  <tr>
                    <td colspan="2" style="text-align: center; color: #9ca3af;">No data available</td>
                  </tr>
"""
    
    html += """                </tbody>
              </table>
            </div>
          </div>

          <a href="../../leaderboard.html" class="back-link">Back to Leaderboard</a>
        </div>
      </article>
    </div>
  </main>

  <footer class="site-footer">
    <div class="wrapper footer-inner">
      <p class="footer-title">AI Agents for Inventory Control Benchmark</p>
      <p class="footer-description">A 1,320-instance benchmark for OR and LLM-based inventory agents on synthetic and real demand trajectories.</p>
      <p class="footer-links">
        <a href="https://github.com/BrunoFu/OR_Agent" target="_blank" rel="noopener">Code &amp; Data on GitHub</a>
      </p>
    </div>
  </footer>
</body>
</html>
"""
    
    return html


def main():
    """Generate all detail pages."""
    # Load data
    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Create directories and generate pages
    for method_data in data["methods"]:
        llm_id = method_data["llm_id"]
        method_id = method_data["method_id"]
        
        # Create directory if needed
        detail_dir = LEADERBOARD_DIR / llm_id
        detail_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML
        html_content = generate_detail_page(method_data)
        
        # Write file
        output_file = detail_dir / f"{method_id}.html"
        output_file.write_text(html_content, encoding="utf-8")
        print(f"Generated: {output_file}")
    
    print(f"\nGenerated {len(data['methods'])} detail pages.")


if __name__ == "__main__":
    main()
