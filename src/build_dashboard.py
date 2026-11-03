"""Build a standalone interactive HTML dashboard (stand-in / preview for Power BI).

python -m src.build_dashboard   ->   dashboard/churn_dashboard.html
"""

import pandas as pd
import plotly.express as px

from src import config

OUT = config.PROJECT_ROOT / "dashboard" / "churn_dashboard.html"


def main() -> None:
    scores = pd.read_csv(config.DATA_PROCESSED / "customer_scores_for_bi.csv")
    mrs = pd.read_csv(config.DATA_PROCESSED / "monthly_revenue_by_segment.csv")
    wb = pd.read_csv(config.DATA_PROCESSED / "high_risk_churn_list.csv")
    summ = pd.read_csv(config.DATA_PROCESSED / "segment_summary.csv")

    total_rev = mrs["revenue"].sum()
    kpis = [
        ("Total revenue", f"{total_rev:,.0f}"),
        ("Customers", f"{len(scores):,}"),
        ("Segments", f"{scores['segment'].nunique()}"),
        ("High-risk win-back targets", f"{len(wb):,}"),
        ("Revenue at risk", f"{wb['revenue_at_risk'].sum():,.0f}"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{k}</div></div>' for k, v in kpis
    )

    f1 = px.area(mrs, x="month", y="revenue", color="segment", title="Monthly revenue by segment")
    f2 = px.pie(
        summ, names="segment", values="revenue", hole=0.45, title="Revenue share by segment"
    )
    f3 = px.box(
        scores,
        x="segment",
        y="churn_probability",
        color="segment",
        title="Churn probability by segment",
    )
    f4 = px.bar(summ, x="segment", y="customers", color="segment", title="Customers per segment")
    figs = "".join(
        f.update_layout(height=360, margin=dict(t=50, b=30))
        .to_html(full_html=False, include_plotlyjs=(i == 0), config={"displaylogo": False})
        .join(['<div class="card">', "</div>"])
        for i, f in enumerate([f1, f2, f3, f4])
    )

    cols = [
        "customer_id",
        "segment",
        "city",
        "recency_days",
        "frequency",
        "monetary",
        "churn_probability",
        "revenue_at_risk",
        "recommended_action",
    ]
    top = wb.sort_values("revenue_at_risk", ascending=False)[cols].round(2)
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in top.itertuples(index=False)
    )
    head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
    options = "".join(f"<option>{s}</option>" for s in sorted(top["segment"].unique()))
    playbook = "".join(
        f"<tr><td><b>{r.segment}</b></td><td>{r.customers:,}</td><td>{r.pct_revenue}%</td>"
        f"<td>{r.recommended_action}</td></tr>"
        for r in summ.itertuples()
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Retail Churn & Segmentation Dashboard</title><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#222}}
header{{background:#1f3a5f;color:#fff;padding:18px 28px}}main{{padding:20px 28px}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}}
.kpi{{background:#fff;padding:14px 22px;border-radius:8px;box-shadow:0 1px 3px #0002;min-width:150px}}
.kpi .v{{font-size:26px;font-weight:700;color:#1f3a5f}}.kpi .l{{font-size:12px;color:#667}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:14px}}
.card{{background:#fff;border-radius:8px;box-shadow:0 1px 3px #0002;padding:6px}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:#fff}}
th,td{{padding:6px 8px;border-bottom:1px solid #e3e6ea;text-align:left;vertical-align:top}}
th{{background:#eef1f5;position:sticky;top:0}}.scroll{{max-height:420px;overflow:auto;border-radius:8px}}
h2{{margin:26px 0 8px}}select{{padding:5px;margin-left:8px}}</style></head><body>
<header><h1 style="margin:0">Customer Segmentation & Churn Dashboard</h1>
<div style="opacity:.8">Axlero DA Internship - Project 1 | scored as of 2026-01-01</div></header><main>
<div class="kpis">{kpi_html}</div><div class="grid">{figs}</div>
<h2>Segment playbook</h2><div class="scroll"><table><tr><th>Segment</th><th>Customers</th><th>% revenue</th><th>Recommended action</th></tr>{playbook}</table></div>
<h2>High-Risk Churn list <span style="font-size:13px;font-weight:400">filter segment:</span>
<select id="f" onchange="flt()"><option value="">All</option>{options}</select></h2>
<div class="scroll"><table id="t"><tr>{head}</tr>{rows}</table></div></main>
<script>function flt(){{var v=document.getElementById('f').value,r=document.querySelectorAll('#t tr');
for(var i=1;i<r.length;i++)r[i].style.display=(!v||r[i].cells[1].innerText===v)?'':'none';}}</script>
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
