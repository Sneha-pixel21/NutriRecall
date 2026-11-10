"""Calendar browsing and chart data for the dashboard; no storage or model calls."""
from calendar import monthrange
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go


READINGS = ["weight", "protein", "protein_required", "sleep_hours", "health_score_10"]
DAILY_HISTORY_LIMIT = 366


def period_bounds(mode, anchor, data):
    if mode == "Week":
        start = anchor - timedelta(days=anchor.weekday())
        return start, start + timedelta(days=6)
    if mode == "Month":
        return anchor.replace(day=1), anchor.replace(day=monthrange(anchor.year, anchor.month)[1])
    if mode == "Year":
        return date(anchor.year, 1, 1), date(anchor.year, 12, 31)
    if mode == "All time":
        if data.empty:
            return None, None
        dates = pd.to_datetime(data["date"])
        return dates.min().date(), dates.max().date()
    raise ValueError(f"Unknown dashboard period: {mode}")


def shift_period(mode, anchor, step):
    """Navigate using period starts, avoiding month-end and leap-day drift."""
    if mode == "Week":
        return anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=step)
    if mode == "Month":
        month = anchor.year * 12 + anchor.month - 1 + step
        year, month = divmod(month, 12)
        return date(year, month + 1, 1)
    if mode == "Year":
        return date(anchor.year + step, 1, 1)
    raise ValueError("All time does not have adjacent periods.")


def can_advance(mode, anchor, today=None):
    today = today or date.today()
    return mode != "All time" and shift_period(mode, anchor, 1) <= today


def dashboard_view(data, mode, anchor, today=None):
    today = today or date.today()
    start, end = period_bounds(mode, anchor, data)
    rows = data.reindex(columns=list(dict.fromkeys([*data.columns, "date", "workout", *READINGS]))).copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    if start is not None:
        rows = rows.loc[rows["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    rows = rows.sort_values("date")
    days = (end - start).days + 1 if start else 0
    coverage_end = min(end, today) if end and mode != "All time" else end
    coverage_days = max(0, (coverage_end - start).days + 1) if start else 0
    coverage_logged = int((rows["date"] <= pd.Timestamp(coverage_end)).sum()) if coverage_end else 0
    monthly = mode == "Year" or (mode == "All time" and days > DAILY_HISTORY_LIMIT)
    means = {field: None if rows.empty or rows[field].isna().all()
             else float(rows[field].mean()) for field in READINGS}
    if start is None:
        readings = pd.DataFrame(columns=["date", *READINGS, "logged_days"])
    elif monthly:
        index = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
        grouped = rows.set_index("date").resample("MS")
        readings = grouped[READINGS].mean().reindex(index)
        readings["logged_days"] = grouped.size().reindex(index, fill_value=0).astype(int)
        readings = readings.rename_axis("date").reset_index()
    else:
        # Reindex with NaN, never zero: a gap must remain visibly unknown.
        index = pd.date_range(start, end, freq="D")
        readings = rows.reindex(columns=["date", *READINGS]).set_index("date").reindex(index)
        readings["logged_days"] = index.isin(rows["date"]).astype(int)
        readings = readings.rename_axis("date").reset_index()
    return dict(start=start, end=end, rows=rows, readings=readings, means=means,
                logged_days=len(rows), period_days=days, coverage_days=coverage_days,
                coverage_logged_days=coverage_logged, missing_days=coverage_days - coverage_logged,
                workout_days=int(rows["workout"].sum()) if not rows.empty else 0,
                monthly=monthly)


def trend_charts(view):
    """All three trends share one filtered/aggregated table and date extent."""
    table = view["readings"]
    dates = table["date"].tolist()
    counts = table[["logged_days"]].values.tolist()
    monthly = view["monthly"]
    date_format = "%b %Y" if monthly else "%d %b %Y"
    count_hover = "<br>%{customdata[0]} logged days" if monthly else ""

    def values(field):
        return [None if pd.isna(v) else float(v) for v in table[field]]

    def hover(unit):
        return f"%{{x|{date_format}}}<br>%{{y:.1f}} {unit}{count_hover}<extra>%{{fullData.name}}</extra>"

    weight = go.Figure(go.Scatter(
        x=dates, y=values("weight"), customdata=counts, mode="lines+markers",
        name="Weight", connectgaps=False, line=dict(color="#00E5FF", width=2),
        marker=dict(size=5), hovertemplate=hover("kg")))
    protein = go.Figure(go.Bar(
        x=dates, y=values("protein"), customdata=counts, name="Protein (g)",
        marker_color="#00E5FF", hovertemplate=hover("g")))
    protein.add_trace(go.Scatter(
        x=dates, y=values("protein_required"), customdata=counts, mode="lines+markers",
        marker=dict(size=4),
        name="Target", connectgaps=False, line=dict(color="#B5AFBF", dash="dash", width=2),
        hovertemplate=hover("g")))
    sleep_values = values("sleep_hours")
    sleep = go.Figure(go.Bar(
        x=dates, y=sleep_values, customdata=counts, name="Sleep (hrs)",
        marker_color=["#68616F" if h is None or h < 6 else
                      ("#9572B5" if h < 7 else "#00E5FF") for h in sleep_values],
        hovertemplate=hover("hours")))
    sleep.add_hline(y=7, line_dash="dash", line_color="#B5AFBF", annotation_text="7h target")
    for fig, unit, height in [(weight, "kg", 260), (protein, "grams", 260), (sleep, "hours", 220)]:
        first = pd.Timestamp(view["start"].replace(day=1) if monthly else view["start"])
        last = pd.Timestamp(view["end"]) + pd.Timedelta(days=1 if monthly else 0.5)
        # Half a bin of space keeps the first and last bars fully visible.
        pad = pd.Timedelta(days=15 if monthly else 0.5)
        fig.update_layout(height=height, margin=dict(t=30, b=10, l=10, r=10),
                          xaxis_title="", yaxis_title=unit, hovermode="x unified",
                          legend=dict(orientation="h", y=1.1))
        fig.update_xaxes(range=[first - pad, last], dtick="M1" if monthly else 86400000,
                         tickformat="%b\n%Y" if monthly else "%d %b",
                         nticks=12 if monthly else 8)
        if (not monthly and view["period_days"] > 7) or (monthly and len(table) > 12):
            fig.update_xaxes(dtick=None, tickmode="auto")
    return weight, protein, sleep
