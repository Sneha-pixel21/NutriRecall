"""Synthetic dashboard data only: no database, embeddings, or API calls."""
import unittest
from datetime import date

import pandas as pd

from utils.dashboard import dashboard_view, period_bounds, shift_period, trend_charts, can_advance


def records(*items):
    return pd.DataFrame([
        dict(date=pd.Timestamp(day), weight=70., protein=float(protein),
             protein_required=112., sleep_hours=float(sleep), health_score_10=8., workout=workout)
        for day, protein, sleep, workout in items
    ], columns=["date", "weight", "protein", "protein_required", "sleep_hours", "health_score_10", "workout"])


class DashboardPeriods(unittest.TestCase):
    def test_next_allows_current_period_but_not_future_period(self):
        today = date(2024, 2, 29)
        for mode, previous in [("Week", date(2024, 2, 19)),
                               ("Month", date(2024, 1, 31)), ("Year", date(2023, 12, 31))]:
            with self.subTest(mode=mode):
                self.assertTrue(can_advance(mode, previous, today))
                self.assertFalse(can_advance(mode, today, today))
        self.assertTrue(can_advance("Month", date(2023, 12, 31), date(2024, 1, 1)))
        self.assertTrue(can_advance("Week", date(2023, 12, 31), date(2024, 1, 1)))

    def test_current_period_coverage_stops_today_without_truncating_range(self):
        today = date(2024, 2, 28)
        data = records(("2024-02-26", 0, 0, 0), ("2024-02-28", 100, 6, 1),
                       ("2024-02-29", 100, 6, 1))
        original = data.copy(deep=True)
        for mode, elapsed, full in [("Week", 3, 7), ("Month", 28, 29), ("Year", 59, 366)]:
            with self.subTest(mode=mode):
                view = dashboard_view(data, mode, today, today=today)
                self.assertEqual(view["coverage_days"], elapsed)
                self.assertEqual(view["coverage_logged_days"], 2)
                self.assertEqual(view["missing_days"], elapsed - 2)
                self.assertEqual(view["period_days"], full)
                self.assertGreater(view["end"], today)
                self.assertEqual(view["logged_days"], 3)  # Existing records remain available.
        pd.testing.assert_frame_equal(data, original)
        historical = dashboard_view(data, "Month", today, today=date(2024, 3, 1))
        self.assertEqual((historical["coverage_days"], historical["coverage_logged_days"]), (29, 3))
        all_time = dashboard_view(data, "All time", today, today=today)
        self.assertEqual((all_time["coverage_days"], all_time["coverage_logged_days"]), (4, 3))

    def test_calendar_boundaries(self):
        empty = records()
        for mode, anchor, expected in [
            ("Week", date(2025, 1, 1), (date(2024, 12, 30), date(2025, 1, 5))),
            ("Week", date(2024, 3, 3), (date(2024, 2, 26), date(2024, 3, 3))),
            ("Month", date(2024, 2, 10), (date(2024, 2, 1), date(2024, 2, 29))),
            ("Month", date(2023, 2, 10), (date(2023, 2, 1), date(2023, 2, 28))),
            ("Year", date(2024, 2, 29), (date(2024, 1, 1), date(2024, 12, 31))),
        ]:
            with self.subTest(mode=mode, anchor=anchor):
                self.assertEqual(period_bounds(mode, anchor, empty), expected)

    def test_navigation_crosses_year_and_leap_boundaries(self):
        self.assertEqual(shift_period("Month", date(2024, 1, 31), 1), date(2024, 2, 1))
        self.assertEqual(shift_period("Month", date(2024, 1, 31), -1), date(2023, 12, 1))
        self.assertEqual(shift_period("Month", date(2024, 12, 31), 1), date(2025, 1, 1))
        self.assertEqual(shift_period("Year", date(2024, 2, 29), 1), date(2025, 1, 1))
        self.assertEqual(shift_period("Week", date(2025, 1, 1), -1), date(2024, 12, 23))

    def test_week_zeros_gaps_coverage_and_inclusive_end(self):
        data = records(("2024-02-25", 999, 20, 1), ("2024-02-26", 0, 0, 0),
                       ("2024-03-03 23:59", 100, 6, 1), ("2024-03-04", 999, 20, 1))
        view = dashboard_view(data, "Week", date(2024, 2, 29))
        self.assertEqual((view["logged_days"], view["period_days"], view["missing_days"]), (2, 7, 5))
        self.assertEqual(view["workout_days"], 1)
        self.assertEqual(view["means"]["protein"], 50)
        self.assertEqual(view["means"]["sleep_hours"], 3)
        self.assertEqual(len(view["readings"]), 7)
        self.assertEqual(view["readings"]["protein"].iloc[0], 0)
        self.assertTrue(pd.isna(view["readings"]["protein"].iloc[1]))
        charts = trend_charts(view)
        for chart in charts:
            for trace in chart.data:
                self.assertEqual(len(trace.x), 7)
                self.assertEqual(trace.x[0], pd.Timestamp("2024-02-26"))
                self.assertEqual(trace.x[-1], pd.Timestamp("2024-03-03"))
        self.assertIsNone(charts[0].data[0].y[1])
        self.assertFalse(charts[0].data[0].connectgaps)

    def test_month_filters_calendar_and_keeps_daily_readings(self):
        data = records(("2024-01-31", 999, 8, 1), ("2024-02-01", 0, 0, 0),
                       ("2024-02-29", 100, 6, 1), ("2024-03-01", 999, 8, 1))
        view = dashboard_view(data, "Month", date(2024, 2, 29))
        self.assertEqual(len(view["readings"]), 29)
        self.assertEqual(view["logged_days"], 2)
        self.assertEqual(view["means"]["protein"], 50)
        self.assertFalse(view["monthly"])

    def test_year_monthly_means_counts_and_weighted_headline(self):
        data = records(("2023-12-31", 999, 9, 1), ("2024-01-01", 0, 0, 0),
                       ("2024-01-31", 100, 6, 1), ("2024-02-29", 200, 9, 1),
                       ("2025-01-01", 999, 9, 1))
        view = dashboard_view(data, "Year", date(2024, 3, 1))
        self.assertEqual(view["period_days"], 366)
        self.assertEqual(view["means"]["protein"], 100)  # Not the monthly mean of 125.
        self.assertEqual(view["readings"]["protein"].iloc[:2].tolist(), [50, 200])
        self.assertEqual(view["readings"]["logged_days"].iloc[:3].tolist(), [2, 1, 0])
        self.assertTrue(pd.isna(view["readings"]["protein"].iloc[2]))
        for chart in trend_charts(view):
            for trace in chart.data:
                self.assertEqual(len(trace.x), 12)
                self.assertEqual(list(trace.customdata[0]), [2])
                self.assertIn("logged days", trace.hovertemplate)

    def test_all_time_extent_and_readability_threshold(self):
        short = records(("2024-01-01", 0, 0, 0), ("2024-12-31", 100, 6, 1))
        daily = dashboard_view(short, "All time", date(2026, 1, 1))
        self.assertEqual(daily["period_days"], 366)
        self.assertFalse(daily["monthly"])
        long = records(("2024-01-01", 0, 0, 0), ("2025-01-01", 100, 6, 1))
        monthly = dashboard_view(long, "All time", date(2026, 1, 1))
        self.assertTrue(monthly["monthly"])
        self.assertEqual(monthly["start"], date(2024, 1, 1))
        self.assertEqual(monthly["end"], date(2025, 1, 1))
        self.assertEqual(len(monthly["readings"]), 13)
        self.assertEqual(monthly["means"]["protein"], 50)

    def test_empty_periods_keep_unknown_means(self):
        data = records(("2020-01-01", 50, 6, 1))
        for mode in ["Week", "Month", "Year"]:
            with self.subTest(mode=mode):
                view = dashboard_view(data, mode, date(2024, 2, 29))
                self.assertEqual(view["logged_days"], 0)
                self.assertIsNone(view["means"]["protein"])
                self.assertTrue(view["readings"]["protein"].isna().all())
        for mode in ["Week", "Month", "Year", "All time"]:
            view = dashboard_view(records(), mode, date(2024, 2, 29))
            self.assertEqual(view["logged_days"], 0)


if __name__ == "__main__":
    unittest.main()
