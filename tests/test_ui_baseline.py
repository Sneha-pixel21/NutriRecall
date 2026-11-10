"""UI regression checks with mocked storage; never touch real health logs."""
import unittest
import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils import db


class BaselineUI(unittest.TestCase):
    def setUp(self):
        self.rows = [self.row(1, date.today())]
        self.mocks = {}
        replacements = {
            "init_db": {},
            "load_data": {"side_effect": self.load},
            "export_csv": {"return_value": b"date,weight\n"},
            "insert_entry": {"return_value": True},
            "update_entry": {},
            "delete_entry": {},
        }
        for name, kwargs in replacements.items():
            patcher = patch.object(db, name, **kwargs)
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))

    @staticmethod
    def row(entry_id, day):
        return dict(id=entry_id, date=pd.Timestamp(day), weight=77.4,
                    protein=100.0, sleep_hours=6.0, workout=1)

    def load(self):
        return pd.DataFrame(self.rows, columns=["id", "date", "weight", "protein", "sleep_hours", "workout"])

    def open(self, page):
        self.app.run(timeout=30)
        self.app.sidebar.radio[0].set_value(page).run(timeout=30)
        self.assertEqual(len(self.app.exception), 0)

    def button(self, label):
        return next(b for b in self.app.button if b.label == label)

    def test_empty_form_and_zero_weight(self):
        self.app.run(timeout=30)
        self.button("💾 Save Entry").click().run()
        self.assertIn("fill in all fields", self.app.error[0].value)
        for field in self.app.number_input:
            field.set_value(0.0)
        self.button("💾 Save Entry").click().run()
        self.assertIn("greater than zero", self.app.error[0].value)
        self.mocks["insert_entry"].assert_not_called()

    def test_zero_protein_and_sleep_are_valid(self):
        self.app.run(timeout=30)
        for field, value in zip(self.app.number_input, [70.0, 0.0, 0.0]):
            field.set_value(value)
        self.button("💾 Save Entry").click().run()
        self.mocks["insert_entry"].assert_called_once_with(str(date.today()), 70.0, 0.0, 0.0, 0)

    def test_history_constraints_and_rounding(self):
        self.open("📅 History")
        self.assertEqual([n.min for n in self.app.number_input], [0.0, 0.0, 0.0])
        self.assertEqual(self.app.number_input[2].max, 24.0)
        self.assertEqual(self.app.dataframe[0].value["Target Protein (g)"].iloc[0], 123.8)
        self.app.number_input[0].set_value(0.0)
        self.button("💾 Save changes").click().run()
        self.mocks["update_entry"].assert_not_called()

    def test_history_hides_index_and_keeps_newest_date_first(self):
        self.rows.append(self.row(2, date.today() - timedelta(days=2)))
        self.open("📅 History")
        table = self.app.dataframe[0]
        self.assertTrue(json.loads(table.proto.columns)["_index"]["hidden"])
        self.assertEqual(table.value["Date"].tolist(), [
            str(date.today()), str(date.today() - timedelta(days=2)),
        ])
        self.mocks["update_entry"].assert_not_called()
        self.mocks["delete_entry"].assert_not_called()

    def test_dashboard_preserves_metrics_and_all_chart_series(self):
        self.open("📊 Dashboard")
        self.assertEqual([m.value for m in self.app.metric],
                         ["100g", "124g", "6.0 hrs", "1 of 1 logged days"])
        charts = [json.loads(c.proto.spec) for c in self.app.get("plotly_chart")]
        self.assertEqual(len(charts), 4)
        self.assertEqual(charts[0]["data"][0]["value"], 8.6)
        self.assertEqual(charts[0]["data"][0]["gauge"]["axis"]["range"], [1, 10])
        self.assertEqual([t["name"] for c in charts[1:] for t in c["data"]],
                         ["Weight", "Protein (g)", "Target", "Sleep (hrs)"])
        self.assertEqual(charts[3]["layout"]["shapes"][0]["y0"], 7)
        self.assertEqual(len(charts[3]["data"][0]["y"]), 7)
        self.assertEqual(sum(v is not None for v in charts[3]["data"][0]["y"]), 1)

    def test_history_delete_needs_confirmation(self):
        self.open("📅 History")
        self.assertTrue(self.button("Delete entry").disabled)
        self.app.checkbox[0].check().run()
        self.assertFalse(self.button("Delete entry").disabled)
        self.button("Delete entry").click().run()
        self.mocks["delete_entry"].assert_called_once_with(1)

    def test_last_added_delete_needs_confirmation(self):
        self.rows.append(self.row(2, date.today() - timedelta(days=10)))
        self.app.run(timeout=30)
        self.assertTrue(self.button("Delete Last Entry").disabled)
        self.app.checkbox[0].check().run()
        self.button("Delete Last Entry").click().run()
        self.mocks["delete_entry"].assert_called_once_with(2)

    def test_missing_previous_week_has_no_delta(self):
        self.open("🔄 Week Compare")
        self.assertTrue(any("Missing data is not zero" in i.value for i in self.app.info))
        self.assertTrue(all(not m.delta for m in self.app.metric))

    def test_real_zero_previous_week_still_has_delta(self):
        previous = self.row(2, date.today() - timedelta(days=7))
        previous.update(protein=0.0, sleep_hours=0.0, workout=0)
        self.rows.append(previous)
        self.open("🔄 Week Compare")
        self.assertEqual(self.app.metric[0].delta, "+100 vs last week")

    def test_stale_dashboard_and_empty_pages(self):
        self.rows = [self.row(1, date.today() - timedelta(days=30))]
        self.open("📊 Dashboard")
        self.assertEqual(len(self.app.metric), 4)
        self.button("Go to current week").click().run()
        self.assertEqual(len(self.app.metric), 0)
        self.assertIn("No entries in this period", self.app.info[0].value)
        self.rows = []
        for page in ["📥 Daily Log", "📊 Dashboard", "📅 History", "🔄 Week Compare", "✨ AI Assistant"]:
            self.open(page)
        self.open("📊 Dashboard")
        self.app.radio(key="dashboard_period").set_value("All time").run()
        self.assertTrue(any("All time: no recorded dates" in m.value for m in self.app.markdown))
        self.assertIn("No entries yet", self.app.info[0].value)
        self.assertFalse(any(d.key == "dashboard_picker" for d in self.app.date_input))
        self.assertFalse(self.button("Go to current week").disabled)

    def test_dashboard_calendar_navigation_and_empty_period_controls(self):
        self.rows = [self.row(1, date(2024, 1, 31)), self.row(2, date(2024, 2, 29))]
        self.open("📊 Dashboard")
        self.assertTrue(any("26 Feb 2024 – 03 Mar 2024" in m.value for m in self.app.markdown))
        self.button("Next →").click().run()
        self.assertEqual(len(self.app.metric), 0)
        self.assertFalse(self.button("← Previous").disabled)
        self.assertTrue(any(d.key == "dashboard_picker" for d in self.app.date_input))
        self.button("← Previous").click().run()
        self.assertEqual(len(self.app.metric), 4)
        self.app.radio(key="dashboard_period").set_value("Month").run()
        self.assertTrue(any("01 Feb 2024 – 29 Feb 2024" in m.value for m in self.app.markdown))
        self.button("← Previous").click().run()
        self.assertTrue(any("01 Jan 2024 – 31 Jan 2024" in m.value for m in self.app.markdown))
        self.app.date_input(key="dashboard_picker").set_value(date(2024, 2, 20)).run()
        self.assertTrue(any("01 Feb 2024 – 29 Feb 2024" in m.value for m in self.app.markdown))
        self.app.radio(key="dashboard_period").set_value("Year").run()
        charts = [json.loads(c.proto.spec) for c in self.app.get("plotly_chart")]
        self.assertEqual(len(charts[1]["data"][0]["y"]), 12)
        self.app.radio(key="dashboard_period").set_value("All time").run()
        self.assertEqual(self.app.metric[3].value, "2 of 2 logged days")
        self.assertFalse(any(d.key == "dashboard_picker" for d in self.app.date_input))
        self.button("Go to current week").click().run()
        self.assertEqual(self.app.radio(key="dashboard_period").value, "Week")
        self.assertEqual(self.app.date_input(key="dashboard_picker").value, date.today())

    def test_dashboard_metrics_and_charts_share_filter_with_zeros(self):
        outside = self.row(1, date(2024, 1, 31))
        outside.update(weight=200., protein=999., sleep_hours=20.)
        zero = self.row(2, date(2024, 2, 26))
        zero.update(protein=0., sleep_hours=0., workout=0)
        self.rows = [outside, zero, self.row(3, date(2024, 2, 29))]
        self.open("📊 Dashboard")
        self.assertEqual([m.value for m in self.app.metric],
                         ["50g", "124g", "3.0 hrs", "1 of 2 logged days"])
        charts = [json.loads(c.proto.spec) for c in self.app.get("plotly_chart")]
        self.assertEqual(charts[0]["data"][0]["value"], 4.8)
        self.assertEqual(charts[1]["data"][0]["y"], [77.4, None, None, 77.4, None, None, None])
        self.assertEqual(charts[2]["data"][0]["y"], [0., None, None, 100., None, None, None])
        self.assertTrue(any("Coverage: 2 of 7" in c.value for c in self.app.caption))
        self.app.radio(key="dashboard_period").set_value("Year").run()
        self.assertEqual(self.app.metric[0].value, "366g")
        self.assertEqual(self.app.metric[3].value, "2 of 3 logged days")

    def test_dashboard_future_navigation_limits(self):
        self.rows = [self.row(1, date.today() - timedelta(days=400))]
        self.open("📊 Dashboard")
        self.button("Go to current week").click().run()
        for mode in ["Week", "Month", "Year"]:
            with self.subTest(mode=mode):
                self.app.radio(key="dashboard_period").set_value(mode).run()
                self.assertTrue(self.button("Next →").disabled)
                self.assertEqual(self.app.date_input(key="dashboard_picker").max, date.today())
                self.assertFalse(any("future dates" in c.value for c in self.app.caption))
                self.assertEqual(len(self.app.warning), 0)
                self.button("← Previous").click().run()
                self.assertFalse(self.button("Next →").disabled)
                self.button("Next →").click().run()
                self.assertTrue(self.button("Next →").disabled)
        self.app.date_input(key="dashboard_picker").set_value(date.today()).run()
        self.assertEqual(len(self.app.exception), 0)

    def test_dashboard_future_record_default_is_capped_without_changing_storage(self):
        self.rows = [self.row(1, date.today() + timedelta(days=20))]
        self.open("📊 Dashboard")
        self.assertEqual(self.app.date_input(key="dashboard_picker").value, date.today())
        self.assertTrue(self.button("Next →").disabled)
        self.assertEqual(len(self.app.warning), 0)
        self.mocks["insert_entry"].assert_not_called()
        self.mocks["delete_entry"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
