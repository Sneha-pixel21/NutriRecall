"""UI regression checks with mocked storage; never touch real health logs."""
import unittest
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
        self.app.radio[0].set_value(page).run(timeout=30)
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
        self.assertEqual(len(self.app.metric), 0)
        self.assertIn("No entries in the last 7 days", self.app.info[0].value)
        self.rows = []
        for page in ["📥 Daily Log", "📊 Dashboard", "📅 History", "🔄 Week Compare", "✨ AI Assistant"]:
            self.open(page)


if __name__ == "__main__":
    unittest.main()
