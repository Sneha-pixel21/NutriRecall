"""
NutriRecall data layer — SQLite via pandas + sqlite3
Drop-in replacement for the CSV approach, no ORM needed.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/health_logs.db")


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT UNIQUE,
                weight      REAL,
                protein     REAL,
                sleep_hours REAL,
                workout     INTEGER
            )
        """)


def load_data() -> pd.DataFrame:
    init_db()
    df = pd.read_sql("SELECT * FROM logs ORDER BY date", _conn())
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def entry_exists(date_str: str) -> bool:
    with _conn() as con:
        row = con.execute("SELECT 1 FROM logs WHERE date=?", (date_str,)).fetchone()
    return row is not None


def insert_entry(date_str: str, weight: float, protein: float, sleep: float, workout: int) -> bool:
    """Returns True on success, False if date already exists."""
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO logs (date, weight, protein, sleep_hours, workout) VALUES (?,?,?,?,?)",
                (date_str, weight, protein, sleep, workout),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def update_entry(entry_id: int, weight: float, protein: float, sleep: float, workout: int):
    with _conn() as con:
        con.execute(
            "UPDATE logs SET weight=?, protein=?, sleep_hours=?, workout=? WHERE id=?",
            (weight, protein, sleep, workout, entry_id),
        )


def delete_entry(entry_id: int):
    with _conn() as con:
        con.execute("DELETE FROM logs WHERE id=?", (entry_id,))


def delete_last_entry():
    with _conn() as con:
        con.execute("DELETE FROM logs WHERE id=(SELECT MAX(id) FROM logs)")


def export_csv() -> bytes:
    df = load_data()
    return df.to_csv(index=False).encode()