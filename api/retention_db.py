
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "retention_actions.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialise_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS retention_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pending',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%S', 'now')
                ),
                updated_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%S', 'now')
                )
            )
        """)


def create_action(customer_id, action_type, notes=""):
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO retention_actions (
                customer_id, action_type, notes
            )
            VALUES (?, ?, ?)
        """, (customer_id, action_type, notes))

        return cursor.lastrowid


def get_actions(customer_id=None):
    with get_connection() as connection:
        if customer_id:
            rows = connection.execute("""
                SELECT *
                FROM retention_actions
                WHERE customer_id = ?
                ORDER BY id DESC
            """, (customer_id,)).fetchall()
        else:
            rows = connection.execute("""
                SELECT *
                FROM retention_actions
                ORDER BY id DESC
            """).fetchall()

        return [dict(row) for row in rows]


def update_action(action_id, status, notes=None):
    allowed_statuses = {
        "Pending",
        "Contacted",
        "Offer Made",
        "Retained",
        "Declined",
        "Unreachable",
        "Closed"
    }

    if status not in allowed_statuses:
        raise ValueError("Invalid retention action status.")

    with get_connection() as connection:
        if notes is None:
            cursor = connection.execute("""
                UPDATE retention_actions
                SET status = ?,
                    updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now')
                WHERE id = ?
            """, (status, action_id))
        else:
            cursor = connection.execute("""
                UPDATE retention_actions
                SET status = ?,
                    notes = ?,
                    updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now')
                WHERE id = ?
            """, (status, notes, action_id))

        return cursor.rowcount > 0
