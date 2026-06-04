"""
data/db_manager.py  –  Persistencia SQLite para clasificaciones de frutas.
"""
import sqlite3
import datetime
import threading
from pathlib import Path

DB_PATH = Path("data/db/clasificaciones.db")


class DBManager:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS clasificaciones (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                fruta         TEXT    NOT NULL,
                estado        TEXT    NOT NULL,
                confianza     REAL    NOT NULL,
                snapshot_path TEXT
            )
        """)
        self.conn.commit()

    # ── Escritura ─────────────────────────────────────────────────────────────

    def insert(self, fruit: str, state: str, confidence: float,
               snap_path: str | None = None):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self.conn.execute(
                "INSERT INTO clasificaciones "
                "(timestamp, fruta, estado, confianza, snapshot_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, fruit, state, confidence, snap_path),
            )
            self.conn.commit()

    # ── Lectura ───────────────────────────────────────────────────────────────

    def get_recent(self, n: int = 10) -> list:
        """Retorna las últimas n filas como (id, timestamp, fruta, estado, confianza, snapshot_path)."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, timestamp, fruta, estado, confianza, snapshot_path "
                "FROM clasificaciones ORDER BY id DESC LIMIT ?",
                (n,),
            )
            return cur.fetchall()

    def get_all(self) -> list:
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, timestamp, fruta, estado, confianza, snapshot_path "
                "FROM clasificaciones ORDER BY id DESC"
            )
            return cur.fetchall()

    def get_summary(self) -> dict:
        """Retorna conteos con claves tipo 'banano_maduro', 'manzana_verde', etc."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT LOWER(fruta), LOWER(estado), COUNT(*) "
                "FROM clasificaciones GROUP BY LOWER(fruta), LOWER(estado)"
            )
            return {f"{fruit}_{state}": count for fruit, state, count in cur.fetchall()}

    def close(self):
        with self._lock:
            self.conn.close()
