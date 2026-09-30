import sqlite3
import os
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

logger = logging.getLogger("trading_bot")

class TradeLedger:
    """
    Trade Ledger database layer supporting both PostgreSQL and SQLite.
    Automatically uses PostgreSQL when DATABASE_URL is configured,
    falling back to SQLite for local development and testing.

    Thread-safe: uses a connection pool for Postgres and a threading lock for SQLite.
    """
    def __init__(self, db_path=None, database_url=None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.is_postgres = bool(self.database_url and ("postgres" in self.database_url))
        self.db_path = db_path or os.path.join(os.getenv("LOG_DIR", "."), "trades.db")
        
        self.placeholder = "%s" if self.is_postgres else "?"
        self._lock = threading.Lock()
        self._pool = None
        
        if self.is_postgres:
            if not POSTGRES_AVAILABLE:
                raise ImportError("psycopg2 is required for PostgreSQL but not installed.")
            # Retry connection pool creation — Postgres may still be starting in K8s
            max_retries = 10
            for attempt in range(1, max_retries + 1):
                try:
                    self._pool = SimpleConnectionPool(minconn=2, maxconn=10, dsn=self.database_url)
                    logger.info("TradeLedger: Configured with PostgreSQL backend (pooled, 2-10 connections).")
                    break
                except psycopg2.OperationalError as e:
                    if attempt == max_retries:
                        logger.error(f"TradeLedger: Failed to connect to PostgreSQL after {max_retries} attempts.")
                        raise
                    logger.warning(f"TradeLedger: PostgreSQL not ready (attempt {attempt}/{max_retries}): {e}")
                    time.sleep(3)
        else:
            logger.info(f"TradeLedger: Configured with SQLite backend ({self.db_path}).")
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            # Persistent connection with thread-safety handled by self._lock
            self._sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """
        Context manager that yields a database connection.
        - Postgres: pulls from pool, returns on exit.
        - SQLite: uses a lock to serialize access with a persistent connection.
        """
        if self.is_postgres:
            conn = self._pool.getconn()
            try:
                yield conn
            finally:
                self._pool.putconn(conn)
        else:
            with self._lock:
                yield self._sqlite_conn

    def _init_db(self):
        """Create trades table if not exists."""
        id_col = "id SERIAL PRIMARY KEY" if self.is_postgres else "id INTEGER PRIMARY KEY AUTOINCREMENT"
        
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS trades (
                {id_col},
                trade_id TEXT UNIQUE,
                instrument TEXT NOT NULL,
                direction TEXT NOT NULL,
                units INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                spread_pips REAL,
                realized_pl REAL DEFAULT 0.0,
                status TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                closed_at TEXT
            )
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_sql)
                conn.commit()
        except Exception as e:
            logger.error(f"TradeLedger initialization error: {e}")

    def record_open(self, trade_id, instrument, direction, units, entry_price, stop_loss=None, take_profit=None, spread_pips=0.0):
        """Record trade opening."""
        now = datetime.now(timezone.utc).isoformat()
        p = self.placeholder
        sql = f"""
            INSERT INTO trades (
                trade_id, instrument, direction, units,
                entry_price, stop_loss, take_profit, spread_pips,
                status, opened_at
            ) VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, 'OPEN', {p})
        """
        params = (
            str(trade_id), instrument, direction, int(units),
            float(entry_price), float(stop_loss) if stop_loss else None,
            float(take_profit) if take_profit else None, float(spread_pips), now
        )
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                logger.info(f"Ledger: Recorded open trade {trade_id} ({instrument} {direction} {units:,}u)")
        except Exception as e:
            logger.error(f"Ledger Error recording open trade {trade_id}: {e}")

    def record_close(self, trade_id, exit_price, realized_pl):
        """Record trade closure."""
        now = datetime.now(timezone.utc).isoformat()
        p = self.placeholder
        sql = f"""
            UPDATE trades
            SET exit_price = {p}, realized_pl = {p}, status = 'CLOSED', closed_at = {p}
            WHERE trade_id = {p}
        """
        params = (float(exit_price), float(realized_pl), now, str(trade_id))
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                logger.info(f"Ledger: Recorded close trade {trade_id} | Exit: {exit_price} | P/L: ${realized_pl:.2f}")
        except Exception as e:
            logger.error(f"Ledger Error recording close trade {trade_id}: {e}")

    def get_daily_pnl(self, target_date=None):
        """
        Calculate total realized PnL for closed trades on a specific UTC day (YYYY-MM-DD).
        Defaults to current UTC day.
        """
        if not target_date:
            target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        p = self.placeholder
        # Use proper date range comparison instead of fragile LIKE pattern
        start_of_day = f"{target_date}T00:00:00"
        end_of_day = f"{target_date}T23:59:59"
        sql = f"""
            SELECT 
                COUNT(*),
                COALESCE(SUM(realized_pl), 0.0),
                SUM(CASE WHEN realized_pl > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN realized_pl < 0 THEN 1 ELSE 0 END)
            FROM trades
            WHERE status = 'CLOSED' AND closed_at >= {p} AND closed_at <= {p}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (start_of_day, end_of_day))
                row = cursor.fetchone()
                return {
                    "date": target_date,
                    "total_trades": int(row[0] or 0),
                    "realized_pl": round(float(row[1] or 0.0), 2),
                    "wins": int(row[2] or 0),
                    "losses": int(row[3] or 0)
                }
        except Exception as e:
            logger.error(f"Ledger Error fetching daily PnL: {e}")
            return {"date": target_date, "total_trades": 0, "realized_pl": 0.0, "wins": 0, "losses": 0}

    def get_recent_trades(self, limit=20):
        """Fetch recent trades ordered by newest first."""
        p = self.placeholder
        sql = f"""
            SELECT trade_id, instrument, direction, units,
                   entry_price, exit_price, stop_loss, take_profit,
                   spread_pips, realized_pl, status, opened_at, closed_at
            FROM trades
            ORDER BY id DESC
            LIMIT {p}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (limit,))
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "trade_id": r[0],
                        "instrument": r[1],
                        "direction": r[2],
                        "units": r[3],
                        "entry_price": r[4],
                        "exit_price": r[5],
                        "stop_loss": r[6],
                        "take_profit": r[7],
                        "spread_pips": r[8],
                        "realized_pl": r[9],
                        "status": r[10],
                        "opened_at": r[11],
                        "closed_at": r[12]
                    })
                return results
        except Exception as e:
            logger.error(f"Ledger Error fetching recent trades: {e}")
            return []
