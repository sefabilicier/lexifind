"""
In-memory API key manager.

Generates a cryptographically secure API key at server startup.
Key is printed to terminal and stored in memory only — never persisted.

Design decisions:
  - No .env storage: key rotates on every restart (dev-friendly)
  - secrets.token_urlsafe: 256-bit entropy, URL-safe
  - Singleton: same key served across all requests in one session
  - Format: lf_v1-{token} (Stripe/GitHub convention)
"""

import secrets
from datetime import datetime, timezone

from app.observability.logger import get_logger

logger = get_logger(__name__)


class APIKeyManager:
    """
    Singleton in-memory API key store.
    Key is generated once at startup and lives until process exits.
    """

    _instance = None
    _key: str | None = None
    _generated_at: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate(self) -> str:
        """
        Generate and store a new API key.
        Called once at application startup.
        """
        token = secrets.token_urlsafe(32)
        self._key = f"lf_v1-{token}"
        self._generated_at = datetime.now(timezone.utc).isoformat()

        self._print_banner()
        logger.info("api_key.generated", generated_at=self._generated_at)

        return self._key

    def get_key(self) -> str | None:
        """Return the current active key."""
        return self._key

    def is_valid(self, key: str) -> bool:
        """Validate an incoming API key against the in-memory key."""
        if not self._key:
            return False
        return secrets.compare_digest(self._key, key)

    def _print_banner(self) -> None:
        """Print the generated key clearly to terminal."""
        border = "=" * 60
        print(f"\n{border}")
        print("  🔑  LexiFind — API Key Generated")
        print(border)
        print(f"\n  Key:  {self._key}")
        print(f"  At:   {self._generated_at}")
        print("\n  Use this key in every request:")
        print(f"  X-API-Key: {self._key}")
        print(f"\n  ⚠️  Key lives in memory only.")
        print("      Restart = new key.")
        print(f"\n{border}\n")


# Global singleton instance
api_key_manager = APIKeyManager()