import tempfile
import unittest
from pathlib import Path

from channel_store import SettingsStore


class SettingsStoreChannelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = SettingsStore(
            database_path=str(Path(self.temporary_directory.name) / "response2chat.db"),
            default_admin_username="admin",
            default_admin_password="admin123456",
        )
        self.store.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def assert_cached_lookup(self, access_key: str, expected_channel_id: int | None) -> None:
        original_connect = self.store._connect
        connection_calls = 0

        def counted_connect():
            nonlocal connection_calls
            connection_calls += 1
            return original_connect()

        self.store._connect = counted_connect
        try:
            channel = self.store.get_channel_by_access_key(access_key)
        finally:
            self.store._connect = original_connect

        actual_channel_id = channel["id"] if channel else None
        self.assertEqual(actual_channel_id, expected_channel_id)
        self.assertEqual(connection_calls, 0)

    def test_access_key_cache_is_refreshed_by_channel_changes(self) -> None:
        channel = self.store.create_channel(
            name="cache-test",
            base_url="https://example.com/v1",
            upstream_api_key="upstream-key",
        )
        channel_id = channel["id"]
        original_access_key = channel["access_key"]

        self.assert_cached_lookup(original_access_key, channel_id)
        self.assert_cached_lookup("r2c_unknown", None)

        updated_channel = self.store.update_channel(
            channel_id=channel_id,
            name="cache-test",
            base_url="https://example.com/v2",
            upstream_api_key="updated-upstream-key",
            description="updated",
            enabled=False,
        )
        self.assertIsNotNone(updated_channel)
        self.assert_cached_lookup(original_access_key, channel_id)
        cached_channel = self.store.get_channel_by_access_key(original_access_key)
        assert cached_channel is not None
        self.assertFalse(cached_channel["enabled"])

        enabled_channel = self.store.set_channel_enabled(channel_id, True)
        self.assertIsNotNone(enabled_channel)
        self.assert_cached_lookup(original_access_key, channel_id)
        cached_channel = self.store.get_channel_by_access_key(original_access_key)
        assert cached_channel is not None
        self.assertTrue(cached_channel["enabled"])

        rotated_channel = self.store.rotate_access_key(channel_id)
        self.assertIsNotNone(rotated_channel)
        assert rotated_channel is not None
        rotated_access_key = rotated_channel["access_key"]
        self.assertNotEqual(rotated_access_key, original_access_key)
        self.assert_cached_lookup(original_access_key, None)
        self.assert_cached_lookup(rotated_access_key, channel_id)

        self.assertTrue(self.store.delete_channel(channel_id))
        self.assert_cached_lookup(rotated_access_key, None)


if __name__ == "__main__":
    unittest.main()