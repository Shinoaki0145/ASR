import os
from unittest import TestCase, mock
from web_asr.config import Settings


class ConfigTests(TestCase):
    def test_secrets_redacted(self):
        self.assertNotIn(
            "secret-key", repr(Settings(api_key="secret-key", password="secret-key"))
        )

    def test_invalid_config(self):
        with mock.patch.dict(
            os.environ,
            {
                "WEB_FAKE": "true",
                "WEB_SHARE": "true",
                "ASR_ENV_PATH": "/nonexistent",
                "WEB_USERNAME": "",
                "WEB_PASSWORD": "",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.load()

    def test_fake_without_credentials(self):
        with mock.patch.dict(
            os.environ, {"WEB_FAKE": "true", "ASR_ENV_PATH": "/nonexistent"}, clear=True
        ):
            self.assertTrue(Settings.load().fake)
