"""Regression checks for Pinecone deployment configuration."""

import os
import unittest
from unittest.mock import patch

from enterprise_rag_chatbot.config import Settings


class SettingsTests(unittest.TestCase):
    def test_missing_or_blank_deployment_uses_defaults(self):
        for values in ({}, {"PINECONE_CLOUD": "", "PINECONE_REGION": ""},
                       {"PINECONE_CLOUD": "  ", "PINECONE_REGION": "\t"}):
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True), \
                 patch("enterprise_rag_chatbot.config.load_dotenv"):
                settings = Settings.from_env()
                self.assertEqual((settings.cloud, settings.region), ("aws", "us-east-1"))

    def test_explicit_deployment_is_preserved_and_trimmed(self):
        values = {"PINECONE_CLOUD": " gcp ", "PINECONE_REGION": " europe-west4 "}
        with patch.dict(os.environ, values, clear=True), \
             patch("enterprise_rag_chatbot.config.load_dotenv"):
            settings = Settings.from_env()
        self.assertEqual((settings.cloud, settings.region), ("gcp", "europe-west4"))

    def test_direct_blank_deployment_fails_locally(self):
        for values in ({"cloud": ""}, {"region": " "}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "PINECONE_CLOUD"):
                Settings(**values)


if __name__ == "__main__":
    unittest.main()
