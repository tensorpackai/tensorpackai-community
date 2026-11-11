"""
Unit tests for PrometheusTelemetryCollector energy integration and token-efficiency calculations.
"""

import time
import unittest
from unittest.mock import patch
from tensorpack.exporter.prometheus import (
    PrometheusTelemetryCollector,
    JOULES_PER_TOKEN,
    CUMULATIVE_JOULES_PER_TOKEN,
    TOKENS_PER_WATT,
    fetch_vllm_generation_tokens,
)


class TestPrometheusExporter(unittest.TestCase):
    def test_energy_integration_and_inproc_tokens(self):
        collector = PrometheusTelemetryCollector(
            sample_interval_s=0.1,
            vllm_metrics_urls=[],
            mock=True,
        )
        # First sample initializes _last_ts and _last_power_w
        collector.sample_once()
        time.sleep(0.1)

        # Record 200 generated tokens via in-process observer hook
        collector.record_tokens(200)
        collector.sample_once()

        self.assertGreater(collector._cumulative_energy_j, 0.0)
        self.assertEqual(collector._cumulative_tokens, 200.0)

        j_per_tok = JOULES_PER_TOKEN._value.get()
        tok_per_w = TOKENS_PER_WATT._value.get()
        cum_j_per_tok = CUMULATIVE_JOULES_PER_TOKEN._value.get()

        self.assertGreater(j_per_tok, 0.0)
        self.assertAlmostEqual(j_per_tok, cum_j_per_tok, places=6)
        self.assertAlmostEqual(tok_per_w, 1.0 / j_per_tok, places=6)

    @patch("urllib.request.urlopen")
    def test_vllm_metrics_scraping(self, mock_urlopen):
        sample_body = (
            '# HELP vllm:generation_tokens_total Number of generation tokens processed.\n'
            '# TYPE vllm:generation_tokens_total counter\n'
            'vllm:generation_tokens_total{model_name="meta-llama/Llama-3-70B"} 1500.0\n'
        ).encode("utf-8")

        mock_urlopen.return_value.__enter__.return_value.read.return_value = sample_body
        tokens = fetch_vllm_generation_tokens(["http://127.0.0.1:8000/metrics"])
        self.assertEqual(tokens, 1500.0)


if __name__ == "__main__":
    unittest.main()
