import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests
from scripts import fetch_ferry_data as fetch
from scripts import predict_tomorrow as predict

NOW = pd.Timestamp("2026-09-29 17:00:00")
HEADER = "Timestamp,Redemption Count,Sales Count\n"
OLD = HEADER + "2015-05-01 13:30:00,1,1\n2026-08-22 10:45:00,2,2\n"
FRESH = HEADER + "2015-05-01 13:30:00,1,1\n2026-09-29 16:45:00,3,3\n"
PACKAGE = {"success": True, "result": {"resources": [
    {"id": "ferry", "datastore_active": True},
    {"datastore_active": False, "format": "CSV", "url": "https://example.test/ferry.csv"},
]}}
PROBE = {"result": {"records": [{"Timestamp": "2026-09-29T16:45:00"}]}}


def response(payload=None, text=None):
    return Mock(json=Mock(return_value=payload), text=text)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ferry.csv"
        self.path.write_text(OLD)

    def run_fetch(self, sources):
        with patch.object(fetch, "get_fresh", side_effect=[response(PACKAGE), response(PROBE), *sources]) as get:
            result = fetch.fetch_data(self.path, NOW)
            return result, get

    def test_stale_dump_falls_back_to_fresh_csv(self):
        df, get = self.run_fetch([response(text=OLD), response(text=FRESH)])
        self.assertEqual(df.Timestamp.max(), pd.Timestamp("2026-09-29 16:45:00"))
        self.assertEqual(get.call_args.args[0], "https://example.test/ferry.csv")
        self.assertIn("2026-09-29", self.path.read_text())

    def test_all_stale_sources_fail_without_overwriting(self):
        with self.assertRaisesRegex(RuntimeError, "preserved existing data"):
            self.run_fetch([response(text=OLD), response(text=OLD)])
        self.assertEqual(self.path.read_text(), OLD)

    def test_http_error_falls_back(self):
        self.run_fetch([requests.HTTPError("502"), response(text=FRESH)])
        self.assertIn("2026-09-29", self.path.read_text())

    def test_fresh_primary_does_not_download_fallback(self):
        _, get = self.run_fetch([response(text=FRESH)])
        self.assertEqual(get.call_count, 3)

    def test_recent_but_behind_live_source_is_rejected(self):
        behind = FRESH.replace("16:45:00", "15:45:00")
        with self.assertRaisesRegex(ValueError, "live datastore"):
            fetch.validate_csv(behind, NOW, pd.Timestamp("2026-09-29 16:45:00"))

    def test_truncated_history_is_rejected(self):
        previous = pd.Series(pd.to_datetime(["2015-05-01", "2026-09-28"]))
        with self.assertRaisesRegex(ValueError, "discard existing"):
            fetch.validate_csv(HEADER + "2026-09-29 16:45:00,3,3\n", NOW, previous=previous)

    def test_invalid_csv_is_rejected(self):
        for data in ["<html>Error</html>", HEADER, FRESH.replace(",3,3", ",-3,3")]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                fetch.validate_csv(data, NOW)

    def test_request_checks_http_and_bypasses_cache(self):
        with patch.object(fetch.requests, "get", return_value=Mock()) as get:
            fetch.get_fresh("https://example.test")
            first_key = get.call_args.kwargs["params"]["_refresh"]
            fetch.get_fresh("https://example.test")
            self.assertNotEqual(first_key, get.call_args.kwargs["params"]["_refresh"])
            self.assertEqual(get.return_value.raise_for_status.call_count, 2)

    def test_live_probe_uses_post_without_unknown_query_fields(self):
        with patch.object(fetch.requests, "post", return_value=Mock()) as post:
            fetch.get_fresh(fetch.BASE_URL + "/api/3/action/datastore_search", resource_id="ferry", limit=1)
            self.assertEqual(post.call_args.kwargs["json"], {"resource_id": "ferry", "limit": 1})
            post.return_value.raise_for_status.assert_called_once()


class ForecastTests(unittest.TestCase):
    def test_stale_data_replaces_old_forecast_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily, hourly, output = [root / name for name in ["daily.csv", "hourly.csv", "prediction.json"]]
            daily.write_text("date,redemptions\n2026-08-22,1777\n")
            hourly.write_text("hour,redemptions\n2026-08-22 10:00:00,10\n")
            output.write_text('{"prediction_date":"2026-09-05"}')
            with patch.multiple(predict, DAILY_CSV=daily, HOURLY_CSV=hourly, OUTPUT_JSON=output), \
                    patch.object(predict, "train") as train, patch.object(predict, "datetime") as dt:
                dt.now.return_value = NOW.to_pydatetime()
                predict.main()
            train.assert_not_called()
            self.assertIn('"status": "unavailable"', output.read_text())
            self.assertIn("2026-09-30", output.read_text())


if __name__ == "__main__":
    unittest.main()
