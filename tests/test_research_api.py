"""Local HTTP API tests: load saved model only, never refit on requests."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bookingforge.api import create_app
from bookingforge.model_artifact import export_arrival_models, load_portable
from bookingforge.train_baseline import FEATURES


class ResearchApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.dataset_path = cls.directory / "sample.csv"
        cls.artifact_dir = cls.directory / "models"
        def row(year, month, day, outcome_date, target, hotel="City Hotel"):
            record = {name: "0" for name in FEATURES}
            record.update(
                hotel=hotel, lead_time="10",
                arrival_date_year=year, arrival_date_month=month,
                arrival_date_day_of_month=day, reserved_room_type="A",
                stays_in_week_nights="2", stays_in_weekend_nights="1",
                adults="2", reservation_status_date=outcome_date,
                is_canceled=target,
            )
            return record
        records = [
            row("2016", "July", "1", "2016-07-02", "0"),
            row("2016", "July", "2", "2016-07-03", "1"),
            row("2016", "July", "3", "2016-07-04", "0", "Resort Hotel"),
            row("2016", "July", "4", "2016-07-05", "1", "Resort Hotel"),
            row("2016", "October", "11", "2016-10-12", "0"),
            row("2016", "October", "12", "2016-10-13", "1"),
            row("2016", "November", "11", "2017-01-03", "0", "Resort Hotel"),
            row("2016", "November", "12", "2017-01-03", "1", "Resort Hotel"),
            row("2017", "January", "3", "invalid-status", "invalid-target"),
        ]
        with cls.dataset_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=FEATURES + ["reservation_status_date", "is_canceled"]
            )
            writer.writeheader()
            writer.writerows(records)
        cls.export_report = export_arrival_models(
            cls.dataset_path, cls.artifact_dir,
            variant="both", epochs=2,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def valid_request(self, variant="without_year"):
        return {
            "variant": variant,
            "booking": {
                "hotel": "City Hotel",
                "lead_time": 10,
                "arrival_date_year": 2016,
                "arrival_date_month": "October",
                "arrival_date_day_of_month": 11,
                "stays_in_weekend_nights": 1,
                "stays_in_week_nights": 2,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "reserved_room_type": "A",
            },
        }

    def test_health_and_models_expose_research_scope(self):
        with TestClient(create_app(self.artifact_dir)) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["loaded_variants"], ["with_year", "without_year"])
            models = client.get("/v1/research/models")
            self.assertEqual(models.status_code, 200)
            self.assertTrue(models.json()["research_only"])
            self.assertEqual(
                models.json()["models"]["without_year"]["training_rows"], 4
            )
            self.assertFalse(
                models.json()["models"]["without_year"]["arrival_year_used_by_model"]
            )

    def test_both_endpoint_predictions_match_loaded_artifacts(self):
        with TestClient(create_app(self.artifact_dir)) as client:
            for variant in ("with_year", "without_year"):
                payload = self.valid_request(variant)
                result = client.post("/v1/research/predict", json=payload)
                self.assertEqual(result.status_code, 200, result.text)
                data = result.json()
                self.assertTrue(data["research_only"])
                self.assertFalse(data["calibrated_for_booking_time"])
                self.assertEqual(data["variant"], variant)
                self.assertEqual(data["arrival_year_used_by_model"], variant == "with_year")
                self.assertIn("not", data["warning"].lower())
                expected = load_portable(
                    self.artifact_dir, "arrival_mlp_" + variant
                ).predict_proba_positive([payload["booking"]])[0]
                self.assertAlmostEqual(data["research_cancellation_score"], expected, places=12)

    def test_outcome_fields_and_missing_features_rejected(self):
        with TestClient(create_app(self.artifact_dir)) as client:
            for extra in ("is_canceled", "reservation_status_date", "reservation_status"):
                req = self.valid_request()
                req["booking"][extra] = "1"
                self.assertEqual(
                    client.post("/v1/research/predict", json=req).status_code,
                    422,
                )
            req = self.valid_request()
            del req["booking"]["lead_time"]
            self.assertEqual(client.post("/v1/research/predict", json=req).status_code, 422)
            req = self.valid_request()
            req["variant"] = "not_a_model"
            self.assertEqual(client.post("/v1/research/predict", json=req).status_code, 422)

    def test_bad_calendar_dates_and_futuristic_inputs_rejected(self):
        with TestClient(create_app(self.artifact_dir)) as client:
            for year, month, day in (
                (2017, "January", 1),
                (2016, "February", 30),
                (2015, "January", 1),
            ):
                req = self.valid_request()
                req["booking"]["arrival_date_year"] = year
                req["booking"]["arrival_date_month"] = month
                req["booking"]["arrival_date_day_of_month"] = day
                self.assertEqual(client.post("/v1/research/predict", json=req).status_code, 422)
            req = self.valid_request()
            req["booking"]["lead_time"] = -10
            self.assertEqual(client.post("/v1/research/predict", json=req).status_code, 422)

    def test_server_fails_closed_without_saved_models(self):
        with tempfile.TemporaryDirectory() as missing:
            with self.assertRaises(FileNotFoundError):
                with TestClient(create_app(missing)):
                    pass


if __name__ == "__main__":
    unittest.main()
