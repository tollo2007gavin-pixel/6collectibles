import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import card_runner as runner


def combined(left=50., top=50., resolved=True, status="Measured", allowance=0.):
    return {"status": status, "axes": {a: {"resolved": resolved, "percentage": p} for a, p in (("lr", left), ("tb", top))},
            "measurement_policy": {"axis_allowance_pp": {"lr": allowance, "tb": allowance}}}


def record(**kwargs):
    return {"item_id": "v1|123|0", "card_id": "22", "title": "Test card", "url": "https://www.ebay.com/itm/123",
            "image_url": "https://example.test/front.jpg", "combined": combined(**kwargs)}


PACKET = {"ready": True, "card_id": "123", "context": "context", "sheet": "sheet", "tips": "tips"}


def fake_front(raw, metadata, audit):
    # Picklable for the real multi-process scheduling test.
    return dict(metadata, combined=combined(), timings={"centering_seconds": 0.01})


class RunnerTests(unittest.TestCase):
    def test_gate_inclusive_and_both_axes_required(self):
        s = runner.Settings()
        self.assertTrue(runner.centering_gate(combined(47, 53), s)["accepted"])
        for c in (combined(46.99, 50), combined(50, 53.01), combined(resolved=False), combined(status="Fail")):
            self.assertFalse(runner.centering_gate(c, s)["accepted"])

    def test_strict_gate_disables_angular_overlap(self):
        c = combined(46.5, 50, allowance=1)
        self.assertTrue(runner.centering_gate(c, runner.Settings())["accepted"])
        self.assertFalse(runner.centering_gate(c, runner.Settings(allow_angular_variance=False))["accepted"])

    def finish(self, row, s, preparer=None, classifier=None, state=None):
        preparer = preparer or Mock(return_value=PACKET)
        classifier = classifier or Mock(return_value={"decision": "clean", "stages": []})
        state = state or runner.new_state(s)
        save = Mock()
        with patch.object(runner.luna, "references", return_value=[]):
            runner.finish_listing(row, {}, None, Path("."), s, [], state, save, preparer=preparer, classifier=classifier)
        return preparer, classifier, state, save

    def test_both_never_calls_whitening_for_centering_failure(self):
        for row in (record(left=55), record(resolved=False), record(status="Fail")):
            p, c, _, _ = self.finish(row, runner.Settings())
            p.assert_not_called()
            c.assert_not_called()
            self.assertFalse(row["accepted"])

    def test_centering_only_never_prepares_or_calls_api(self):
        row = record()
        p, c, _, _ = self.finish(row, runner.Settings(pipeline="centering"))
        self.assertTrue(row["accepted"])
        p.assert_not_called()
        c.assert_not_called()

    def test_api_only_does_not_require_or_run_centering(self):
        row = record()
        del row["combined"]
        _, c, state, save = self.finish(row, runner.Settings(pipeline="api"))
        self.assertTrue(row["accepted"])
        c.assert_called_once()
        save.assert_called_once()
        self.assertEqual(state["whitening_attempts"], 1)
        self.assertNotIn("centering_gate", row)

    def test_invalid_or_uncertain_whitening_never_passes(self):
        for verdict in ({"decision": "not_clean"}, {"decision": "needs_review"}, {"decision": "clean", "stages": [{"invalid": True}]}):
            row = record()
            _, c, _, _ = self.finish(row, runner.Settings(), classifier=Mock(return_value=verdict))
            self.assertFalse(row["accepted"])
            c.assert_called_once()

    def test_missing_back_and_dry_run_do_not_spend(self):
        for settings, packet in ((runner.Settings(), {"ready": False, "reason": "no_validated_back_photo"}),
                                 (runner.Settings(prepare_api_only=True), PACKET)):
            row = record()
            _, c, state, _ = self.finish(row, settings, preparer=Mock(return_value=packet))
            c.assert_not_called()
            self.assertFalse(row["accepted"])
            self.assertEqual(state["whitening_attempts"], 0)

    def test_api_cap_checked_before_preparation(self):
        s = runner.Settings(max_whitening_cards=1)
        state = runner.new_state(s)
        state["whitening_attempts"] = 1
        row = record()
        p, c, _, _ = self.finish(row, s, state=state)
        p.assert_not_called()
        c.assert_not_called()
        self.assertEqual(row["reason"], "whitening_limit_reached")

    def test_api_attempt_checkpoint_precedes_paid_call(self):
        s = runner.Settings()
        state = runner.new_state(s)
        row = record()
        events = []
        def save():
            events.append((state["whitening_attempts"], row["reason"]))
        def classify(*args):
            self.assertEqual(events, [(1, "api_in_flight")])
            return {"decision": "clean", "stages": []}
        with patch.object(runner.luna, "references", return_value=[]):
            runner.finish_listing(row, {}, None, Path("."), s, [], state, save, preparer=Mock(return_value=PACKET), classifier=classify)

    def test_api_error_stops_and_does_not_pass(self):
        row = record()
        with self.assertRaisesRegex(RuntimeError, "Whitening API failed"):
            self.finish(row, runner.Settings(), classifier=Mock(return_value={"decision": "needs_review", "error": "account unavailable"}))
        self.assertFalse(row["accepted"])

    def test_reference_hashes_are_valid(self):
        self.assertEqual(len(runner.load_bank("luna_references.json")), 16)

    def test_validation_rejects_bad_controls(self):
        for changes in ({"pipeline": "typo"}, {"max_listings": 0}, {"max_back_photos": 100},
                        {"max_listing_age_hours": -1}, {"centering_max_side_pct": 49}, {"max_whitening_cards": -1},
                        {"allow_angular_variance": "false"}, {"target_passes": 50}, {"run_label": "../../escape"},
                        {"centering_max_side_pct": "53"}, {"max_listing_age_hours": "24"}, {"resume_run": True}):
            with self.assertRaises(ValueError):
                runner.validate(replace(runner.Settings(), **changes))

    def test_selected_csv_cards_are_active(self):
        self.assertEqual([c["id"] for c in runner.selected_cards(runner.Settings(card_ids=["22"]))], ["22"])
        with self.assertRaisesRegex(ValueError, "not active"):
            runner.selected_cards(runner.Settings(card_ids=["1"]))

    def test_overrides_are_typed_and_unknown_keys_rejected(self):
        s = runner.apply_overrides(runner.Settings(), ['card_ids=["22"]', 'allow_angular_variance=false', 'max_listing_age_hours=null', 'pipeline=api'])
        self.assertEqual(s.card_ids, ["22"])
        self.assertFalse(s.allow_angular_variance)
        self.assertIsNone(s.max_listing_age_hours)
        self.assertEqual(s.pipeline, "api")
        with self.assertRaises(ValueError):
            runner.apply_overrides(s, ["unknown=2"])

    def test_github_boolean_and_number_inputs(self):
        with patch.dict(os.environ, {"CARD_RUN_INPUTS": json.dumps({"pipeline": "centering", "allow_angular_variance": False, "max_listings": 2})}):
            with patch.object(runner, "run") as execute:
                with patch.dict(os.environ, {"EBAY_CLIENT_ID": "fake", "EBAY_CLIENT_SECRET": "fake"}):
                    self.assertEqual(runner.main(runner.Settings(), []), 0)
            s = execute.call_args.args[0]
            self.assertIs(s.allow_angular_variance, False)
            self.assertEqual(s.max_listings, 2)

    def test_check_only_does_not_run_pipelines(self):
        with patch.dict(os.environ, {"CARD_RUN_INPUTS": ""}):
            with patch.object(runner, "run") as execute:
                self.assertEqual(runner.main(runner.Settings(check_only=True, pipeline="centering"), []), 0)
                execute.assert_not_called()

    def test_filter_forwarding_condition_dedup_and_per_card_cap(self):
        s = runner.Settings(max_listing_age_hours=24, max_listings_per_card=1)
        client = Mock()
        client.search.return_value = {"itemSummaries": [{"itemId": "used"}, {"itemId": "bad"}, {"itemId": "good"}, {"itemId": "later"}]}
        nm = {"conditionId": "4000", "conditionDescriptors": [{"name": "Card Condition", "values": [{"content": "Near Mint or Better"}]}],
              "image": {"imageUrl": "https://example.test/front"}, "title": "good"}
        client.details.side_effect = [{"conditionId": "2750"}, nm]
        state = runner.new_state(s)
        rows = list(runner.ebay_candidates(s, client, [{"id": "39", "search_query": "ditto"}], {"used"}, state))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0]["item_id"], "good")
        self.assertEqual(client.search.call_args.kwargs["age_hours"], 24)
        self.assertEqual(client.search.call_args.kwargs["card_id"], "39")
        self.assertEqual(state["fetch_counts"]["not_near_mint_ungraded"], 1)

    def test_reports_filter_production_and_escape_listing_text(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "_audit").mkdir()
            s = runner.Settings(output_mode="production")
            state = runner.new_state(s)
            passed = dict(record(), status="pass", accepted=True, reason="clean_back", whitening={"decision": "clean"})
            failed = dict(record(), title="REJECTED_SENTINEL", status="reject", accepted=False, reason="outside_centering_gate")
            passed["title"] = "<script>alert('x')</script>"
            passed["url"] = "javascript:alert(1)"
            state["records"] = [passed, failed]
            runner.checkpoint(out, state, s)
            page = (out / "review.html").read_text(encoding="utf-8")
            self.assertNotIn("REJECTED_SENTINEL", page)
            self.assertNotIn("<script>", page)
            self.assertNotIn("javascript:", page)
            self.assertFalse((out / "results.csv").exists())
            self.assertEqual(len(json.loads((out / "_audit/run.json").read_text())["records"]), 2)
            runner.checkpoint(out, state, replace(s, output_mode="testing"))
            self.assertIn("REJECTED_SENTINEL", (out / "review.html").read_text(encoding="utf-8"))

    def test_csv_titles_are_not_formulas(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "results.csv"
            runner.write_csv(dest, [{"title": "=DANGEROUS()", "left_pct": 51.5}])
            with dest.open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["title"], "'=DANGEROUS()")

    def test_full_run_parallel_local_cap_and_resume_without_reprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            uploads = base / "uploads"
            for n in range(4):
                folder = uploads / f"listing_{n}"
                folder.mkdir(parents=True)
                (folder / "front.jpg").write_bytes(b"fake-only-for-mocked-processor")
            s = runner.Settings(source="local", pipeline="centering", max_listings=3, target_passes=2, centering_workers=2,
                                local_input_dir=str(uploads), output_root=str(base / "out"))
            out = runner.run(s, processor=fake_front)
            state = json.loads((out / "_audit/run.json").read_text())
            self.assertEqual(len(state["records"]), 2)
            self.assertEqual(state["state"], "target_reached")
            s = replace(s, resume_run=str(out), target_passes=3, max_listings=4)
            runner.run(s, processor=fake_front)
            state = json.loads((out / "_audit/run.json").read_text())
            self.assertEqual(len(state["records"]), 3)
            self.assertEqual(len({r["item_id"] for r in state["records"]}), 3)
            with self.assertRaisesRegex(ValueError, "Resume settings differ"):
                runner.run(replace(s, centering_max_side_pct=52), processor=fake_front)

    def test_local_api_only_back_and_paid_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for n in range(3):
                folder = base / "uploads" / str(n)
                folder.mkdir(parents=True)
                (folder / "back.jpg").write_bytes(b"mock-back")
            s = runner.Settings(source="local", pipeline="api", max_listings=3, max_whitening_cards=1,
                                local_input_dir=str(base / "uploads"), output_root=str(base / "out"))
            classifier = Mock(return_value={"decision": "not_clean", "stages": []})
            preparer = Mock(return_value=PACKET)
            # No actual image paths are emitted by this mocked packet.
            s.output_mode = "production"
            with patch.object(runner.luna, "references", return_value=[]):
                out = runner.run(s, preparer=preparer, classifier=classifier)
            classifier.assert_called_once()
            preparer.assert_called_once()
            state = json.loads((out / "_audit/run.json").read_text())
            self.assertEqual([r["reason"] for r in state["records"]], ["whitening_not_clean", "whitening_limit_reached", "whitening_limit_reached"])
            state["records"][0]["reason"] = "api_in_flight"
            (out / "_audit/run.json").write_text(json.dumps(state))
            classifier.reset_mock()
            with patch.object(runner.luna, "references", return_value=[]):
                runner.run(replace(s, resume_run=str(out)), preparer=preparer, classifier=classifier)
            classifier.assert_not_called()
            resumed = json.loads((out / "_audit/run.json").read_text())
            self.assertEqual(resumed["records"][0]["reason"], "interrupted_review_required")
            self.assertEqual(resumed["whitening_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
