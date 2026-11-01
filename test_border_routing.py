from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import border_routing as routing
import card_runner as runner
from gx_border import adapter


class BorderRoutingTests(unittest.TestCase):
    def test_modes_use_disjoint_approved_lists(self):
        yellow = runner.selected_cards(runner.Settings(border_type="yellow"))
        gx = runner.selected_cards(runner.Settings(border_type="gx"))
        all_cards = runner.selected_cards(runner.Settings(border_type="all"))
        self.assertEqual((len(yellow), len(gx), len(all_cards)), (15, 9, 24))
        self.assertTrue(all(c["border_type"] == "yellow" for c in yellow))
        self.assertTrue(all(c["border_type"] == "gx" for c in gx))
        self.assertFalse({c["id"] for c in yellow} & {c["id"] for c in gx})
        self.assertNotIn("zekrom_sm138", {c["id"] for c in all_cards})

    def test_unapproved_and_cross_family_ids_fail_closed(self):
        for card, family in (("22", "gx"), ("eevee_sm233", "yellow"), ("zekrom_sm138", "gx"), ("gx1", "gx")):
            with self.assertRaises(ValueError):
                routing.approved(card, family)
        with self.assertRaises(ValueError):
            runner.selected_cards(runner.Settings(border_type="gx", card_ids=["22"]))

    def test_csv_cannot_change_a_cards_fitter_or_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cards.csv"
            for text, s in (("id,search_query,active\n22,Charizard GX SM195,1\n", runner.Settings()),
                            ("id,search_query,active\n22,Charizard TG03/TG30,1\n", runner.Settings(border_type="gx"))):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    runner.selected_cards(replace(s, cards_csv=str(path), gx_cards_csv=str(path)))

    def test_titles_must_match_exact_approved_design(self):
        gx = routing.approved("pikachu_sm232")
        yellow = routing.approved("20")
        self.assertTrue(routing.listing_eligible("Pokemon Pikachu-GX SM232 Black Star Promo NM", gx))
        self.assertTrue(routing.listing_eligible("Pikachu GX SM 232 NM", gx))
        self.assertTrue(routing.listing_eligible("Pikachu GG30/GG70 Crown Zenith", yellow))
        self.assertTrue(routing.listing_eligible("Charizard TG3/TG30", routing.approved("22")))
        for title in ("Pikachu GG30/GG70", "Pikachu GX SM233", "Pikachu GX", "Pikachu V SM232", "Raichu GX SM232",
                      "Pikachu GX SM232 + Eevee GX SM233", "Pikachu GX SM232 custom", "Pikachu GX SM232 lot", "Pikachu GX SM232 SM999"):
            self.assertFalse(routing.listing_eligible(title, gx), title)
        self.assertFalse(routing.listing_eligible("Pikachu GX SM232", yellow))
        self.assertFalse(routing.listing_eligible("Pikachu GG30/GG70 GX", yellow))

    def test_worker_dispatches_once_to_only_approved_fitter(self):
        with patch.object(runner.yellow, "process_front", return_value={}) as yellow, patch.object(adapter, "process_front", return_value={}) as gx:
            runner.front_worker(b"", {"card_id": "22", "border_type": "yellow"}, ".")
            yellow.assert_called_once()
            gx.assert_not_called()
            yellow.reset_mock()
            runner.front_worker(b"", {"card_id": "eevee_sm233", "border_type": "gx"}, ".")
            yellow.assert_not_called()
            gx.assert_called_once()
            gx.reset_mock()
            with self.assertRaises(ValueError):
                runner.front_worker(b"", {"card_id": "22", "border_type": "gx"}, ".")
            yellow.assert_not_called()
            gx.assert_not_called()

    def test_rejected_gx_never_falls_back_to_yellow(self):
        with patch.object(adapter, "process_front", side_effect=ValueError("unresolved GX")), patch.object(runner.yellow, "process_front") as yellow:
            with self.assertRaises(ValueError):
                runner.front_worker(b"", {"card_id": "eevee_sm233", "border_type": "gx"}, ".")
            yellow.assert_not_called()

    def test_fetch_skips_crossovers_before_images_and_keeps_yellow_filters_off_gx(self):
        s = runner.Settings(border_type="gx", card_ids=["pikachu_sm232"])
        state = runner.new_state(s)
        client = Mock()
        client.search.return_value = {"itemSummaries": [{"itemId": "wrong"}, {"itemId": "good"}]}
        def detail(title):
            return {"title": title, "conditionId": "4000", "conditionDescriptors": [{"name": "Card Condition", "values": [{"content": "Near Mint or Better"}]}],
                    "image": {"imageUrl": "https://example.test/front.jpg"}}
        client.details.side_effect = [detail("Pikachu GG30/GG70"), detail("Pikachu GX SM232")]
        rows = list(runner.ebay_candidates(s, client, runner.selected_cards(s), set(), state))
        self.assertEqual([row[0]["item_id"] for row in rows], ["good"])
        self.assertEqual(rows[0][0]["border_type"], "gx")
        self.assertIsNone(client.search.call_args.kwargs["card_id"])
        client.download.assert_not_called()
        self.assertEqual(state["fetch_counts"]["identity_not_approved"], 1)

    def test_api_only_also_refuses_invalid_routes(self):
        p, c = Mock(), Mock()
        s = runner.Settings(pipeline="api")
        with self.assertRaises(ValueError):
            runner.finish_listing({"card_id": "22", "border_type": "gx"}, {}, None, Path("."), s, [], runner.new_state(s), Mock(), preparer=p, classifier=c)
        p.assert_not_called()
        c.assert_not_called()

    def test_local_mixed_inputs_route_by_id_and_reject_conflicting_tags(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for number, card in enumerate(("22", "eevee_sm233")):
                folder = base / str(number)
                folder.mkdir()
                (folder / "front.jpg").write_bytes(b"dummy")
                (folder / "metadata.json").write_text(json.dumps({"card_id": card}))
            s = runner.Settings(source="local", border_type="all", local_input_dir=str(base))
            rows = list(runner.local_candidates(s, set(), runner.new_state(s)))
            self.assertEqual([row[0]["border_type"] for row in rows], ["yellow", "gx"])
            rows = list(runner.local_candidates(replace(s, border_type="yellow"), set(), runner.new_state(s)))
            self.assertEqual(len(rows), 1)
            (base / "1/metadata.json").write_text(json.dumps({"card_id": "eevee_sm233", "border_type": "yellow"}))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                list(runner.local_candidates(s, set(), runner.new_state(s)))

    def test_gx_reportability_and_no_yellow_allowance(self):
        result = {"status": "complete", "axes": {"lr": {"measured": True, "first_pct": 53.5}, "tb": {"measured": True, "first_pct": 50}}}
        mapped = adapter.to_combined(result)
        self.assertFalse(runner.centering_gate(mapped, runner.Settings())["accepted"])
        self.assertEqual(mapped["measurement_policy"]["axis_allowance_pp"], {"lr": 0, "tb": 0})
        for status in ("hard_failure", "unresolved"):
            mapped = adapter.to_combined(dict(result, status=status))
            self.assertTrue(all(v["percentage"] is None for v in mapped["axes"].values()))
        mapped = adapter.to_combined(dict(result, reportability={"blocked_axes": ["tb"]}))
        self.assertTrue(mapped["axes"]["lr"]["resolved"])
        self.assertIsNone(mapped["axes"]["tb"]["percentage"])

    def test_gx_partial_cannot_pass_or_reach_whitening(self):
        result = {"status": "partial", "axes": {"lr": {"measured": True, "first_pct": 50}, "tb": {"measured": False}}}
        row = {"card_id": "eevee_sm233", "border_type": "gx", "combined": adapter.to_combined(result)}
        s = runner.Settings(border_type="gx")
        p, c = Mock(), Mock()
        runner.finish_listing(row, {}, None, Path("."), s, [], runner.new_state(s), Mock(), preparer=p, classifier=c)
        self.assertFalse(row["accepted"])
        p.assert_not_called()
        c.assert_not_called()

    def test_packaged_engine_matches_installed_release_manifest(self):
        root = Path(adapter.__file__).parent
        hashes = json.loads((root / "source_manifest.json").read_text())["files"]
        self.assertEqual(len(hashes), 38)
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((root / "gx_guarded_engine" / name).read_bytes()).hexdigest(), expected, name)

    def test_adapter_saves_supported_lines_and_masks_unresolved_axis(self):
        result = {"status": "partial", "axes": {"lr": {"measured": True, "first_pct": 51}, "tb": {"measured": False}},
                  "geometry": {"outer": [[10, 10], [390, 10], [390, 490], [10, 490]], "inner": [[25, 25], [375, 25], [375, 475], [25, 475]]}}
        buf = io.BytesIO()
        Image.fromarray(np.full((500, 400, 3), 140, np.uint8)).save(buf, "PNG")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("downloads", "Combined", "details"):
                (root / name).mkdir()
            with patch.object(adapter, "analyze", return_value=result):
                row = adapter.process_front(buf.getvalue(), {"card_id": "eevee_sm233", "border_type": "gx", "item_id": "sample"}, root)
            self.assertEqual(row["combined"]["axes"]["lr"]["percentage"], 51)
            self.assertIsNone(row["combined"]["axes"]["tb"]["percentage"])
            self.assertTrue((root / row["overlays"]["combined"]).is_file())


if __name__ == "__main__":
    unittest.main()
