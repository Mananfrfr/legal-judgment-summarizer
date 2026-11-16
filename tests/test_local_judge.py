import io
import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from local_judge import JudgeError, judge_summary, verified_finding
from review_server import create_app


class CitationTests(unittest.TestCase):
    def test_only_server_supplied_evidence_ids_produce_a_citation(self):
        passages = {"S1": (3, 87, "pay within six weeks.")}
        answer = {"verdict": "supported", "evidence_id": "S1"}
        self.assertEqual(verified_finding(answer, passages, "decision", "pay within six weeks.")["citation"]["start"], 87)
        self.assertEqual(verified_finding(answer, passages, "decision", "pay within ten weeks.")["verdict"], "insufficient_evidence")
        self.assertEqual(verified_finding(answer, {"S1": (1, 0, "The trial court dismissed the appeal.")},
                                         "decision", "The trial court dismissed the appeal.")["verdict"], "insufficient_evidence")
        for change in ({"evidence_id": "S999"}, {"evidence_id": []}, {"evidence_id": "none"}):
            self.assertEqual(verified_finding({**answer, **change}, passages, "decision", "pay within six weeks.")["verdict"], "insufficient_evidence")
        with self.assertRaises(JudgeError):
            verified_finding({"verdict": "accurate"}, passages, "decision", "pay within six weeks.")

    def test_report_covers_at_most_one_complete_extract_per_section(self):
        page = "The earlier court dismissed the appeal. We allow the appeal."
        entries = [{"id": i, "sentence": sentence, "fragments": [{"page": 1, "start": page.index(sentence),
                    "end": page.index(sentence) + len(sentence)}]} for i, sentence in enumerate(["The earlier court dismissed the appeal.", "We allow the appeal."])]
        data = {"sections": {"decision": entries, "reasoning": [], "background": [], "issue": []},
                "source_sha256": "abc", "canonical_pages": [page]}
        infer = Mock(return_value={"verdict": "supported", "evidence_id": "S2"})
        report = judge_summary(data, infer)
        self.assertEqual(report["total_extracts"], 2)
        self.assertEqual(report["findings"][0]["extract_id"], 1)
        self.assertEqual(report["findings"][0]["citation"]["text"], entries[-1]["sentence"])
        self.assertEqual(infer.call_count, 1)


class ReviewFlowTests(unittest.TestCase):
    def test_unavailable_retry_cache_ownership_and_export(self):
        runner = Mock(side_effect=[JudgeError("Start the local judge"), {"findings": [], "total_extracts": 0}])
        with tempfile.TemporaryDirectory() as root:
            app = create_app(root, password="", judge_runner=runner)
            with TestClient(app) as client:
                headers = {"x-review-request": "1"}
                upload = client.post("/api/jobs", headers=headers, data={"evidence_only": "true"},
                                     files={"files": ("case.txt", b"We allow the appeal. Payment is due within six weeks.")})
                key = upload.json()["id"]
                job = f"/api/jobs/{key}"
                for _ in range(200):
                    if client.get(job).json()["status"] == "ready":
                        break
                    time.sleep(0.02)
                url = job + "/documents/0"
                original = client.get(url).json()["result"]
                with TestClient(app, base_url="http://testserver") as other:
                    self.assertEqual(other.post(url + "/judge", headers=headers).status_code, 404)
                    self.assertEqual(client.post(url + "/judge").status_code, 403)
                    self.assertEqual(client.post(url + "/judge", headers=headers).status_code, 503)
                    report = client.post(url + "/judge", headers=headers).json()
                    self.assertEqual(client.post(url + "/judge", headers=headers).json(), report)
                    self.assertEqual(runner.call_count, 2)
                    after = client.get(url).json()
                    self.assertEqual(after["judge"], report)
                    self.assertEqual(after["result"], original)
                    with zipfile.ZipFile(io.BytesIO(client.get(job + "/export").content)) as archive:
                        self.assertEqual(json.loads(archive.read("document_01/local_judge.json")), report)
