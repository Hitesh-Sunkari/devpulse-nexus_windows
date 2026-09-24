"""Run and document deterministic guardrail effectiveness tests."""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.guardrails import evaluate_input


BASE = Path("evaluation")
CASES = json.loads((BASE / "guardrail_cases.json").read_text(encoding="utf-8"))
RESULT_JSON = BASE / "guardrail_test_report.json"
RESULT_MD = BASE / "guardrail_test_report.md"


def main():
    rows = []
    for case in CASES:
        before = evaluate_input(case["question"], enforce=False).to_dict()
        after = evaluate_input(case["question"]).to_dict()
        passed = (
            after["allowed"] == case["expected_allowed"]
            and after["code"] == case["expected_code"]
        )
        rows.append({
            "id": case["id"],
            "without_guardrail": before,
            "with_guardrail": after,
            "passed": passed,
        })

    report = {
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "pass_rate_percent": round(100 * sum(row["passed"] for row in rows) / len(rows), 1),
        "results": rows,
    }
    RESULT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# DevPulse Nexus Guardrail Test Report", "", "| Case | Without guardrail | With guardrail | Pass |", "|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['without_guardrail']['code']} | {row['with_guardrail']['code']} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(["", f"Pass rate: **{report['pass_rate_percent']}%** ({report['passed']}/{report['cases']})"])
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
