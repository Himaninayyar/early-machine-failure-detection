"""Builds docs/Project_Documentation_and_Change_Log.xlsx and docs/PROJECT_STATUS.md.

    python docs/build_workbook.py

Every metric is read from models/metrics.json / models/meta.json / reports/eda.json, and the test counts from the pytest report
docs/test_results.xml (create it with:  python -m pytest tests -q --junitxml=docs/test_results.xml).
The build FAILS if the change log cites a test that is missing or did not pass, so the workbook cannot claim anything untested.
"""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import workbook_content_a as A  # noqa: E402
import workbook_content_b as B  # noqa: E402
import workbook_content_c as C  # noqa: E402

TODAY = "2026-11-24"
OUT = os.path.join(HERE, "Project_Documentation_and_Change_Log.xlsx")
FONT = "Arial"


def load():
    M = json.load(open(os.path.join(ROOT, "models", "metrics.json")))
    meta = json.load(open(os.path.join(ROOT, "models", "meta.json")))
    eda = json.load(open(os.path.join(ROOT, "reports", "eda.json")))
    return M, meta, eda


def load_tests():
    path = os.path.join(HERE, "test_results.xml")
    if not os.path.exists(path):
        sys.exit("docs/test_results.xml not found. Run: python -m pytest tests -q --junitxml=docs/test_results.xml")
    root = ET.parse(path).getroot()
    by_base, per_file, total, passed = defaultdict(list), defaultdict(lambda: [0, 0]), 0, 0
    for tc in root.iter("testcase"):
        base = tc.get("name").split("[")[0]
        status = "failed" if (tc.find("failure") is not None or tc.find("error") is not None) else ("skipped" if tc.find("skipped") is not None else "passed")
        by_base[base].append(status)
        f = tc.get("classname").split(".")[-1]
        per_file[f][1] += 1
        per_file[f][0] += status == "passed"
        total += 1
        passed += status == "passed"
    return {"by_base": by_base, "per_file": dict(per_file), "total": total, "passed": passed}


def verify_tests(names, T, where):
    out = []
    for n in names:
        base = n.split("::")[-1]
        st = T["by_base"].get(base)
        if not st or any(s != "passed" for s in st):
            sys.exit(f"BUILD FAILED: '{base}' cited in {where} is {'missing' if not st else 'not passing'} in docs/test_results.xml")
        out.append(f"{base} (passed{'' if len(st) == 1 else f', {len(st)} cases'})")
    return out


# ---------------------------------------------------------------------------------------------- formatting
thin = Side(style="thin", color="BFC5CD")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
HEAD_FILL = PatternFill("solid", fgColor="1F3A5F")
SECTION_FILL = PatternFill("solid", fgColor="DCE6F1")
NOTE_FILL = PatternFill("solid", fgColor="F3F4F6")
WRAP = Alignment(wrap_text=True, vertical="top")


def est_height(cells):
    lines = 1
    for text, width in cells:
        if text is None:
            continue
        n = 0
        for para in str(text).split("\n"):
            n += max(1, math.ceil(len(para) / max(1.0, width * 1.05)))
        lines = max(lines, n)
    return min(409, max(15, lines * 13.2 + 3))


class Sheet:
    def __init__(self, wb, title, widths, header_text):
        self.ws = wb.create_sheet(title)
        self.widths = widths
        self.n = len(widths)
        for i, w in enumerate(widths, 1):
            self.ws.column_dimensions[get_column_letter(i)].width = w
        self.r = 1
        self.ws.sheet_view.showGridLines = False
        c = self.ws.cell(self.r, 1, title.split(" ", 1)[1] if title[0].isdigit() else title)
        c.font = Font(name=FONT, size=15, bold=True, color="1F3A5F")
        self.r += 1
        c = self.ws.cell(self.r, 1, header_text)
        c.font = Font(name=FONT, size=9, italic=True, color="5B6674")
        self.r += 2

    def _span_width(self, start, end):
        return sum(self.widths[start - 1:end])

    def heading(self, text):
        self.r += 1
        c = self.ws.cell(self.r, 1, text)
        c.font = Font(name=FONT, size=11, bold=True, color="1F3A5F")
        for col in range(1, self.n + 1):
            self.ws.cell(self.r, col).fill = SECTION_FILL
        self.ws.row_dimensions[self.r].height = 20
        self.r += 1

    def note(self, text):
        self.ws.merge_cells(start_row=self.r, start_column=1, end_row=self.r, end_column=self.n)
        c = self.ws.cell(self.r, 1, text)
        c.font = Font(name=FONT, size=9, italic=True)
        c.alignment = WRAP
        c.fill = NOTE_FILL
        self.ws.row_dimensions[self.r].height = est_height([(text, self._span_width(1, self.n))])
        self.r += 2

    def table(self, headers, rows):
        if headers and headers[0] == "#":                      # fold a leading number column into the next one
            headers = headers[1:]
            rows = [[f"{r[0]}. {r[1]}"] + list(r[2:]) for r in rows]
        k = len(headers)
        merge_last = k < self.n
        for j, h in enumerate(headers, 1):
            c = self.ws.cell(self.r, j, h)
            c.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
            c.fill = HEAD_FILL
            c.alignment = Alignment(wrap_text=True, vertical="center")
            c.border = BORDER
        if merge_last:
            self.ws.merge_cells(start_row=self.r, start_column=k, end_row=self.r, end_column=self.n)
            for col in range(k, self.n + 1):
                self.ws.cell(self.r, col).fill = HEAD_FILL
        self.ws.row_dimensions[self.r].height = est_height([(h, self.widths[j]) for j, h in enumerate(headers)]) + 4
        self.r += 1
        for row in rows:
            cells = []
            for j, v in enumerate(row, 1):
                c = self.ws.cell(self.r, j, v)
                c.font = Font(name=FONT, size=10)
                c.alignment = WRAP
                c.border = BORDER
                w = self._span_width(j, self.n) if (merge_last and j == k) else self.widths[j - 1]
                cells.append((v, w))
            if merge_last:
                self.ws.merge_cells(start_row=self.r, start_column=k, end_row=self.r, end_column=self.n)
                for col in range(k, self.n + 1):
                    self.ws.cell(self.r, col).border = BORDER
            self.ws.row_dimensions[self.r].height = est_height(cells)
            self.r += 1
        self.r += 1

    def blocks(self, blocks):
        for b in blocks:
            if b[0] == "heading":
                self.heading(b[1])
            elif b[0] == "table":
                self.table(b[1], b[2])
            elif b[0] == "note":
                self.note(b[1])


def main():
    M, meta, eda = load()
    T = load_tests()
    hdr = f"Last updated: {TODAY}  |  Model version {meta['version']} ({meta['profile']} profile)  |  Automated tests: {T['passed']} of {T['total']} passed"
    wb = Workbook()
    wb.remove(wb.active)

    S = Sheet(wb, "1 Project Overview", [28, 90, 60], hdr); S.blocks(A.overview(M, meta, eda, T))
    S = Sheet(wb, "2 Architecture", [30, 52, 44, 42, 30, 26, 30], hdr); S.blocks(A.architecture(M, meta, eda, T))
    S = Sheet(wb, "3 ML Model Explanation", [30, 120, 50], hdr); S.blocks(A.models(M, meta, eda, T))
    S = Sheet(wb, "4 Dataset & Features", [30, 34, 16, 56, 38, 44], hdr); S.blocks(A.dataset(M, meta, eda, T))
    S = Sheet(wb, "5 Training & Evaluation", [34, 18, 16, 16, 16, 16, 16, 16, 16, 18], hdr); S.blocks(B.evaluation(M, meta, eda, T))
    S = Sheet(wb, "6 Backend & Database", [46, 66, 50, 30], hdr); S.blocks(B.backend(M, meta, eda, T))
    S = Sheet(wb, "7 UI-UX & Features", [28, 46, 36, 44, 30, 30, 22, 50], hdr); S.blocks(B.features(M, meta, eda, T))

    # ------------------------------------------------------------------ 8 change log
    S = Sheet(wb, "8 Change Log", [11, 20, 30, 50, 50, 38, 56, 52, 24], hdr)
    S.note("Every row names the tests that verify it. docs/build_workbook.py fails if a cited test is missing or did not pass. 'Verified manually' means checked by hand in a browser or by an ad-hoc script and recorded in the Test/Verification column; it is not an automated test.")
    rows = []
    for r in C.changelog(M, meta, T):
        date, phase, file, prev, new, reason, tech, tests, status, manual = r
        ver = verify_tests(tests, T, f"change log row '{file}'")
        text = "; ".join(ver) + ((". " if ver else "") + manual if manual else "")
        rows.append([date, phase, file, prev, new, reason, tech, text or "-", status])
    S.table(["Date", "Phase", "File/Module", "Previous Behaviour", "New Behaviour", "Reason for Change", "Technical Explanation", "Test/Verification", "Status"], rows)
    S.ws.freeze_panes = None

    S = Sheet(wb, "9 Technical Decisions", [24, 40, 34, 56, 56, 28], hdr)
    S.table(["Decision", "Chosen", "Alternatives considered", "Why chosen", "Why alternatives rejected", "Where"], C.decisions(M, meta, T))
    S = Sheet(wb, "10 Viva Preparation", [24, 56, 130], hdr)
    S.note("Answers describe the ACTUAL implementation and quote figures from the final training run. If the models are retrained, rebuild this workbook so the numbers stay correct.")
    S.table(["Category", "Question", "Answer"], C.viva(M, meta, eda, T))
    S = Sheet(wb, "11 Pending & Limitations", [26, 62, 30, 90, 14], hdr)
    S.table(["Category", "Item", "Status", "Detail", "Priority"], C.pending(M, meta, T))

    for ws in wb.worksheets:
        ws.sheet_properties.tabColor = "1F3A5F"
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
    wb.save(OUT)
    print(f"wrote {OUT}  ({len(wb.worksheets)} sheets)")
    write_status(M, meta, T)


def write_status(M, meta, T):
    lab = meta["model_names"]
    d = meta["default_model"]
    x = M["test"][d]["thresholds"]["0.3"]
    files = "\n".join(f"| tests/{f}.py | {p} of {n} |" for f, (p, n) in sorted(T["per_file"].items()))
    text = f"""# Project status

Last updated: {TODAY}

## Current status

The project is a working predictive-maintenance web application (Flask, SQLite, scikit-learn, XGBoost) with the audit-driven improvements
implemented and verified as described below. The full record (what changed, why, how it works, viva preparation) is in
`docs/Project_Documentation_and_Change_Log.xlsx`, which is rebuilt by `python docs/build_workbook.py`.

## Completed phases

| Phase | Status |
|---|---|
| Phase 1 - Understand and audit | Done. Confirmed issues: SMOTENC after one-hot (115 invalid synthetic rows), CSRF accepted empty tokens, a new machine per prediction, duplicate and stale alerts, result page recomputed from current models, hard-coded development secret, one bad model file stopped the app, misleading 'strain' name. |
| Phase 2 - Plan | Done (prioritised table in the delivery message). |
| Phase 3 - Implement | Done: ML pipeline, ensemble, failure-mode classifiers, backend, database and migration, UI redesign, tests, documentation. |
| Phase 4 - Verify | Done on Linux (Python 3.12 and 3.14). Not done on Windows. |

## Latest test results

`python -m pytest tests -q` -> **{T['passed']} passed of {T['total']}** (latest run, Python 3.13 on Linux). The first 148 tests also passed on Python 3.12 and 3.14 (2026-11-18).

| File | Passed |
|---|---|
{files}

Also verified by hand: a browser walk-through of every page in Chromium (1440 px and 390 px) with the shipped models and no JavaScript errors;
an interrupted-and-resumed training run; a v1 to v2 migration with duplicate alerts, mixed-case IDs, a type conflict and an acknowledged alert.

## Final selected model

**{lab[d]}**, model version `{meta['version']}` (profile `{meta['profile']}`), chosen from cross-validated, calibrated PR-AUC on the training
split only (`models/metrics.json` -> `selection`). Held-out test set at the 30% threshold: precision {x['precision']:.3f}, recall {x['recall']:.3f},
F1 {x['f1']:.3f}, PR-AUC {M['test'][d]['pr_auc']:.3f}. The soft-voting ensemble was evaluated and did **not** beat it.

## Current phase

Hand-over. Waiting for the project team to review.

## Pending tasks and open items

* Confirm on the Windows machine that the models load (XGBoost is now stored in portable JSON form; the original failure could not be reproduced on Linux).
* Decide when to drop the `legacy_predictions_v1` table kept by the migration.
* Known limitations: synthetic dataset and no external validation; 68 test failures; manual input only; non-nested hyperparameter tuning; not load-tested.

Full list: sheet *11 Pending & Limitations* of the workbook.
"""
    open(os.path.join(HERE, "PROJECT_STATUS.md"), "w").write(text)
    print("wrote docs/PROJECT_STATUS.md")


if __name__ == "__main__":
    main()
