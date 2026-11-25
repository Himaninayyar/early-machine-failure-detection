"""Spreadsheet -> CSV normalisation, so the rest of the project only ever reads CSV.

Training / EDA:   ensure_csv("data/raw/my_data.xlsx")  -> writes data/raw/my_data.csv (once) and returns that path.
Web batch upload: xlsx_bytes_to_csv_bytes(upload_bytes) -> UTF-8 CSV bytes, then the normal CSV code path runs.

CLI:  python -m ml.data_io data/raw/my_data.xlsx [--sheet NAME]
"""
import argparse
import csv
import io
import os

XLSX_EXT = (".xlsx", ".xlsm")
CSV_EXT = (".csv",)
SUPPORTED_EXT = CSV_EXT + XLSX_EXT


class DataFileError(ValueError):
    pass


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))          # 1551.0 -> 1551
    return str(v)


def _rows_from_workbook(source, sheet=None, max_rows=None):
    try:
        from openpyxl import load_workbook
        wb = load_workbook(source, read_only=True, data_only=True)   # data_only: cached formula results, not formulas
    except ImportError as exc:
        raise DataFileError("Reading .xlsx needs openpyxl (pip install openpyxl).") from exc
    except Exception as exc:
        raise DataFileError("The file is not a valid .xlsx workbook.") from exc
    try:
        if sheet is not None and sheet not in wb.sheetnames:
            raise DataFileError(f"Sheet '{sheet}' not found. Sheets: {', '.join(wb.sheetnames)}.")
        ws = wb[sheet] if sheet else wb.worksheets[0]
        out = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if max_rows is not None and i > max_rows:        # header + max_rows data rows; stops early on huge sheets
                raise DataFileError(f"The sheet has more than {max_rows} rows. Split it into smaller files.")
            cells = [_cell(v) for v in row]
            if any(c.strip() for c in cells):                # skip fully blank rows
                out.append(cells)
        if not out:
            raise DataFileError("The sheet is empty.")
        width = len(out[0])
        while width and not out[0][width - 1].strip():       # trim trailing empty header columns
            width -= 1
        return [r[:width] + [""] * (width - len(r)) for r in out]
    finally:
        wb.close()


def _to_csv_text(rows):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


def xlsx_bytes_to_csv_bytes(raw, sheet=None, max_rows=None):
    """For uploads. Returns UTF-8 CSV bytes of the first (or named) sheet."""
    return _to_csv_text(_rows_from_workbook(io.BytesIO(raw), sheet, max_rows)).encode("utf-8")


def ensure_csv(path, sheet=None):
    """Return a path to a CSV. A .csv is returned as is; an .xlsx is converted next to itself (re-converted
    only if the workbook is newer than its CSV)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in CSV_EXT:
        return path
    if ext not in XLSX_EXT:
        raise DataFileError(f"Unsupported data file '{os.path.basename(path)}'. Use .csv or .xlsx.")
    out = os.path.splitext(path)[0] + ".csv"
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        text = _to_csv_text(_rows_from_workbook(path, sheet))
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, out)
        print(f"Converted {path} -> {out}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert an .xlsx dataset to CSV (CSV files are left untouched).")
    ap.add_argument("path")
    ap.add_argument("--sheet", default=None)
    a = ap.parse_args()
    print(ensure_csv(a.path, a.sheet))
