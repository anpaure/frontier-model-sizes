#!/usr/bin/env python3
import argparse
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}


def col_number(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def row_number(ref):
    return int(re.search(r"\d+", ref).group(0))


def load_workbook(path):
    zf = zipfile.ZipFile(path)
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {r.attrib["Id"]: r.attrib["Target"].lstrip("/") for r in rels}
    sheets = []
    for s in wb.find("x:sheets", NS):
        rid = s.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        sheets.append((s.attrib["name"], rel_map[rid]))
    shared = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall("x:si", NS):
            shared.append("".join(t.text or "" for t in si.iterfind(".//x:t", NS)))
    return zf, sheets, shared


def cell_value(c, shared):
    t = c.attrib.get("t")
    if t == "inlineStr":
        return "".join(x.text or "" for x in c.iterfind(".//x:t", NS))
    v = c.find("x:v", NS)
    raw = "" if v is None or v.text is None else v.text
    if t == "s" and raw:
        return shared[int(raw)]
    if t == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workbook")
    p.add_argument("--sheet")
    p.add_argument("--min-row", type=int, default=1)
    p.add_argument("--max-row", type=int, default=10**9)
    p.add_argument("--min-col", type=int, default=1)
    p.add_argument("--max-col", type=int, default=10**9)
    p.add_argument("--formulas-only", action="store_true")
    args = p.parse_args()
    zf, sheets, shared = load_workbook(args.workbook)
    if not args.sheet:
        for i, (name, target) in enumerate(sheets, 1):
            root = ET.fromstring(zf.read(target))
            cells = root.findall(".//x:c", NS)
            refs = [c.attrib["r"] for c in cells]
            max_ref = max(refs, key=lambda r: (row_number(r), col_number(r))) if refs else ""
            print(f"{i}\t{name}\t{target}\t{len(cells)} cells\tlast={max_ref}")
        return
    target = dict(sheets)[args.sheet]
    root = ET.fromstring(zf.read(target))
    print("sheet\tcell\tvalue\tformula\ttype\tstyle")
    for c in root.findall(".//x:c", NS):
        ref = c.attrib["r"]
        row, col = row_number(ref), col_number(ref)
        if not (args.min_row <= row <= args.max_row and args.min_col <= col <= args.max_col):
            continue
        f = c.find("x:f", NS)
        formula = "" if f is None or f.text is None else f.text
        if args.formulas_only and not formula:
            continue
        value = cell_value(c, shared)
        if not value and not formula:
            continue
        clean = lambda x: x.replace("\t", " ").replace("\r", " ").replace("\n", "\\n")
        print("\t".join([clean(args.sheet), ref, clean(value), clean(formula), c.attrib.get("t", ""), c.attrib.get("s", "")]))


if __name__ == "__main__":
    main()
