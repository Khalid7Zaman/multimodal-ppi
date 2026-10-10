#!/usr/bin/env python
"""Phase 5 Part C - build a formatted workbook of per-residue interface predictions for manual
checking (e.g. alongside PyMOL). One sheet per example complex (1H0T, 2MCN, 4KVG) plus a
summary sheet. Reads phase5/interp/interface_predictions.csv (from export_residue_table.py).
Run in the mmppi env:  python phase5/build_interface_workbook.py
Writes reports/interface_predictions.xlsx"""
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "interp", "interface_predictions.csv")
OUT = os.path.join(HERE, "..", "reports", "interface_predictions.xlsx")

df = pd.read_csv(SRC)
complexes = ["1H0T", "2MCN", "4KVG"]

FONT = "Times New Roman"
hdr_fill = PatternFill("solid", fgColor="1F3864")
hdr_font = Font(name=FONT, bold=True, color="FFFFFF", size=11)
title_font = Font(name=FONT, bold=True, size=14, color="1F3864")
note_font = Font(name=FONT, italic=True, size=9, color="595959")
base_font = Font(name=FONT, size=11)
center = Alignment(horizontal="center", vertical="center")
left = Alignment(horizontal="left", vertical="center")
thin = Side(style="thin", color="BFBFBF")
border = Border(left=thin, right=thin, top=thin, bottom=thin)

HEADERS = ["Protein", "Chain", "Residue #", "Amino acid",
           "Predicted interface prob.", "Experimental interface",
           "Predicted call (>0.5)", "Agreement"]

wb = Workbook()

for pdb in complexes:
    sub = df[df["pdb"] == pdb].reset_index(drop=True)
    ws = wb.create_sheet(title=pdb)
    ws.sheet_view.showGridLines = False
    ws["A1"] = f"Complex {pdb} - per-residue interface predictions"
    ws["A1"].font = title_font
    ws.merge_cells("A1:H1")
    ws["A2"] = ("Predicted probability is the model's per-residue interface score (0-1). "
                "Experimental interface: 1 = contact residue in the crystal structure, 0 = not. "
                "Predicted call applies a 0.5 cut-off; Agreement compares it to the experiment.")
    ws["A2"].font = note_font
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:H2")
    ws.row_dimensions[2].height = 30
    hr = 4
    for c, h in enumerate(HEADERS, start=1):
        cell = ws.cell(row=hr, column=c, value=h)
        cell.font = hdr_font
        cell.fill = hdr_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    ws.row_dimensions[hr].height = 30
    r = hr + 1
    for _, row in sub.iterrows():
        ws.cell(row=r, column=1, value=row["protein"]).alignment = left
        ws.cell(row=r, column=2, value=row["chain"]).alignment = center
        ws.cell(row=r, column=3, value=int(row["resseq"])).alignment = center
        ws.cell(row=r, column=4, value=row["aa"]).alignment = center
        pc = ws.cell(row=r, column=5, value=float(row["predicted_prob"]))
        pc.number_format = "0.000"; pc.alignment = center
        ws.cell(row=r, column=6, value=int(row["experimental_interface"])).alignment = center
        ws.cell(row=r, column=7, value=f"=IF(E{r}>0.5,1,0)").alignment = center
        ws.cell(row=r, column=8, value=f'=IF(G{r}=F{r},"match","mismatch")').alignment = center
        for c in range(1, 9):
            cc = ws.cell(row=r, column=c); cc.font = base_font; cc.border = border
        r += 1
    last = r - 1
    ws.conditional_formatting.add(
        f"E{hr+1}:E{last}",
        ColorScaleRule(start_type="num", start_value=0, start_color="4472C4",
                       mid_type="num", mid_value=0.5, mid_color="FFFFFF",
                       end_type="num", end_value=1, end_color="C00000"))
    ws.freeze_panes = f"A{hr+1}"
    for i, w in enumerate([11, 7, 11, 11, 15, 15, 14, 12], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

ws = wb.create_sheet(title="Summary", index=0)
ws.sheet_view.showGridLines = False
ws["A1"] = "Interface prediction - summary by complex"
ws["A1"].font = title_font
ws.merge_cells("A1:F1")
ws["A2"] = ("Per-residue agreement between the model's predicted interface (0.5 cut-off) and the "
            "experimental contact residues, for the three interpretability example complexes. "
            "See each complex's own sheet for the full residue list.")
ws["A2"].font = note_font
ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
ws.merge_cells("A2:F2")
ws.row_dimensions[2].height = 30
sh = ["Complex", "Residues modelled", "Experimental interface residues",
      "Predicted interface residues (>0.5)", "Residues in agreement", "Agreement (%)"]
hr = 4
for c, h in enumerate(sh, start=1):
    cell = ws.cell(row=hr, column=c, value=h)
    cell.font = hdr_font; cell.fill = hdr_fill
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = border
ws.row_dimensions[hr].height = 44
r = hr + 1
for pdb in complexes:
    n = len(df[df["pdb"] == pdb])
    ws.cell(row=r, column=1, value=pdb).alignment = center
    ws.cell(row=r, column=2, value=n).alignment = center
    ws.cell(row=r, column=3, value=f"=SUM('{pdb}'!F5:F{4+n})").alignment = center
    ws.cell(row=r, column=4, value=f"=SUM('{pdb}'!G5:G{4+n})").alignment = center
    ws.cell(row=r, column=5, value=f"=COUNTIF('{pdb}'!H5:H{4+n},\"match\")").alignment = center
    pct = ws.cell(row=r, column=6, value=f"=E{r}/B{r}")
    pct.number_format = "0.0%"; pct.alignment = center
    for c in range(1, 7):
        cc = ws.cell(row=r, column=c); cc.font = base_font; cc.border = border
    r += 1
tot = r
ws.cell(row=tot, column=1, value="All three").alignment = center
ws.cell(row=tot, column=2, value=f"=SUM(B{hr+1}:B{tot-1})").alignment = center
ws.cell(row=tot, column=3, value=f"=SUM(C{hr+1}:C{tot-1})").alignment = center
ws.cell(row=tot, column=4, value=f"=SUM(D{hr+1}:D{tot-1})").alignment = center
ws.cell(row=tot, column=5, value=f"=SUM(E{hr+1}:E{tot-1})").alignment = center
ptot = ws.cell(row=tot, column=6, value=f"=E{tot}/B{tot}")
ptot.number_format = "0.0%"; ptot.alignment = center
for c in range(1, 7):
    cc = ws.cell(row=tot, column=c)
    cc.font = Font(name=FONT, bold=True, size=11)
    cc.fill = PatternFill("solid", fgColor="D9E1F2"); cc.border = border
for i, w in enumerate([12, 18, 24, 26, 20, 14], start=1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.freeze_panes = f"A{hr+1}"

if "Sheet" in wb.sheetnames:
    del wb["Sheet"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
wb.save(OUT)
print("wrote", os.path.abspath(OUT))
