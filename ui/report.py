"""
ui/report.py  –  Generación de reportes PDF y Excel/CSV.
Dependencias adicionales: reportlab (PDF), openpyxl (Excel).
Si no están disponibles, se genera CSV compatible con Excel.
"""
import csv
import datetime
import os
from pathlib import Path

OUTPUT_DIR = Path("data/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PDF con reportlab
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(rows: list, summary: dict) -> str:
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(OUTPUT_DIR / f"reporte_{ts}.pdf")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                        Paragraph, Spacer)
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm

        doc    = SimpleDocTemplate(path, pagesize=A4,
                                   topMargin=2*cm, bottomMargin=2*cm,
                                   leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        story.append(Paragraph("Reporte de Clasificación de Frutas", styles["Title"]))
        story.append(Paragraph(f"Generado: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                                styles["Normal"]))
        story.append(Spacer(1, 0.5*cm))

        # ── Resumen ──────────────────────────────────────────────────────────
        story.append(Paragraph("Resumen por Fruta y Estado", styles["Heading2"]))
        sum_data = [["Fruta", "Verde", "Maduro", "Podrido", "Total"]]
        fruits = ["Banano", "Manzana"]
        grand  = 0
        for fruit in fruits:
            row = [fruit]
            total_f = 0
            for state in ["verde", "maduro", "podrido"]:
                key = f"{fruit.lower()}_{state}"
                val = summary.get(key, 0)
                row.append(str(val))
                total_f += val
            row.append(str(total_f))
            grand += total_f
            sum_data.append(row)
        sum_data.append(["TOTAL", "", "", "", str(grand)])

        tbl = Table(sum_data, colWidths=[4*cm]*5)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.8*cm))

        # ── Detalle ───────────────────────────────────────────────────────────
        story.append(Paragraph("Historial Detallado", styles["Heading2"]))
        det_data = [["ID", "Fecha/Hora", "Fruta", "Estado", "Confianza"]]
        for r in rows:
            det_data.append([str(r[0]), str(r[1])[:19], r[2], r[3], f"{float(r[4]):.1%}"])

        tbl2 = Table(det_data, colWidths=[1.5*cm, 5*cm, 3*cm, 3*cm, 2.5*cm])
        tbl2.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        story.append(tbl2)

        doc.build(story)
        print(f"PDF generado: {path}")

    except ImportError:
        # Fallback: PDF básico sin reportlab usando solo texto
        path = str(OUTPUT_DIR / f"reporte_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("REPORTE DE CLASIFICACION DE FRUTAS\n")
            f.write(f"Generado: {datetime.datetime.now()}\n\n")
            f.write("=== RESUMEN ===\n")
            for k, v in summary.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n=== DETALLE ===\n")
            for r in rows:
                f.write(f"  {r}\n")
        print(f"reportlab no disponible, se generó TXT: {path}")

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Excel con openpyxl (fallback a CSV)
# ─────────────────────────────────────────────────────────────────────────────
def generate_excel(rows: list, summary: dict) -> str:
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment

        path = str(OUTPUT_DIR / f"reporte_{ts}.xlsx")
        wb   = Workbook()
        ws_sum = wb.active
        ws_sum.title = "Resumen"

        header_fill = PatternFill("solid", fgColor="0F3460")
        header_font = Font(bold=True, color="FFFFFF")

        # ── Hoja resumen ──────────────────────────────────────────────────────
        headers = ["Fruta", "Verde", "Maduro", "Podrido", "Total"]
        ws_sum.append(headers)
        for cell in ws_sum[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        fruits = ["Banano", "Manzana"]
        grand  = 0
        for fruit in fruits:
            row_vals = [fruit]
            total_f  = 0
            for state in ["verde", "maduro", "podrido"]:
                val = summary.get(f"{fruit.lower()}_{state}", 0)
                row_vals.append(val)
                total_f += val
            row_vals.append(total_f)
            grand += total_f
            ws_sum.append(row_vals)
        ws_sum.append(["TOTAL", "", "", "", grand])

        # ── Hoja detalle ──────────────────────────────────────────────────────
        ws_det = wb.create_sheet("Detalle")
        ws_det.append(["ID", "Fecha/Hora", "Fruta", "Estado", "Confianza", "Snapshot"])
        for cell in ws_det[1]:
            cell.fill = header_fill
            cell.font = header_font
        for r in rows:
            ws_det.append([r[0], str(r[1])[:19], r[2], r[3],
                           f"{float(r[4]):.1%}", r[5] if len(r) > 5 else ""])

        for col in ws_det.columns:
            ws_det.column_dimensions[col[0].column_letter].width = 18

        wb.save(path)
        print(f"Excel generado: {path}")

    except ImportError:
        path = str(OUTPUT_DIR / f"reporte_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Fecha/Hora", "Fruta", "Estado",
                             "Confianza", "Snapshot"])
            for r in rows:
                writer.writerow(r)
        print(f"openpyxl no disponible, se generó CSV: {path}")

    return path
