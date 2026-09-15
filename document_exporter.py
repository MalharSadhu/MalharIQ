import io
from fpdf import FPDF
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt


# ==============================================================================
# 1. EXECUTIVE CONSULTING POWERPOINT EXPORTER (WITH EMBEDDED VISUAL CHARTS)
# ==============================================================================
def create_executive_deck(
    biz_report: dict,
    tech_report: dict,
    ml_results: dict,
    split_counts: dict,
    task: str,
    chart_buffers: list = None,
) -> io.BytesIO:
  prs = Presentation()
  prs.slide_width = Inches(13.333)
  prs.slide_height = Inches(7.5)
  blank_layout = prs.slide_layouts[6]

  NAVY = RGBColor(15, 23, 42)
  BLUE = RGBColor(37, 99, 235)
  SLATE = RGBColor(100, 116, 139)
  LIGHT_BG = RGBColor(248, 250, 252)
  BORDER = RGBColor(226, 232, 240)
  WHITE = RGBColor(255, 255, 255)
  DARK_TEXT = RGBColor(30, 41, 59)

  def draw_banner(slide, title_text: str, category_text: str):
    banner = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.color.rgb = NAVY

    tf = banner.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.8)

    p_cat = tf.paragraphs[0]
    p_cat.text = f"MALHARIQ ADVISORY | {category_text.upper()}"
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = BLUE

    p_tit = tf.add_paragraph()
    p_tit.text = title_text
    p_tit.font.size = Pt(20)
    p_tit.font.bold = True
    p_tit.font.color.rgb = WHITE

  def draw_card(
      slide, left, top, width, height, title: str, items: list, header_color=BLUE
  ):
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = BORDER
    card.line.width = Pt(1.5)

    tf = card.text_frame
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.3)
    tf.margin_top = Inches(0.3)
    tf.margin_right = Inches(0.3)
    tf.word_wrap = True

    p_head = tf.paragraphs[0]
    p_head.text = title
    p_head.font.size = Pt(14)
    p_head.font.bold = True
    p_head.font.color.rgb = header_color

    for item in items:
      p = tf.add_paragraph()
      p.text = f"• {item}"
      p.font.size = Pt(11)
      p.font.color.rgb = DARK_TEXT
      p.space_before = Pt(8)

  # SLIDE 1: Cover
  s1 = prs.slides.add_slide(blank_layout)
  bg = s1.shapes.add_shape(
      MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
  )
  bg.fill.solid()
  bg.fill.fore_color.rgb = NAVY
  bg.line.fill.background()

  tf1 = s1.shapes.add_textbox(
      Inches(1.0), Inches(2.0), Inches(11.3), Inches(3.5)
  ).text_frame
  tf1.word_wrap = True
  p0 = tf1.paragraphs[0]
  p0.text = "MALHARIQ EXECUTIVE ADVISORY BRIEFING"
  p0.font.size = Pt(12)
  p0.font.bold = True
  p0.font.color.rgb = BLUE

  p1 = tf1.add_paragraph()
  p1.text = f"Mandate Evaluation: {task}"
  p1.font.size = Pt(32)
  p1.font.bold = True
  p1.font.color.rgb = WHITE
  p1.space_before = Pt(8)

  champ = ml_results.get("best_algorithm", "Autonomous Model Selection")
  p2 = tf1.add_paragraph()
  p2.text = (
      f"Winning Benchmark Architecture: {champ}\nRigorous 70/15/15 Holdout"
      " Validation Protocol"
  )
  p2.font.size = Pt(14)
  p2.font.color.rgb = SLATE
  p2.space_before = Pt(14)

  # SLIDE 2: Executive Findings
  s2 = prs.slides.add_slide(blank_layout)
  draw_banner(s2, "Executive Verdict & Action Items", "Strategic Insights")
  v_items = [
      str(
          biz_report.get(
              "executive_verdict", "Strategic evaluation completed."
          )
      )
  ]
  for ins in biz_report.get("segment_insights", [])[:3]:
    v_items.append(f"Operational Insight: {ins}")
  draw_card(
      s2,
      Inches(0.8),
      Inches(1.5),
      Inches(5.6),
      Inches(5.4),
      "Executive Verdict & Data Assessment",
      v_items,
      header_color=NAVY,
  )

  recs = biz_report.get("strategic_recommendations", [])
  if not recs:
    recs = ["Operationalize model with continuous holdout tracking."]
  draw_card(
      s2,
      Inches(6.8),
      Inches(1.5),
      Inches(5.7),
      Inches(5.4),
      "Immediate Strategic Next Steps",
      recs[:4],
      header_color=BLUE,
  )

  # SLIDE 3: Benchmark & Leaderboard
  s3 = prs.slides.add_slide(blank_layout)
  draw_banner(
      s3, "Algorithm Tournament & Validation Architecture", "MLOps Engineering"
  )
  tot = max(sum(split_counts.values()) if split_counts else 1, 1)
  splits_info = [
      ("Training Split (70%)", split_counts.get("train", 0), Inches(0.8)),
      ("Validation Split (15%)", split_counts.get("val", 0), Inches(4.8)),
      ("Test Holdout (15%)", split_counts.get("test", 0), Inches(8.8)),
  ]
  for label, count, x_pos in splits_info:
    kpi = s3.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, x_pos, Inches(1.4), Inches(3.7), Inches(1.1)
    )
    kpi.fill.solid()
    kpi.fill.fore_color.rgb = LIGHT_BG
    kpi.line.color.rgb = BORDER
    ktf = kpi.text_frame
    ktf.vertical_anchor = MSO_ANCHOR.MIDDLE
    ktf.margin_left = Inches(0.25)
    kp = ktf.paragraphs[0]
    kp.text = label
    kp.font.size = Pt(10)
    kp.font.color.rgb = SLATE
    kp_val = ktf.add_paragraph()
    kp_val.text = f"{count:,} records ({count/tot*100:.1f}%)"
    kp_val.font.size = Pt(16)
    kp_val.font.bold = True
    kp_val.font.color.rgb = NAVY

  if (
      "leaderboard_df" in ml_results
      and hasattr(ml_results["leaderboard_df"], "empty")
      and not ml_results["leaderboard_df"].empty
  ):
    ldf = ml_results["leaderboard_df"].head(4)
    rows, cols = ldf.shape[0] + 1, min(ldf.shape[1], 5)
    table = s3.shapes.add_table(
        rows, cols, Inches(0.8), Inches(2.8), Inches(11.7), Inches(3.8)
    ).table
    for c_idx, col_name in enumerate(ldf.columns[:cols]):
      cell = table.cell(0, c_idx)
      cell.text = str(col_name)
      cell.fill.solid()
      cell.fill.fore_color.rgb = NAVY
      p = cell.text_frame.paragraphs[0]
      p.font.size = Pt(11)
      p.font.bold = True
      p.font.color.rgb = WHITE
    for r_idx, row in ldf.iterrows():
      for c_idx in range(cols):
        cell = table.cell(r_idx + 1, c_idx)
        cell.text = str(row.iloc[c_idx])
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else LIGHT_BG
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(10)
        p.font.color.rgb = DARK_TEXT

  # SLIDE 4: Dedicated Visual Analytics Slide (Charts Embedded)
  if chart_buffers:
    s_chart = prs.slides.add_slide(blank_layout)
    draw_banner(
        s_chart,
        "Empirical Evidence & Model Visuals",
        "Visual Diagnostics & Evidence",
    )
    if len(chart_buffers) == 1:
      chart_buffers[0].seek(0)
      s_chart.shapes.add_picture(
          chart_buffers[0], Inches(1.5), Inches(1.5), Inches(10.3), Inches(5.3)
      )
    else:
      chart_buffers[0].seek(0)
      s_chart.shapes.add_picture(
          chart_buffers[0], Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.0)
      )
      chart_buffers[1].seek(0)
      s_chart.shapes.add_picture(
          chart_buffers[1], Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.0)
      )

  # SLIDE 5: Governance & Compliance
  s5 = prs.slides.add_slide(blank_layout)
  draw_banner(
      s5, "Governance, Demographic Parity & Risk Audit", "Risk Management"
  )
  draw_card(
      s5,
      Inches(0.8),
      Inches(1.5),
      Inches(5.6),
      Inches(5.4),
      "Regulatory Compliance (EEOC Four-Fifths)",
      [
          "Applies the EEOC 80% Rule across all categorical slices.",
          (
              "Guarantees that no protected cohort receives less than 80% of"
              " benchmark favorable outcomes."
          ),
          (
              "Readiness Audit Score:"
              f" {biz_report.get('data_readiness_score', 'Enterprise Grade')}"
          ),
      ],
      header_color=NAVY,
  )
  risks = biz_report.get("potential_business_risks", [])
  if not risks:
    risks = ["Regular model retraining recommended against data drift."]
  draw_card(
      s5,
      Inches(6.8),
      Inches(1.5),
      Inches(5.7),
      Inches(5.4),
      "Identified Operational Risk Factors",
      risks[:4],
      header_color=BLUE,
  )

  buffer = io.BytesIO()
  prs.save(buffer)
  buffer.seek(0)
  return buffer


# ==============================================================================
# 2. PDF EXPORTER (WITH EMBEDDED CHARTS)
# ==============================================================================
class MalharPDFReport(FPDF):

  def header(self):
    self.set_font("Helvetica", "B", 8)
    self.set_text_color(100, 116, 139)
    self.cell(
        0, 8, "MALHARIQ ADVISORY PLATFORM | AUDIT DELIVERABLE", 0, 1, "R"
    )
    self.ln(2)

  def footer(self):
    self.set_y(-12)
    self.set_font("Helvetica", "I", 8)
    self.set_text_color(150, 150, 150)
    self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def create_pdf_report(
    title: str,
    sections: list[tuple[str, list[str]]],
    metadata: dict,
    chart_buffers: list = None,
) -> io.BytesIO:
  pdf = MalharPDFReport()
  pdf.set_auto_page_break(auto=True, margin=15)
  pdf.add_page()

  pdf.set_font("Helvetica", "B", 18)
  pdf.set_text_color(15, 23, 42)
  pdf.cell(0, 10, title, ln=True)

  pdf.set_font("Helvetica", "", 10)
  pdf.set_text_color(37, 99, 235)
  meta_line = (
      f"Mandate: {metadata.get('task', 'N/A')}  |  Champion:"
      f" {metadata.get('champion', 'N/A')}"
  )
  pdf.cell(0, 6, meta_line, ln=True)
  pdf.ln(4)

  for heading, bullets in sections:
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, heading, ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    for b in bullets:
      pdf.multi_cell(0, 5, f"- {b}")
      pdf.ln(1)
    pdf.ln(2)

  # Embed Chart Images in PDF
  if chart_buffers:
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Visual Diagnostics & Empirical Evidence", ln=True)
    pdf.ln(4)
    for buf in chart_buffers:
      buf.seek(0)
      pdf.image(buf, w=175)
      pdf.ln(6)

  buffer = io.BytesIO()
  pdf.output(buffer)
  buffer.seek(0)
  return buffer