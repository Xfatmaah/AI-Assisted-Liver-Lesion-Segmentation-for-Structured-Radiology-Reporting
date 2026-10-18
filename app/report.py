"""Structured HTML report (self-contained, printable)."""

from __future__ import annotations



import base64

import datetime as dt

import html

import io



import config





CSS = """

<style>

.rpt{

    font-family:Segoe UI,Roboto,Arial,sans-serif;

    color:#111827;

    background:#ffffff;

    padding:22px 26px;

    border-radius:10px;

    max-width:1100px;

    margin:auto;

    line-height:1.55;

}



/* Main headings */

.rpt h1{

    font-size:24px;

    margin:0 0 4px;

    color:#111827;

    font-weight:750;

}



.rpt h2{

    font-size:17px;

    margin:24px 0 10px;

    color:#172554;

    border-bottom:2px solid #334155;

    padding-bottom:6px;

    font-weight:700;

}



/* Study metadata */

.rpt .meta{

    color:#374151;

    font-size:13px;

}



.rpt .meta b{

    color:#111827;

}



/* General notice */

.rpt .banner{

    background:#fff7ed;

    border:1px solid #c2410c;

    border-left:5px solid #9a3412;

    color:#7c2d12;

    padding:10px 12px;

    border-radius:5px;

    font-size:13px;

    margin:12px 0;

}



/* Warnings */

.rpt .warn{

    background:#fef2f2;

    border:1px solid #b91c1c;

    border-left:5px solid #991b1b;

    color:#7f1d1d;

    padding:10px 12px;

    border-radius:5px;

    font-size:13px;

    margin:8px 0;

}



/* Tables */

.rpt table{

    border-collapse:collapse;

    width:100%;

    font-size:13px;

    margin:8px 0;

    background:#ffffff;

    color:#111827;

}



.rpt th{

    background:#1e293b;

    color:#ffffff;

    text-align:left;

    padding:9px 8px;

    border:1px solid #334155;

    font-weight:700;

}



.rpt td{

    padding:8px;

    border:1px solid #9ca3af;

    color:#111827;

    background:#ffffff;

}



.rpt tr:nth-child(even) td{

    background:#f1f5f9;

}



.rpt td b{

    color:#111827;

}



/* KPI cards */

.rpt .kpis{

    display:flex;

    gap:10px;

    flex-wrap:wrap;

    margin-top:16px;

}



.rpt .kpi{

    flex:1;

    min-width:140px;

    background:#f1f5f9;

    border:1px solid #64748b;

    border-top:4px solid #1e293b;

    border-radius:6px;

    padding:10px 12px;

}



.rpt .kpi b{

    display:block;

    font-size:22px;

    color:#111827;

    font-weight:750;

}



.rpt .kpi span{

    font-size:12px;

    color:#374151;

}



/* Impression */

.rpt .imp{

    background:#f1f5f9;

    border:1px solid #94a3b8;

    border-left:5px solid #1e293b;

    padding:12px 14px;

    border-radius:4px;

    font-size:14px;

    color:#111827;

}



/* Probability bars */

.rpt .bar{

    background:#cbd5e1;

    border:1px solid #64748b;

    border-radius:4px;

    height:10px;

    width:140px;

    display:inline-block;

    vertical-align:middle;

    overflow:hidden;

}



.rpt .bar i{

    display:block;

    height:10px;

    border-radius:3px;

    background:#1d4ed8;

}



.rpt .bar.pos i{

    background:#b91c1c;

}



/* Images */

.rpt .imgs{

    display:flex;

    flex-wrap:wrap;

    gap:12px;

}



.rpt .imgs figure{

    margin:0;

    width:300px;

}



.rpt .imgs img{

    width:100%;

    border-radius:5px;

    border:1px solid #64748b;

}



.rpt figcaption{

    font-size:12px;

    color:#1f2937;

    margin-top:4px;

}



/* Secondary text */

.rpt .small{

    font-size:12px;

    color:#374151;

}



/* Follow-up changes */

.rpt .up{

    color:#991b1b;

    font-weight:700;

}



.rpt .down{

    color:#166534;

    font-weight:700;

}



/* Printable report */

@media print{

    .rpt{

        max-width:none;

        padding:10px;

        border-radius:0;

    }



    .rpt h1,

    .rpt h2,

    .rpt .kpi,

    .rpt .imp,

    .rpt .banner,

    .rpt .warn,

    .rpt th{

        -webkit-print-color-adjust:exact;

        print-color-adjust:exact;

    }



    .rpt table{

        break-inside:auto;

    }



    .rpt tr,

    .rpt figure{

        break-inside:avoid;

    }

}







/* Force readable text throughout the report */

.rpt,

.rpt p,

.rpt span,

.rpt div,

.rpt td,

.rpt th,

.rpt label,

.rpt figcaption,

.rpt .small,

.rpt h1,

.rpt h2,

.rpt .meta,

.rpt .imp,

.rpt .kpi span {

    color: #111111 !important;

    opacity: 1 !important;

    -webkit-text-fill-color: currentColor !important;

}



/* Table body */

.rpt table td {

    color: #111111 !important;

    background-color: #ffffff;

    opacity: 1 !important;

}



/* Alternating rows */

.rpt table tr:nth-child(even) td {

    background-color: #e5e7eb;

}



/* Table headers */

.rpt table th {

    color: #ffffff !important;

    background-color: #1e293b !important;

    opacity: 1 !important;

    -webkit-text-fill-color: #ffffff !important;

}



/* Main headings */

.rpt h1,

.rpt h2 {

    color: #111827 !important;

    opacity: 1 !important;

    -webkit-text-fill-color: #111827 !important;

}



/* KPI values */

.rpt .kpi b {

    color: #000000 !important;

    opacity: 1 !important;

    -webkit-text-fill-color: #000000 !important;

}



</style>
"""





def e(x):

    return html.escape(str(x))





def _b64(img):

    buf = io.BytesIO()

    img.save(buf, "PNG")

    return base64.b64encode(buf.getvalue()).decode()





def _fmt(x, nd=1, unit=""):

    return "—" if x is None else f"{x:.{nd}f}{unit}"





def build_report_html(res: dict, key_images) -> str:

    cur = res["current"]

    s, liver, g = cur["summary"], cur["liver"], cur["geometry"]

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    h = [CSS, '<div class="rpt">']

    h.append(f"<h1>AI-Assisted Liver Lesion Segmentation</h1><div class='meta'>Study: <b>{e(cur['case_id'])}</b> · {now}</div>")

    h.append(f"<div class='banner'>{e(config.DISCLAIMER)}</div>")

    for n in res.get("notes", []):

        h.append(f"<div class='warn'>⚠ {e(n)}</div>")



    # KPIs

    burden = _fmt(s["lesion_burden_pct"], 2, "%")

    h.append("<div class='kpis'>"

             f"<div class='kpi'><b>{'Yes' if s['lesion_present'] else 'No'}</b><span>Liver lesion detected</span></div>"

             f"<div class='kpi'><b>{s['lesion_count']}</b><span>Lesion count</span></div>"

             f"<div class='kpi'><b>{s['total_lesion_volume_mm3']:,.0f} mm³</b><span>Total lesion volume</span></div>"
             f"<div class='kpi'><b>{liver['liver_volume_mm3']:,.0f} mm³</b><span>Liver volume</span></div>"

             f"<div class='kpi'><b>{burden}</b><span>Lesion burden</span></div></div>")



    h.append(f"<h2>Impression (draft)</h2><div class='imp'>{e(res['impression'])}</div>"

             f"<div class='small'>Impression source: {e(res.get('impression_source', 'template'))}</div>")



    # Lesion table

    h.append("<h2>Lesions</h2>")

    if cur["lesions"]:

        h.append("<table><tr><th>Lesion</th><th>Couinaud segment</th><th>Spans</th><th>Lobe</th>"

                 "<th>X (mm)</th><th>Y (mm)</th><th>Z (mm)</th><th>Volume (mm³)</th><th>Mean HU</th></tr>")

        for l in cur["lesions"]:

            h.append(

                f"<tr><td><b>L{l['id']}</b></td><td>{e(l['primary_segment_label'])}</td><td>{e(l['segments_involved_label'])}</td>"

                f"<td>{e(l['lobe'])}</td><td>{l['bbox_x_LR_mm']:.1f}</td><td>{l['bbox_y_AP_mm']:.1f}</td><td>{l['bbox_z_SI_mm']:.1f}</td>"

                f"<td>{l['volume_mm3']:,.0f}</td><td>{l['mean_hu']:.0f}</td></tr>")

        h.append("</table>")

    else:

        h.append("<div class='small'>No lesion above the minimum volume threshold.</div>")



    # Segment table

    h.append("<h2>Couinaud segment distribution</h2><table><tr><th>Segment</th>" + "".join(f"<th>{k}</th>" for k in liver["segment_volume_mm3"]) + "</tr>")
    h.append("<tr><td>Segment volume (mm³)</td>" + "".join(f"<td>{v:,.0f}</td>" for v in liver["segment_volume_mm3"].values()) + "</tr>")
    h.append("<tr><td>Lesion volume (mm³)</td>" + "".join(f"<td>{v:,.0f}</td>" for v in liver["lesion_volume_per_segment_mm3"].values()) + "</tr></table>")


    # Images

    if key_images:

        h.append("<h2>Segmentation overlays</h2><div class='imgs'>")

        for img, cap in key_images:

            h.append(f"<figure><img src='data:image/png;base64,{_b64(img)}'/><figcaption>{e(cap)}</figcaption></figure>")

        h.append("</div>")



    # RADAR

    radar = res.get("radar")

    h.append("<h2>RADAR liver findings (model probabilities)</h2>")

    if radar and radar.get("findings"):

        h.append("<table><tr><th>Finding</th><th>Probability</th><th></th></tr>")

        for f in radar["findings"]:

            pos = f["probability"] >= config.RADAR_POSITIVE_THRESHOLD

            h.append(f"<tr><td>{e(f['finding'])}</td><td>{f['probability']:.3f}</td>"

                     f"<td><span class='bar {'pos' if pos else ''}'><i style='width:{min(f['probability'], 1) * 100:.0f}%'></i></span> "

                     f"{'<b>≥ threshold</b>' if pos else ''}</td></tr>")

        h.append(f"</table><div class='small'>Threshold {config.RADAR_POSITIVE_THRESHOLD}. Probabilities are not calibrated diagnoses. Source: {e(radar.get('source', ''))}</div>")

    else:

        h.append(f"<div class='small'>Not available ({e((radar or {}).get('reason', 'disabled'))}).</div>")



    # Follow-up

    cmp_ = res.get("comparison")

    if cmp_:

        h.append("<h2>Follow-up comparison (prior → current)</h2><table><tr><th>Parameter</th><th>Prior</th><th>Current</th><th>Δ</th><th>Δ %</th></tr>")

        for r in cmp_["rows"]:

            cls = "" if r["delta"] in (None, 0) else ("up" if r["delta"] > 0 else "down")

            d = "—" if r["delta"] is None else f"{r['delta']:+.{r['nd']}f}"

            p = "—" if r["pct"] is None else f"{r['pct']:+.1f}%"

            h.append(f"<tr><td>{e(r['name'])}</td><td>{_fmt(r['prior'], r['nd'])}</td><td>{_fmt(r['current'], r['nd'])}</td><td class='{cls}'>{d}</td><td class='{cls}'>{p}</td></tr>")

        h.append(f"</table><div class='small'>Prior study: {e(res['prior']['case_id'])}. Study-level comparison only (no lesion-to-lesion registration); "

                 "this is a descriptive change, not a treatment-response classification.</div>")



    h.append("</div>")

    return "".join(h)
