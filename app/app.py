"""Gradio front-end: upload NIfTI CT -> segmentation + measurements + RADAR -> overlay viewer + structured report."""
from __future__ import annotations

import pandas as pd
import gradio as gr

import config
import viz
from agent import LiverReportingAgent

agent = LiverReportingAgent()
LESION_COLS = {
    "id": "Lesion", "primary_segment_label": "Couinaud segment", "segments_involved_label": "Spans", "lobe": "Lobe",
    "bbox_x_LR_mm": "X (mm)", "bbox_y_AP_mm": "Y (mm)", "bbox_z_SI_mm": "Z (mm)",
    "volume_mm3": "Volume (mm³)", "mean_hu": "Mean HU",
}
NO = gr.update()


def lesion_table(lesions):
    if not lesions:
        return pd.DataFrame(columns=list(LESION_COLS.values()))
    df = pd.DataFrame(lesions)[list(LESION_COLS)].rename(columns=LESION_COLS)
    df["Lesion"] = df["Lesion"].map(lambda i: f"L{i}")
    return df.round(2)


def analyze(ct_file, run_radar, min_mm3):
    if not ct_file:
        gr.Warning("Please upload a CT NIfTI file first.")
        yield ("Upload a CT NIfTI (.nii / .nii.gz) to begin.",) + (NO,) * 11
        return
    log = []
    final = None
    for kind, payload in agent.run(ct_file, None, run_radar, False, float(min_mm3)):
        if kind == "log":
            log.append(payload)
            yield ("\n".join(log),) + (NO,) * 11
        elif kind == "result":
            final = payload
        elif kind == "error":
            gr.Warning(payload)
    if final is None:
        return
    cur = final["current"]
    cd = cur["case_dir"]
    idx = viz.default_index(cd, "axial")
    lesion_choices = [f"L{l['id']} · {l['primary_segment_label']} · {l['axial_long_mm']:.0f} mm" for l in cur["lesions"]]
    yield (
        "\n".join(log),
        {"case_dir": cd, "lesions": cur["lesions"]},
        viz.render_slice(cd, "axial", idx),
        gr.update(minimum=1, maximum=viz.n_slices(cd, "axial"), value=idx + 1, step=1),
        gr.update(value="axial"),
        gr.update(choices=lesion_choices, value=None),
        final["key_images"],
        final["report_html"],
        lesion_table(cur["lesions"]),
        final["files"],
        gr.update(),
        gr.update(),
    )


def _render(state, plane, slice_no, window, show_seg, show_les, zoom, opacity):
    if not state:
        return None
    return viz.render_slice(state["case_dir"], plane, int(slice_no) - 1, window, show_seg, show_les, zoom, opacity=opacity / 100)


def on_plane(state, plane, window, show_seg, show_les, zoom, opacity):
    if not state:
        return NO, None
    cd = state["case_dir"]
    idx = viz.default_index(cd, plane)
    return gr.update(minimum=1, maximum=viz.n_slices(cd, plane), value=idx + 1), viz.render_slice(cd, plane, idx, window, show_seg, show_les, zoom, opacity=opacity / 100)


def on_lesion(state, choice, window, show_seg, show_les, zoom, opacity):
    if not state or not choice:
        return NO, NO, NO
    lid = int(choice.split("·")[0].strip()[1:])
    les = next(l for l in state["lesions"] if l["id"] == lid)
    z = les["axial_slice"]
    img = viz.render_slice(state["case_dir"], "axial", z, window, show_seg, show_les, zoom, highlight=lid, opacity=opacity / 100)
    return gr.update(value="axial"), gr.update(minimum=1, maximum=viz.n_slices(state["case_dir"], "axial"), value=z + 1), img


with gr.Blocks(title="AI-Assisted Liver Lesion Reporting") as demo:
    gr.Markdown("# AI-Assisted Liver Lesion Segmentation")
    state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            ct_in = gr.File(label="Current CT (.nii / .nii.gz)", file_types=[".nii", ".gz"], type="filepath")
            # prior_in = gr.File(label="Prior CT for follow-up (optional)", file_types=[".nii", ".gz"], type="filepath")
            run_radar = gr.Checkbox(True, label="Run RADAR (slow on CPU; skipped if unavailable)")
            min_mm3 = gr.Number(config.MIN_LESION_MM3, label="Ignore lesions smaller than (mm³)", minimum=0)
            btn = gr.Button("Analyze", variant="primary")
            log_box = gr.Textbox(label="Agent log", lines=14, interactive=False)
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Viewer"):
                    with gr.Row():
                        plane = gr.Radio(["axial", "coronal", "sagittal"], value="axial", label="Plane")
                        window = gr.Radio(list(viz.WINDOWS), value=viz.DEFAULT_WINDOW, label="CT window preset")
                    with gr.Row():
                        show_seg = gr.Checkbox(True, label="Couinaud segments")
                        show_les = gr.Checkbox(True, label="Lesions")
                        zoom = gr.Checkbox(True, label="Zoom on liver")
                        jump = gr.Dropdown([], label="Jump to lesion")
                        opacity = gr.Slider(0, 100, value=30, step=5, label="Overlay opacity %")
                    viewer = gr.Image(label="CT + AI overlay", type="pil", interactive=False, height=720)
                    slider = gr.Slider(1, 100, value=1, step=1, label="Slice")
                    gr.HTML(viz.LEGEND_HTML)
                with gr.Tab("Key lesion slices"):
                    gallery = gr.Gallery(columns=3, height=560, object_fit="contain")
                with gr.Tab("Structured report"):
                    report_html = gr.HTML()
                with gr.Tab("Lesion table"):
                    table = gr.Dataframe(interactive=False, wrap=True)
                with gr.Tab("Downloads"):
                    files = gr.File(label="Downloads", file_count="multiple")

    view_inputs = [state, plane, slider, window, show_seg, show_les, zoom, opacity]
    run_outputs = [log_box, state, viewer, slider, plane, jump, gallery, report_html, table, files, window, zoom]
    btn.click(analyze, [ct_in, run_radar, min_mm3], run_outputs, concurrency_limit=1)
    
    for c in (slider, window, show_seg, show_les, zoom, opacity):
        c.change(_render, view_inputs, viewer, show_progress="hidden")
    plane.input(on_plane, [state, plane, window, show_seg, show_les, zoom, opacity], [slider, viewer])
    jump.input(on_lesion, [state, jump, window, show_seg, show_les, zoom, opacity], [plane, slider, viewer])

demo.queue(max_size=4)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
