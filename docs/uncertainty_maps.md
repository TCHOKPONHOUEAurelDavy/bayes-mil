# Uncertainty map generation

Bayes-MIL generates per-patch uncertainty overlays during the heatmap export
step that runs after training. The workflow is implemented in the
`create_heatmaps_bmil.py` helper and the visualization utilities it calls.

## Where the uncertainties are computed

When `create_heatmaps_bmil.py` iterates over each slide, the
`infer_single_slide` function draws Monte-Carlo samples from the Bayesian MIL
model to estimate three patch-level uncertainty tensors:

- **Data uncertainty** (`vis_data`)
- **Total uncertainty** (`vis_total`)
- **Model uncertainty** (`vis_model`)

For every slide the script stores these arrays alongside the attention scores
and patch coordinates in an HDF5 block map file so they can be reused for
plotting.【F:create_heatmaps_bmil.py†L82-L141】【F:create_heatmaps_bmil.py†L398-L430】

## Rendering the uncertainty maps

Later in the same script the `drawHeatmap` helper is invoked three times with
different `uncs_type` values (0=data, 1=total, 2=model). This function is a
thin wrapper that delegates to `WholeSlideImage.visHeatmap`, which handles the
actual overlay rendering. The `uncs_type` flag slices the concatenated
uncertainty array so that the correct modality is visualised before the image
is written to disk.【F:create_heatmaps_bmil.py†L440-L492】【F:vis_utils/heatmap_utils.py†L24-L35】【F:wsi_core/WholeSlideImage.py†L492-L568】

Together these components are responsible for creating and saving the
uncertainty heatmaps that accompany the standard attention maps.
