# Sources of two review pages (private claude.ai artifacts)

These HTML files are the page sources; the images they reference are rendered from files already in the repo.
To change a page from a new session: `Artifact read` the URL, then publish this file with `url` set to it and
the images passed through `files` (published path -> local file).

| page | URL | source | images (published path <- how to make it) |
|---|---|---|---|
| CropCode Figures 1 and 2 | https://claude.ai/artifact/8H94wSuSYCju5zkSYLB1oS | `fig12_page.html` | `fig1_overview.png`, `fig2_stages.png` <- 400-dpi renders of `paper/figs/fig1_overview.pdf`, `paper/figs/fig2_stages.pdf` (PyMuPDF `get_pixmap(dpi=400)`) |
| E10 Real-Row Figures | https://claude.ai/artifact/3YXDLMKWedYMr4MbqJS5EP | `e10_page.html` | `fig_e10_real.png`, `fig_e10_real_sorghum.png` <- 170-dpi renders of the PDFs in `paper/figs/`; `fig_e10_height.png` <- `paper/figs/fig_e10_height.png`; `gallery/<name>.png` <- `paper/figs/e10_gallery/<name>.png` (20 tiles, listed in `index.json`) |

The figures themselves come from `figures/make_fig12_assets.py` + `figures/make_fig12_saga.py` (Figures 1 and 2),
`figures/make_e10_fig.py`, `figures/make_e10_height_fig.py` and `figures/make_e10_gallery.py` (E10).
Table numbers quoted on the Figures 1 and 2 page follow the paper as of 2026-11-11 (the encoder comparison was
Table 10 then; it is Table 11 since the E10 detail table was added).
