"""Command line interface, for batch runs and for scripting the same pipeline.

    python -m somtrack run spots_*.csv --pixel-size 1.6 --frame-interval 0.05 -o out
    python -m somtrack run features.csv --features-table -o out --som supervised
    python -m somtrack demo ./demo_data
    python -m somtrack scan spots_*.csv --pixel-size 1.6 -o out
    python -m somtrack scan features.csv --features-table --method tsne -o out
    python -m somtrack methods
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from .config import AnalysisConfig


def _expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out.extend(hits or [p])
    return out


def _progress(message: str, fraction: float) -> None:
    bar = int(fraction * 30)
    sys.stdout.write(f"\r[{'#' * bar}{'.' * (30 - bar)}] {message[:60]:<60}")
    sys.stdout.flush()
    if fraction >= 1.0:
        sys.stdout.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="somtrack", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse tables and write the full figure set")
    run.add_argument("inputs", nargs="+", help="coordinate tables, or one feature table")
    run.add_argument("--features-table", action="store_true",
                     help="treat the input as a ready-made one-row-per-sample table")
    run.add_argument("-o", "--out", default="somtrack_output")
    run.add_argument("--pixel-size", type=float, default=1.0)
    run.add_argument("--length-unit", default="um")
    run.add_argument("--frame-interval", type=float, default=0.1)
    run.add_argument("--time-unit", default="s")
    run.add_argument("--min-spots", type=int, default=20)
    run.add_argument("--trim", type=float, default=0.6, help="robust trim fraction")
    run.add_argument("--som", default="batch",
                     choices=["batch", "supervised", "relevance", "online", "growing"])
    run.add_argument("--epochs", type=int, default=200)
    run.add_argument("--map", default="", metavar="WxH",
                     help="fixed map size, e.g. 12x8 (default: 5*sqrt(N) rule)")
    run.add_argument("--label-weight", type=float, default=0.35)
    run.add_argument("--scaler", default="zscore",
                     choices=["zscore", "minmax", "robust", "rank", "none"])
    run.add_argument("--toroidal", action="store_true")
    run.add_argument("--projections", default="",
                     help="comma-separated projection keys, e.g. pca,pacmap,umap. "
                          "'somtrack methods' lists them; the default is every "
                          "unsupervised method that is installed.")
    run.add_argument("--allow-supervised", action="store_true",
                     help="also run the LDA / PLS-DA / SVM projections. They "
                          "separate groups by construction, so their figures carry "
                          "that caveat and are drawn out-of-fold as well.")
    run.add_argument("--unit", default="sample", choices=["sample", "replicate"],
                     help="the experimental unit. 'replicate' averages within each "
                          "replicate before testing, which is the conservative "
                          "choice when several samples share a dish or a session.")
    run.add_argument("--no-blocks", action="store_true",
                     help="ignore the replicate column when permuting and "
                          "cross-validating. Not recommended: it inflates "
                          "significance whenever samples share a replicate.")
    run.add_argument("--distance", default="euclidean",
                     choices=["euclidean", "correlation", "cityblock", "cosine"],
                     help="distance used by PERMANOVA, PERMDISP and MDS")
    run.add_argument("--classifier", default="lda_shrinkage",
                     choices=["lda_shrinkage", "svm_linear", "plsda", "logistic",
                              "random_forest"])
    run.add_argument("--permutations", type=int, default=999)
    run.add_argument("--no-stats", action="store_true",
                     help="skip the statistics layer entirely")
    run.add_argument("--report", default="core", choices=["core", "full"],
                     help="'core' writes about a dozen figures, 'full' writes all")
    run.add_argument("--cvd-proof", action="store_true",
                     help="also export colour-vision-deficiency and greyscale "
                          "versions of the leading figures")
    run.add_argument("--no-tsne", action="store_true")
    run.add_argument("--no-umap", action="store_true")
    run.add_argument("--no-mp4", action="store_true")
    run.add_argument("--metrics", default="", help="comma-separated metric names")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--config", default="", help="load an AnalysisConfig JSON first")

    scan = sub.add_parser("scan",
                          help="grid-search hyper-parameters in parallel, for the "
                               "SOM or for any projection method")
    scan.add_argument("inputs", nargs="+")
    scan.add_argument("--features-table", action="store_true")
    scan.add_argument("-o", "--out", default="somtrack_scan")
    scan.add_argument("--pixel-size", type=float, default=1.0)
    scan.add_argument("--frame-interval", type=float, default=0.1)
    scan.add_argument("--method", default="som",
                      help="'som' (the default) or a projection key such as tsne, "
                           "umap or pacmap")
    scan.add_argument("--criterion", default="dubious_fraction",
                      choices=["dubious_fraction", "rnx_auc", "trustworthiness",
                               "group_silhouette"],
                      help="what to score each setting by. The default counts the "
                           "points the projection places unreliably, which is the "
                           "criterion of Xia, Lee & Li (2024).")
    scan.add_argument("--param", action="append", default=[], metavar="NAME=V1,V2",
                      help="parameter grid, repeatable. Omit it to use the grid "
                           "the method declares for itself.")
    scan.add_argument("--epochs", default="50,100,200,400")
    scan.add_argument("--algorithms", default="batch,supervised,relevance")
    scan.add_argument("--time-constants", default="0.1,0.25,0.5")
    scan.add_argument("--jobs", type=int, default=-1)

    demo = sub.add_parser("demo", help="write a synthetic four-treatment dataset")
    demo.add_argument("out", nargs="?", default="somtrack_demo")
    demo.add_argument("--tracks", type=int, default=18)
    demo.add_argument("--frames", type=int, default=320)
    demo.add_argument("--replicates", type=int, default=2)

    sub.add_parser("metrics", help="list every available metric")
    sub.add_parser("methods",
                   help="list every projection method and its parameters")
    return ap


# --------------------------------------------------------------------------
def _load_features(args, cfg: AnalysisConfig):
    from .io_tables import (build_feature_dataset, build_spot_dataset,
                            detect_spot_columns, guess_group_replicate,
                            load_table, numeric_columns)
    from .metrics import compute_features

    files = _expand(args.inputs)
    if getattr(args, "features_table", False):
        df = load_table(files[0])
        return build_feature_dataset(df, numeric_columns(df))

    cm = detect_spot_columns(load_table(files[0]))
    if not cm.required_ok():
        raise SystemExit(
            "Could not detect the track/X/Y/time columns automatically. "
            "Use the GUI to map them, or rename them to TrackMate's names.")
    groups, reps = [], []
    for f in files:
        g, r = guess_group_replicate(Path(f).stem)
        groups.append(g)
        reps.append(r)
    spots = build_spot_dataset(files, cm, groups, reps)
    print(f"{len(spots.frame)} spots, {spots.n_tracks} tracks, "
          f"{len(set(groups))} group(s)")
    selected = [m.strip() for m in args.metrics.split(",") if m.strip()] if getattr(
        args, "metrics", "") else None
    return compute_features(spots, cfg.track, selected=selected,
                            progress=lambda i, n: _progress(f"metrics {i}/{n}", i / n))


def cmd_run(args) -> int:
    from . import pipeline

    cfg = AnalysisConfig.from_json(args.config) if args.config else AnalysisConfig()
    t = cfg.track
    t.pixel_size = args.pixel_size
    t.length_unit = args.length_unit
    t.frame_interval = args.frame_interval
    t.time_unit = args.time_unit
    t.min_spots = args.min_spots
    t.trim_fraction = args.trim

    cfg.preprocess.scaler = args.scaler
    s = cfg.som
    s.algorithm = args.som
    s.epochs = args.epochs
    s.toroidal = args.toroidal
    s.label_weight = args.label_weight
    s.random_state = args.seed
    if args.map:
        w, h = args.map.lower().split("x")
        s.width, s.height = int(w), int(h)

    cfg.embedding.run_tsne = not args.no_tsne
    cfg.embedding.run_umap = not args.no_umap
    cfg.embedding.random_state = args.seed
    cfg.embedding.allow_supervised = args.allow_supervised
    if args.projections:
        cfg.embedding.methods = [m.strip() for m in args.projections.split(",")
                                 if m.strip()]

    st = cfg.stats
    st.enabled = not args.no_stats
    st.unit_of_analysis = args.unit
    st.use_replicate_blocks = not args.no_blocks
    st.distance = args.distance
    st.classifier = args.classifier
    st.n_permutations = args.permutations
    st.classification_permutations = args.permutations
    st.random_state = args.seed

    cfg.report.profile = args.report
    cfg.figure.cvd_proof = args.cvd_proof
    cfg.export.out_dir = Path(args.out)
    cfg.export.mp4 = not args.no_mp4
    if args.metrics:
        cfg.selected_features = [m.strip() for m in args.metrics.split(",") if m.strip()]

    features = _load_features(args, cfg)
    print(f"feature table: {len(features.frame)} samples x "
          f"{len(features.feature_names)} metrics")

    result = pipeline.run_analysis(features, cfg, progress=_progress)
    for w in result.warnings:
        print("  ! " + w)
    print("quality: " + ", ".join(f"{k}={v:.3f}" for k, v in result.quality.items()))

    if result.verdict is not None:
        print("\n" + "=" * 72)
        print(result.verdict.headline)
        for line in result.verdict.bullets():
            print("  - " + line)
        print("=" * 72 + "\n")

    manifest = pipeline.export_all(result, progress=_progress)
    print(f"wrote {len(manifest.figures)} figures and {len(manifest.tables)} tables "
          f"to {manifest.out_dir}")
    if cfg.report.enabled:
        print(f"conclusion:   {manifest.out_dir / 'RESULTS_REPORT.html'}")
        print(f"methods text: {manifest.out_dir / 'methods.txt'}")
    return 0


def cmd_scan(args) -> int:
    """Grid-search either the SOM or any registered projection method.

    Both write the same thing: a table with one row per setting, and the
    *plateau* -- every setting that scored within tolerance of the best.  The
    plateau is the useful output.  A wide plateau means the parameter does not
    matter for this data set, which is worth knowing and worth reporting; a
    narrow one means the result depends on the setting and must be stated.
    """
    import pandas as pd

    from .analysis import DataContext
    from .analysis.scan import CRITERIA, scan, scan_som
    from .citations import MethodsLog
    from .preprocess import prepare

    cfg = AnalysisConfig()
    cfg.track.pixel_size = args.pixel_size
    cfg.track.frame_interval = args.frame_interval
    features = _load_features(args, cfg)
    prep = prepare(features, cfg.preprocess)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = MethodsLog()

    grid = _parse_grid(args.param)

    if args.method == "som":
        grid = grid or {
            "algorithm": [a.strip() for a in args.algorithms.split(",") if a.strip()],
            "epochs": [int(e) for e in args.epochs.split(",")],
            "time_constant": [float(t) for t in args.time_constants.split(",")],
        }
        total = 1
        for v in grid.values():
            total *= len(v)
        print(f"scanning {total} SOM settings on {args.jobs} worker(s)...")
        res = scan_som(prep.X, cfg.som, grid, prep.group_codes, prep.n_groups,
                       n_jobs=args.jobs, log=log)
    else:
        ctx = DataContext.from_prepared(prep, random_state=cfg.embedding.random_state)
        criterion = args.criterion
        if CRITERIA[criterion][2] and not ctx.has_groups:
            print(f"'{criterion}' needs group labels; falling back to rnx_auc.")
            criterion = "rnx_auc"
        res = scan(ctx, args.method, grid or None, criterion=criterion,
                   n_jobs=args.jobs, log=log)
        print(f"scanned {res.n_cells} settings of {res.method_label} "
              f"by {criterion}.")

    path = out / f"{args.method}_parameter_scan.csv"
    res.rows.to_csv(path, index=False, encoding="utf-8-sig")

    wanted = res.keys + [res.criterion, "rnx_auc", "dubious_fraction",
                         "group_silhouette", "group_purity", "quantisation_error"]
    cols = list(dict.fromkeys(c for c in wanted if c in res.rows.columns))
    print()
    print(res.rows[cols].head(20).to_string(index=False))
    print()
    print(res.plateau_text())
    print(f"\nfull grid written to {path}")

    if res.coords:
        try:
            from matplotlib import pyplot as plt

            from . import viz

            viz.apply_style(cfg.figure)
            for name, fn in (("scan_surface", viz.plot_scan_surface),
                             ("scan_thumbnails",
                              lambda r, c: viz.plot_scan_thumbnails(r, prep, c))):
                panel = fn(res, cfg.figure)
                panel.fig.savefig(out / f"{args.method}_{name}.png", dpi=200,
                                  facecolor="white")
                plt.close(panel.fig)
            print(f"figures written to {out}")
        except Exception as exc:
            print(f"  ! scan figures skipped: {exc}")

    (out / "scan_methods.txt").write_text(
        "\n".join(e.sentence() + (f" [{e.param_text()}]" if e.param_text() else "")
                  for e in log.entries)
        + "\n\n" + "\n".join(c.formatted for c in log.references()) + "\n",
        encoding="utf-8")
    return 0


def _parse_grid(entries: list[str]) -> dict[str, list]:
    """``--param perplexity=5,10,30`` -> ``{"perplexity": [5.0, 10.0, 30.0]}``."""
    grid: dict[str, list] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise SystemExit(f"--param needs NAME=V1,V2 (got '{entry}')")
        name, values = entry.split("=", 1)
        parsed = []
        for v in values.split(","):
            v = v.strip()
            if not v:
                continue
            try:
                parsed.append(int(v) if v.lstrip("-").isdigit() else float(v))
            except ValueError:
                parsed.append(v)
        grid[name.strip()] = parsed
    return grid


def cmd_methods(_args) -> int:
    """List the projection methods, what they are for, and how to tune them."""
    from .analysis import FAMILY_LABELS, by_family
    from .citations import cite

    for family, specs in by_family().items():
        title = FAMILY_LABELS.get(family, family)
        print(f"\n{title}")
        print("-" * len(title))
        for spec in specs:
            mark = "" if spec.available() else "   [not installed]"
            print(f"\n  {spec.key:16s} {spec.label}{mark}")
            if spec.summary:
                print(f"    {spec.summary}")
            if spec.citations:
                print(f"    cite: {cite(*spec.citations)}")
            if spec.caveat:
                print(f"    note: {spec.caveat}")
            if not spec.available() and spec.install_hint:
                print(f"    {spec.install_hint}")
            for prm in spec.params:
                scan = (f"  scan: {', '.join(str(v) for v in prm.scan_default)}"
                        if prm.scan_default else "")
                print(f"      --param {prm.name}=...  "
                      f"({prm.kind}, default {prm.default}){scan}")
                if prm.help:
                    print(f"          {prm.help}")
    print("\nRun a subset with:  somtrack run ... --projections pca,pacmap")
    print("Scan one of them with:  somtrack scan ... --method tsne\n")
    return 0


def cmd_demo(args) -> int:
    from .demo import write_demo_dataset

    paths = write_demo_dataset(args.out, n_tracks=args.tracks, n_frames=args.frames,
                               replicates=args.replicates)
    print(f"wrote {len(paths)} files to {Path(args.out).resolve()}")
    for p in paths:
        print("  " + p.name)
    return 0


def cmd_metrics(_args) -> int:
    from .metrics import metrics_by_category

    for category, specs in metrics_by_category().items():
        print(f"\n{category}")
        print("-" * len(category))
        for s in specs:
            tag = " [v1.2]" if s.legacy else ""
            req = f" (needs {', '.join(s.requires)})" if s.requires else ""
            print(f"  {s.name:24s} {s.unit:14s} {s.description}{tag}{req}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return {"run": cmd_run, "scan": cmd_scan, "demo": cmd_demo,
            "metrics": cmd_metrics, "methods": cmd_methods}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
