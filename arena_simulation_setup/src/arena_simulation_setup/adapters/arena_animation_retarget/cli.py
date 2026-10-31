"""animation_retarget: SMPL-X motion files -> Arena Animation clips, in batches, whole-body IK on the GPU.

    ros2 run arena_simulation_setup animation_retarget SRC [SRC ...] -o OUT_DIR [options]

SRC: .npz files and/or directories (searched recursively for *.npz). Each motion becomes OUT_DIR/<name>/<name>.anim +
annotation.yaml, name = file stem (--name-from-path for path-based names when stems repeat). Point OUT_DIR at
arena_simulation_setup/assets/Common/Animation to play the clips in Arena right away.

Pipeline: worker processes load the files and run the rotation retarget (--jobs), the frames are gathered into chunks
of up to --chunk-frames and each chunk is solved in one batched IK on the GPU (--device), then written and scored.

--pair: every SRC is a two-person interaction (an .npz with p0_/p1_ key prefixes, e.g. the InterMask export, or a
directory with P1.npz + P2.npz, the Inter-X layout) and becomes two clips, <name>_a and <name>_b, retargeted together so
their contacts meet on Arena's bodies (see .pair). Each annotation gets a `pair` block: the
partner clip and where the partner stands (x forward, y left, relative heading).
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .clip import write_clip
from .retarget import Prepared, prepare, resample
from .smplx import load


def _find(srcs: list[Path]) -> list[Path]:
    out = []
    for s in srcs:
        if s.is_dir():
            out += sorted(p for p in s.rglob("*.npz") if p.is_file())
        elif s.is_file():
            out.append(s)
        else:
            raise SystemExit(f"retarget: no such file or directory: {s}")
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def _names(files: list[Path], srcs: list[Path], a: argparse.Namespace) -> list[str]:
    if a.name:
        return [a.name]
    if a.name_from_path:
        roots = [s.resolve() for s in srcs if s.is_dir()]

        def rel(p: Path) -> str:
            r = p.resolve()
            for root in roots:
                if r != root and r.is_relative_to(root):
                    return "_".join(r.relative_to(root).with_suffix("").parts)
            return p.stem

        stems = [rel(p) for p in files]
    else:
        stems = [p.stem for p in files]
    names = [a.prefix + s.replace(" ", "_") + a.suffix for s in stems]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"retarget: {len(dup)} clip names repeat (e.g. {', '.join(dup[:3])}); use --name-from-path")
    return names


def _find_pairs(srcs: list[Path]) -> list[Path]:
    from .pair import is_pair_file

    out = []
    for s in srcs:
        if s.is_dir() and is_pair_file(s):
            out.append(s)
        elif s.is_dir():
            dirs = sorted({p.parent for p in s.rglob("P1.npz") if (p.parent / "P2.npz").is_file()})
            files = sorted(p for p in s.rglob("*.npz") if p.parent not in dirs and p.name not in ("P1.npz", "P2.npz"))
            out += dirs + [f for f in files if is_pair_file(f)]
        elif s.is_file():
            out.append(s)
        else:
            raise SystemExit(f"retarget: no such file or directory: {s}")
    return out


def _run_pairs(a: argparse.Namespace, files: list[Path], names: list[str]) -> int:
    from . import ik
    from .pair import load_pair, pair_annotation, retarget_pairs, window

    say = (lambda *x: None) if a.quiet else print
    cfg = None if a.method == "none" else ik.IKConfig.preset(a.method)
    if cfg is not None and a.iters:
        cfg = cfg.with_(iters=a.iters)
    todo = [i for i, n in enumerate(names) if not (a.skip_existing and (a.out / f"{n}_a" / f"{n}_a.anim").exists())]
    say(f"retarget: {len(todo)} of {len(files)} pairs, method {a.method}, window {a.window}, {a.passes} contact passes")
    pairs, ok, failed, rows, t0 = [], [], [], [], time.time()
    for i in todo:
        try:
            pair = load_pair(files[i], up=a.up, fps=a.fps_in, model=a.model)
            window(*pair, a.window)
            pairs.append(pair)
            ok.append(i)
        except Exception as e:  # noqa: BLE001
            failed.append((files[i], f"{type(e).__name__}: {e}"))
            say(f"  skip {files[i]}: {e}")
    if pairs:
        res = retarget_pairs(pairs, window_mode=a.window, passes=a.passes, fps=a.fps, method=a.method, device=a.device, config=cfg, chunk_frames=a.chunk_frames)
        for i, (m0, _), r in zip(ok, pairs, res, strict=True):
            pn = (f"{names[i]}_a", f"{names[i]}_b")
            ce = " ".join(f"{k}={v:.1f}" for k, v in r.contact.items() if k.startswith("contact_m"))
            for role in (0, 1):
                note = " ".join(x for x in (a.note, f"method {a.method}, pair {'ab'[role]} of {names[i]}.") if x)
                d = write_clip(r.clips[role], a.out, pn[role], loop=a.loop, desc=a.desc, tags=a.tags, note=note, extra=pair_annotation(r, pn, role, m0.fps))
                rows.append({"name": pn[role], "src": str(files[i]), "frames": len(r.clips[role].t), "seconds": round(r.clips[role].duration, 3), **{k: round(v, 3) for k, v in r.contact.items()}})
                say(f"{d}  {len(r.clips[role].t)} frames" + (f"  |  contact error (cm) {ce}" if ce else ""))
    if a.report and rows:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"retarget: wrote {len(ok)} pairs ({2 * len(ok)} clips) in {time.time() - t0:.1f} s" + (f", {len(failed)} failed" if failed else ""))
    for f, e in failed[:20]:
        print(f"  failed {f}: {e}", file=sys.stderr)
    return 1 if failed else 0


def _prep_job(job: tuple[int, Path, str, float | None, str | None, bool]) -> tuple[int, Prepared | None, str | None]:
    i, path, up, fps_in, model, fold = job
    try:
        p = prepare(load(path, up=up, fps=fps_in, model=model), fold)
        return i, p, None
    except Exception as e:  # noqa: BLE001  (a bad file must not stop the batch)
        return i, None, f"{type(e).__name__}: {e}"


def _window(value: str) -> str | tuple[int, int]:
    if value in ("auto", "all"):
        return value
    lo, sep, hi = value.partition(":")
    if not (sep and lo.isdigit() and hi.isdigit() and int(lo) < int(hi)):
        raise argparse.ArgumentTypeError(f"{value!r}: use auto, all, or F0:F1 with F0 < F1")
    return int(lo), int(hi)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="animation_retarget", description="Retarget SMPL-X motion onto Arena's 40-DOF human wire.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("src", nargs="+", type=Path, help="SMPL-X .npz files and/or directories (recursive)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="Animation directory to write clip folders into")
    g = ap.add_argument_group("naming and annotation")
    g.add_argument("--name", help="clip name (single SRC file only)")
    g.add_argument("--name-from-path", action="store_true", help="name = path below the SRC directory, '/' -> '_'")
    g.add_argument("--prefix", default="", help="prepended to every clip name")
    g.add_argument("--suffix", default="", help="appended to every clip name")
    g.add_argument("--loop", action="store_true", help="mark the clips as looping")
    g.add_argument("--desc", help="annotation description (single SRC only; default: the name)")
    g.add_argument("--tag", action="append", dest="tags", help="annotation tag, repeatable (default: domain::common)")
    g.add_argument("--note", default="", help="free text for the annotation note")
    g.add_argument("--skip-existing", action="store_true", help="skip clips whose output .anim exists")
    g = ap.add_argument_group("source")
    g.add_argument("--fps-in", type=float, help="source rate, if a file has none (default 30)")
    g.add_argument("--up", choices=("auto", "y", "z"), default="auto", help="source world up axis (auto: y for Kimodo-style rotations, z for AMASS-style parameters)")
    g.add_argument("--model", type=Path, help="SMPL-X model .npz for the exact rest skeleton of each file's betas")
    g = ap.add_argument_group("retarget")
    g.add_argument("--method", choices=("ik", "rotation", "none"), default="ik", help="ik: whole-body IK; rotation: bone directions only; none: closed-form rotation retarget")
    g.add_argument("--device", default="auto", help="torch device for the IK (auto: cuda if available)")
    g.add_argument("--iters", type=int, help="L-BFGS iterations per chunk (default: the method's, 300)")
    g.add_argument("--chunk-frames", type=int, default=400_000, help="frames per batched IK solve (GPU memory)")
    g.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="processes loading/preparing files (--pair loads in this process)")
    g.add_argument("--fps", type=float, default=20.0, help="output rate (Arena's clip rate)")
    g.add_argument("--no-fold", action="store_true", help="keep the body pelvis-relative instead of folding pelvis tilt into waist/hips; the clip is annotated pelvis_tilt and Arena tilts the ped by root_quat (falls, lying)")
    g = ap.add_argument_group("two-person interactions")
    g.add_argument("--pair", action="store_true", help="each SRC is a two-person interaction -> <name>_a, <name>_b")
    g.add_argument("--window", type=_window, default="auto", help="source frames to keep: auto (touching frames, padded), all, or F0:F1")
    g.add_argument("--passes", type=int, default=2, help="contact IK passes (anchor targets from the previous pass)")
    g = ap.add_argument_group("report")
    g.add_argument("--report", type=Path, help="write per-clip fidelity metrics to this CSV")
    g.add_argument("--check", action="store_true", help="print the per-clip fidelity table")
    g.add_argument("--no-score", action="store_true", help="skip the fidelity check (faster)")
    g.add_argument("-q", "--quiet", action="store_true", help="only the final summary")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    if a.pair and (a.no_fold or a.check or a.no_score):
        ap.error("--pair takes no --no-fold (contacts are solved on upright bodies), --check or --no-score (pairs are scored by their contact error)")
    from . import fidelity, ik  # torch import after argument parsing: --help stays fast

    files = _find_pairs(a.src) if a.pair else _find(a.src)
    if not files:
        raise SystemExit("retarget: no .npz files found")
    if (a.name or a.desc) and len(files) > 1:
        raise SystemExit("retarget: --name/--desc need a single SRC file")
    if a.pair:
        return _run_pairs(a, files, _names(files, a.src, a))
    names = _names(files, a.src, a)
    todo = [i for i, n in enumerate(names) if not (a.skip_existing and (a.out / n / f"{n}.anim").exists())]
    cfg = None
    if a.method != "none":
        cfg = ik.IKConfig.preset(a.method)
        if a.iters:
            cfg = cfg.with_(iters=a.iters)
    dev = ik.pick_device(a.device)
    say = (lambda *x: None) if a.quiet else print
    say(f"retarget: {len(todo)} of {len(files)} files, method {a.method}, IK on {dev}, {a.jobs} loader processes")

    rows, failed, done, frames, t0 = [], [], 0, 0, time.time()
    pending: list[tuple[int, Prepared]] = []

    def flush():
        nonlocal done, frames
        if not pending:
            return
        preps = [p for _, p in pending]
        n = sum(len(p) for p in preps)
        t = time.time()
        sols = [p.init for p in preps] if a.method == "none" else ik.solve(preps, cfg, device=str(dev), chunk_frames=a.chunk_frames)
        say(f"  solved {len(preps)} clips / {n} frames in {time.time() - t:.1f} s")
        for (i, p), s in zip(pending, sols, strict=True):
            clip = resample(p, s, a.fps)
            note = " ".join(x for x in (a.note, f"method {a.method}.") if x)
            d = write_clip(clip, a.out, names[i], loop=a.loop, desc=a.desc, tags=a.tags, note=note)
            row = {"name": names[i], "src": str(files[i]), "frames": len(clip.t), "seconds": round(clip.duration, 3)}
            if not a.no_score:
                sc = fidelity.score(None, clip, prep=p)
                row.update({f"{k}_mean": round(v[0], 3) for k, v in sc.items()})
                row.update({f"{k}_p95": round(v[1], 3) for k, v in sc.items()})
                say(f"{d}  {len(clip.t)} frames  |  {fidelity.summary(sc)}")
                if a.check:
                    say(fidelity.table(sc))
            else:
                say(d)
            rows.append(row)
        done += len(pending)
        frames += n
        pending.clear()

    jobs = [(i, files[i], a.up, a.fps_in, a.model, not a.no_fold) for i in todo]
    budget = 0
    ctx = mp.get_context("spawn")  # never fork a process that may hold a CUDA context
    with ProcessPoolExecutor(max_workers=max(1, a.jobs), mp_context=ctx) as pool:
        for i, prep, err in pool.map(_prep_job, jobs, chunksize=4):
            if err:
                failed.append((files[i], err))
                say(f"  skip {files[i]}: {err}")
                continue
            assert prep is not None  # _prep_job returns a prep whenever it returns no error
            pending.append((i, prep))
            budget += len(prep)
            if budget >= a.chunk_frames:
                try:
                    flush()
                except Exception:  # noqa: BLE001
                    failed += [(files[j], traceback.format_exc(limit=1).strip()) for j, _ in pending]
                    pending.clear()
                budget = 0
    try:
        flush()
    except Exception:  # noqa: BLE001
        failed += [(files[j], traceback.format_exc(limit=1).strip()) for j, _ in pending]

    if a.report and rows:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    dt = time.time() - t0
    print(f"retarget: wrote {done} clips ({frames} source frames) in {dt:.1f} s" + (f", {len(failed)} failed" if failed else "") + (f", report {a.report}" if a.report and rows else ""))
    for f, e in failed[:20]:
        print(f"  failed {f}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
