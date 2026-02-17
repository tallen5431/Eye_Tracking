# eye_pipeline/report.py
import statistics as stats
from eye_ops import settings as S

def ms(x):
    return 1000.0 * float(x)

def _mean(lst):
    return stats.mean(lst) if lst else 0.0

def print_timing_summary(timings, total_runtime):
    # Backwards-compatible: if old timing dict is passed in, fall back to fine-only
    has_2pass = "total_2pass" in timings and "coarse_total" in timings and "fine_total" in timings and "crop" in timings

    print("\n========== PIPELINE TIMING ==========")
    print(f"STEP2_MODE: {S.STEP2_MODE}")
    print(f"DOWNSCALE (coarse): {S.DOWNSCALE}")
    print(f"DOWNSCALE_FINE:     {S.DOWNSCALE_FINE}")
    print(f"PAD_PX={S.PAD_PX}, PAD_REL={S.PAD_REL}")
    print(f"Images processed: {len(next(iter(timings.values()))) if timings else 0}")
    print(f"Total runtime (batch wall): {total_runtime:.4f} sec")

    if not timings:
        return

    if has_2pass:
        avg_total = _mean(timings["total_2pass"])
        print("\n========== 2-PASS TOTAL (COARSE + CROP + FINE) ==========")
        print(f"Avg per image: {ms(avg_total):.2f} ms")
        print(f"Approx FPS: {1.0 / avg_total:.2f}" if avg_total > 0 else "Approx FPS: inf")

        avg_coarse = _mean(timings["coarse_total"])
        avg_crop = _mean(timings["crop"])
        avg_fine = _mean(timings["fine_total"])

        print("\n--- Major Breakdown (avg ms) ---")
        print(f"Coarse total {ms(avg_coarse):8.3f}")
        print(f"Crop step   {ms(avg_crop):8.3f}   (bbox map + padding + slice)")
        print(f"Fine total  {ms(avg_fine):8.3f}")
        print(f"2-pass sum  {ms(avg_coarse + avg_crop + avg_fine):8.3f}")

        # Compute-only (exclude viz draw) — good proxy for “production”
        coarse_viz = _mean(timings.get("coarse_viz", []))
        fine_viz = _mean(timings.get("fine_viz", []))
        coarse_ell = _mean(timings.get("coarse_ellipse", []))
        fine_ell = _mean(timings.get("fine_ellipse", []))

        compute_only = (avg_coarse - coarse_viz) + avg_crop + (avg_fine - fine_viz)
        compute_no_viz_no_ellipse = (avg_coarse - coarse_viz - coarse_ell) + avg_crop + (avg_fine - fine_viz - fine_ell)

        print("\n--- Production-ish (avg ms) ---")
        print(f"No viz draw (keep ellipse): {ms(compute_only):.3f}")
        print(f"No viz + no ellipse:        {ms(compute_no_viz_no_ellipse):.3f}")

        # Optional: show detailed step breakdowns (coarse+fine)
        print("\n--- Coarse step breakdown (avg ms) ---")
        for k in ["roi", "downscale", "step2", "threshold", "pick", "ellipse", "viz", "total"]:
            key = f"coarse_{k}"
            if key in timings:
                print(f"{key:14s} {ms(_mean(timings[key])):.3f}")

        print("\n--- Fine step breakdown (avg ms) ---")
        for k in ["roi", "downscale", "step2", "threshold", "pick", "ellipse", "viz", "total"]:
            key = f"fine_{k}"
            if key in timings:
                print(f"{key:14s} {ms(_mean(timings[key])):.3f}")

    else:
        # Old fine-only behavior
        print("\n========== PIPELINE TIMING (FINE PASS / LEGACY) ==========")
        avg_total = _mean(timings.get("total", []))
        print(f"Avg per image: {ms(avg_total):.2f} ms")
        print(f"Approx FPS: {1.0 / avg_total:.2f}" if avg_total > 0 else "Approx FPS: inf")

        print("\n--- Step Breakdown (avg ms) ---")
        for key in ["roi", "downscale", "step2", "threshold", "pick", "ellipse", "viz"]:
            if key in timings:
                print(f"{key.capitalize():10s} {ms(_mean(timings[key])):.3f}")
