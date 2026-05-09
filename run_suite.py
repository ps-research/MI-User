#!/usr/bin/env python3
"""
MI USER — COMPLETE EXPERIMENT SUITE
====================================
"Mechanistic Interpretability of User Modelling in Language Models"

4 Experiments across 6 user properties:
  1. Feature Discovery — contrastive activation differentials
  2. Circuit Tracing — attribution graphs for top user-model features
  3. Causal Validation — generation with ablation + behavioral metrics
  4. Formation Timeline — when do user-model features first activate?

Special handling for P5 (multi-turn sycophancy escalation).

Usage:
  python run_suite.py --quick          # 1 property (~20 min)
  python run_suite.py --property P1    # specific property
  python run_suite.py                  # all 6 properties (~2 hr)

Requires: Gemma 3 1B IT, CLT-IT (affine)
GPU Memory: ~17 GB
"""

import sys, os, json, time, argparse, traceback, logging
from datetime import datetime

WORKSPACE = "/workspace"
INFRA = os.path.join(WORKSPACE, "Gemma-Scope-2-Study")
PROJECT = os.path.join(WORKSPACE, "MI-User")
sys.path.insert(0, INFRA)
sys.path.insert(0, PROJECT)

import torch
import numpy as np
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt, GEMMA3_1B_NUM_LAYERS
from src.hooks import gather_clt_activations
from src.attribution import build_attribution_graph, prune_graph, compute_graph_metrics, save_graph
from src.generation import generate_with_ablation, compare_generations, compute_behavioral_metrics

from prompts.prompts_mi_user import (
    PROPERTIES, get_all_property_ids, get_property_info, get_property_pairs,
)

CACHE = os.path.join(INFRA, "cache")
OUT = os.path.join(PROJECT, "outputs")
os.makedirs(OUT, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUT, "experiment.log"), mode="w"),
    ],
)
log = logging.getLogger("mi_user")


def save_results(data, filename):
    path = os.path.join(OUT, filename)
    data["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")


# ============================================================
# EXPERIMENT 1: Feature Discovery
# ============================================================

def run_exp1_discovery(model, tokenizer, clt, property_ids):
    """Find differentially active features for each user property."""
    log.info("=" * 70)
    log.info("EXPERIMENT 1: FEATURE DISCOVERY")
    log.info("=" * 70)

    results = {"properties": {}}

    for prop_id in property_ids:
        info = get_property_info(prop_id)
        pairs = get_property_pairs(prop_id)
        log.info(f"\n  Property: {info['name']} ({info['high_label']} vs {info['low_label']})")

        all_diffs = {}  # (layer, feat) -> list of delta values

        for pair_idx, (high_text, low_text) in enumerate(pairs):
            log.info(f"    Pair {pair_idx+1}/{len(pairs)}...")

            try:
                for variant, text in [("high", high_text), ("low", low_text)]:
                    inputs = tokenizer.encode(text, return_tensors="pt",
                                              add_special_tokens=True).to("cuda")
                    clt_in, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
                    if next(clt.parameters()).dtype == torch.float16:
                        clt_in = clt_in.half()
                    features = clt.encode(clt_in)
                    last = features[-1].float().cpu()  # (26, 10080)

                    if variant == "high":
                        feats_high = last
                    else:
                        feats_low = last

                delta = feats_high - feats_low

                for layer in range(delta.shape[0]):
                    nz = (delta[layer].abs() > 10).nonzero(as_tuple=True)[0]
                    for idx in nz:
                        key = (layer, idx.item())
                        if key not in all_diffs:
                            all_diffs[key] = []
                        all_diffs[key].append(delta[layer, idx].item())

                log.info(f"      {len(all_diffs)} candidate features so far")

            except Exception as e:
                log.error(f"    Pair {pair_idx} failed: {e}")
                log.error(traceback.format_exc())

        # Rank features
        ranked = []
        for (layer, feat_idx), deltas in all_diffs.items():
            ranked.append({
                "layer": layer, "feature_idx": feat_idx,
                "mean_abs_diff": float(np.mean([abs(d) for d in deltas])),
                "mean_signed_diff": float(np.mean(deltas)),
                "std": float(np.std(deltas)) if len(deltas) > 1 else 0,
                "consistency": float(np.std(deltas) / (np.mean([abs(d) for d in deltas]) + 1e-8)),
                "n_pairs": len(deltas),
                "all_deltas": [float(d) for d in deltas],
            })
        ranked.sort(key=lambda x: x["mean_abs_diff"], reverse=True)

        # Layer distribution of top features
        top30_layers = [f["layer"] for f in ranked[:30]]
        layer_hist = {}
        for l in top30_layers:
            layer_hist[l] = layer_hist.get(l, 0) + 1

        results["properties"][prop_id] = {
            "name": info["name"],
            "high_label": info["high_label"],
            "low_label": info["low_label"],
            "n_pairs": len(pairs),
            "total_candidates": len(ranked),
            "top_features": ranked[:50],
            "layer_distribution_top30": layer_hist,
        }

        log.info(f"    Top 5 features:")
        for f in ranked[:5]:
            log.info(f"      L{f['layer']}/f{f['feature_idx']}: "
                     f"|Δ|={f['mean_abs_diff']:.0f}, signed={f['mean_signed_diff']:+.0f}")

    save_results(results, "exp1_feature_discovery.json")
    return results


# ============================================================
# EXPERIMENT 2: Circuit Tracing
# ============================================================

def run_exp2_circuits(model, tokenizer, clt, property_ids, discovery_results):
    """Build attribution graphs for top properties."""
    log.info("=" * 70)
    log.info("EXPERIMENT 2: CIRCUIT TRACING")
    log.info("=" * 70)

    results = {"circuits": {}}

    for prop_id in property_ids:
        info = get_property_info(prop_id)
        pairs = get_property_pairs(prop_id)
        if not pairs:
            continue

        high_text, low_text = pairs[0]
        top_feats = discovery_results.get("properties", {}).get(prop_id, {}).get("top_features", [])

        log.info(f"\n  {info['name']}: tracing circuits...")

        prop_results = {}
        for variant, text, label in [("high", high_text, info["high_label"]),
                                      ("low", low_text, info["low_label"])]:
            try:
                graph = build_attribution_graph(
                    model, clt, tokenizer, text,
                    top_k_output_tokens=10,
                    min_ff_edge_weight=50.0, min_fl_edge_weight=5.0)
                pruned = prune_graph(graph, top_k_edges_per_node=5,
                                    max_feature_nodes=40, min_edge_weight=10.0)
                metrics = compute_graph_metrics(pruned)

                # Check overlap with discovery features
                top_keys = {(f["layer"], f["feature_idx"]) for f in top_feats[:20]}
                graph_keys = {(n.layer, n.feature_idx) for n in pruned.feature_nodes.values()}
                overlap = top_keys & graph_keys

                prop_results[variant] = {
                    "label": label,
                    "n_feature_nodes": metrics.get("num_feature_nodes", 0),
                    "n_ff_edges": metrics.get("feature_to_feature_edges", 0),
                    "n_fl_edges": metrics.get("feature_to_logit_edges", 0),
                    "avg_path_length": metrics.get("avg_path_length", 0),
                    "max_path_length": metrics.get("max_path_length", 0),
                    "layer_distribution": metrics.get("layer_distribution", {}),
                    "user_feature_overlap": len(overlap),
                    "overlap_features": [list(x) for x in overlap],
                }

                # Save full graph
                save_graph(pruned, os.path.join(OUT, f"circuit_{prop_id}_{variant}.json"))
                log.info(f"    {label}: nodes={prop_results[variant]['n_feature_nodes']}, "
                         f"ff={prop_results[variant]['n_ff_edges']}, "
                         f"path={prop_results[variant]['avg_path_length']:.1f}, "
                         f"overlap={len(overlap)}")

            except Exception as e:
                log.error(f"    {label} circuit failed: {e}")
                log.error(traceback.format_exc())
                prop_results[variant] = {"error": str(e)}

        results["circuits"][prop_id] = prop_results

    save_results(results, "exp2_circuits.json")
    return results


# ============================================================
# EXPERIMENT 3: Causal Validation
# ============================================================

def run_exp3_causal(model, tokenizer, clt, property_ids, discovery_results):
    """Ablate user-model features and measure behavioral change."""
    log.info("=" * 70)
    log.info("EXPERIMENT 3: CAUSAL VALIDATION")
    log.info("=" * 70)

    results = {"interventions": {}}

    for prop_id in property_ids:
        info = get_property_info(prop_id)
        pairs = get_property_pairs(prop_id)
        top_feats = discovery_results.get("properties", {}).get(prop_id, {}).get("top_features", [])

        if not top_feats or not pairs:
            log.warning(f"  {prop_id}: no features or pairs, skipping")
            continue

        log.info(f"\n  {info['name']}: ablating top 30 features...")

        feature_specs = [(f["layer"], f["feature_idx"]) for f in top_feats[:30]]
        high_text = pairs[0][0]
        low_text = pairs[0][1]

        prop_results = {"feature_specs": [list(s) for s in feature_specs]}

        # Generate with ablation on HIGH prompt
        try:
            gen_result = generate_with_ablation(
                model, clt, tokenizer, high_text,
                feature_specs=feature_specs,
                max_new_tokens=100, amplification=3.0)
            metrics = compare_generations(gen_result)

            prop_results["high_prompt"] = {
                "clean_generation": gen_result.generation_clean[:500],
                "ablated_generation": gen_result.generation_intervened[:500],
                "metrics_clean": metrics["clean"],
                "metrics_ablated": metrics["intervened"],
                "metric_deltas": metrics["deltas"],
            }

            log.info(f"    HIGH clean: {gen_result.generation_clean[:80]}...")
            log.info(f"    HIGH ablated: {gen_result.generation_intervened[:80]}...")
            for key in ["word_count", "avg_word_length", "empathy_score",
                        "agreement_score", "refusal_score", "technical_score"]:
                c = metrics["clean"].get(key, 0)
                a = metrics["intervened"].get(key, 0)
                d = a - c
                if d != 0:
                    log.info(f"      {key}: {c} → {a} (Δ={d:+})")

        except Exception as e:
            log.error(f"    HIGH ablation failed: {e}")
            log.error(traceback.format_exc())
            prop_results["high_prompt"] = {"error": str(e)}

        # Also generate on LOW prompt (control: ablation should have less effect)
        try:
            gen_control = generate_with_ablation(
                model, clt, tokenizer, low_text,
                feature_specs=feature_specs,
                max_new_tokens=100, amplification=3.0)
            metrics_ctrl = compare_generations(gen_control)

            prop_results["low_prompt_control"] = {
                "clean_generation": gen_control.generation_clean[:500],
                "ablated_generation": gen_control.generation_intervened[:500],
                "metrics_clean": metrics_ctrl["clean"],
                "metrics_ablated": metrics_ctrl["intervened"],
                "metric_deltas": metrics_ctrl["deltas"],
            }

        except Exception as e:
            log.error(f"    LOW control failed: {e}")
            prop_results["low_prompt_control"] = {"error": str(e)}

        results["interventions"][prop_id] = prop_results

    save_results(results, "exp3_causal_validation.json")
    return results


# ============================================================
# EXPERIMENT 4: Formation Timeline
# ============================================================

def run_exp4_timeline(model, tokenizer, clt, property_ids, discovery_results):
    """Track when user-model features first activate across token positions."""
    log.info("=" * 70)
    log.info("EXPERIMENT 4: FORMATION TIMELINE")
    log.info("=" * 70)

    results = {"timelines": {}}

    for prop_id in property_ids:
        info = get_property_info(prop_id)
        pairs = get_property_pairs(prop_id)
        top_feats = discovery_results.get("properties", {}).get(prop_id, {}).get("top_features", [])

        if not top_feats or not pairs:
            continue

        track = [(f["layer"], f["feature_idx"]) for f in top_feats[:5]]
        log.info(f"\n  {info['name']}: tracking {len(track)} features")

        prop_results = {"tracked_features": [list(t) for t in track]}

        for vi, (high_text, low_text) in enumerate(pairs[:2]):  # First 2 pairs
            for variant, text, label in [("high", high_text, info["high_label"]),
                                          ("low", low_text, info["low_label"])]:
                try:
                    inputs = tokenizer.encode(text, return_tensors="pt",
                                              add_special_tokens=True).to("cuda")
                    tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

                    clt_in, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
                    if next(clt.parameters()).dtype == torch.float16:
                        clt_in = clt_in.half()
                    all_features = clt.encode(clt_in)

                    timelines = {}
                    for layer, feat_idx in track:
                        key = f"L{layer}_f{feat_idx}"
                        acts = all_features[:, layer, feat_idx].float().cpu().tolist()
                        timelines[key] = acts

                        # Find first activation
                        first = next((i for i, a in enumerate(acts) if a > 0), -1)
                        if first >= 0:
                            log.info(f"    {label} pair{vi} | {key}: fires at pos {first} "
                                     f"('{tokens[first] if first < len(tokens) else '?'}')")

                    result_key = f"pair{vi}_{variant}"
                    prop_results[result_key] = {
                        "label": label,
                        "tokens": tokens,
                        "timelines": timelines,
                    }

                except Exception as e:
                    log.error(f"    {label} timeline failed: {e}")

        results["timelines"][prop_id] = prop_results

    save_results(results, "exp4_formation_timeline.json")
    return results


# ============================================================
# MULTI-TURN: P5 Sycophancy Escalation
# ============================================================

def run_multiturn_p5(model, tokenizer, clt):
    """Track feature changes across escalating sycophancy pressure turns."""
    log.info("=" * 70)
    log.info("SPECIAL: P5 MULTI-TURN SYCOPHANCY ESCALATION")
    log.info("=" * 70)

    prop = PROPERTIES.get("P5_sycophancy_escalation")
    if not prop:
        log.warning("P5 not found in properties")
        return {}

    results = {"turns": []}

    for pair_idx, pair in enumerate(prop["pairs"]):
        high_text = pair["high"]
        low_text = pair["low"]

        log.info(f"\n  Pair {pair_idx+1}: {pair['notes']}")

        # Run control (neutral version)
        try:
            inputs_ctrl = tokenizer.encode(low_text, return_tensors="pt",
                                           add_special_tokens=True).to("cuda")
            clt_in, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs_ctrl)
            if next(clt.parameters()).dtype == torch.float16:
                clt_in = clt_in.half()
            feats_ctrl = clt.encode(clt_in)[-1].float().cpu()
        except Exception as e:
            log.error(f"  Control failed: {e}")
            continue

        # Run pressure version
        try:
            inputs_press = tokenizer.encode(high_text, return_tensors="pt",
                                            add_special_tokens=True).to("cuda")
            clt_in, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs_press)
            if next(clt.parameters()).dtype == torch.float16:
                clt_in = clt_in.half()
            feats_press = clt.encode(clt_in)[-1].float().cpu()
        except Exception as e:
            log.error(f"  Pressure failed: {e}")
            continue

        delta = feats_press - feats_ctrl
        flat_delta = delta.abs().flatten()
        top_vals, top_flat = flat_delta.topk(20)
        d_sae = delta.shape[1]

        top_features = []
        for flat_idx, val in zip(top_flat, top_vals):
            layer = flat_idx.item() // d_sae
            feat = flat_idx.item() % d_sae
            top_features.append({
                "layer": layer, "feature": feat,
                "delta": delta[layer, feat].item(),
            })

        # Generate with and without ablation
        feature_specs = [(f["layer"], f["feature"]) for f in top_features[:30]]
        try:
            gen = generate_with_ablation(
                model, clt, tokenizer, high_text,
                feature_specs=feature_specs,
                max_new_tokens=100, amplification=3.0)
            metrics = compare_generations(gen)

            results["turns"].append({
                "pair_idx": pair_idx,
                "notes": pair["notes"],
                "top_features": top_features[:20],
                "clean_generation": gen.generation_clean[:500],
                "ablated_generation": gen.generation_intervened[:500],
                "metrics_clean": metrics["clean"],
                "metrics_ablated": metrics["intervened"],
                "metric_deltas": metrics["deltas"],
            })

            log.info(f"    Agreement clean: {metrics['clean'].get('agreement_score', 0)}")
            log.info(f"    Agreement ablated: {metrics['intervened'].get('agreement_score', 0)}")

        except Exception as e:
            log.error(f"  Generation failed: {e}")

    save_results(results, "exp_p5_multiturn.json")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MI User — Complete Experiment Suite")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--property", type=str, default=None,
                        help="Run specific property (e.g., P1_vulnerability)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["discovery", "circuits", "causal", "timeline", "multiturn"])
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("MI USER — COMPLETE EXPERIMENT SUITE")
    log.info(f"  Time: {datetime.now().isoformat()}")
    log.info("=" * 70)

    # Load IT model
    log.info("\nLoading Gemma 3 1B IT...")
    model, tokenizer = load_gemma3_1b("it", device="cuda")

    log.info("Loading CLT-IT (affine)...")
    clt = load_clt(width="262k", l0="big", affine=True, variant="it",
                   device="cuda", half_precision=True, cache_dir=CACHE)

    log.info(f"GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    # Select properties
    if args.property:
        property_ids = [args.property]
    elif args.quick:
        property_ids = ["P1_vulnerability", "P3_manipulation_sophistication"]
    else:
        property_ids = get_all_property_ids()

    log.info(f"Properties: {property_ids}")
    t_start = time.time()

    # Exp 1: Discovery (always needed first)
    discovery = None
    if args.only is None or args.only == "discovery":
        discovery = run_exp1_discovery(model, tokenizer, clt, property_ids)
    else:
        # Load from saved
        disc_path = os.path.join(OUT, "exp1_feature_discovery.json")
        if os.path.exists(disc_path):
            with open(disc_path) as f:
                discovery = json.load(f)
            log.info("Loaded discovery results from disk")
        else:
            log.error("No discovery results found. Run discovery first.")
            return

    # Exp 2: Circuits
    if args.only is None or args.only == "circuits":
        run_exp2_circuits(model, tokenizer, clt, property_ids, discovery)

    # Exp 3: Causal validation
    if args.only is None or args.only == "causal":
        run_exp3_causal(model, tokenizer, clt, property_ids, discovery)

    # Exp 4: Timeline
    if args.only is None or args.only == "timeline":
        run_exp4_timeline(model, tokenizer, clt, property_ids, discovery)

    # P5 multi-turn
    if (args.only is None or args.only == "multiturn") and \
       ("P5_sycophancy_escalation" in property_ids or args.only == "multiturn"):
        run_multiturn_p5(model, tokenizer, clt)

    total = time.time() - t_start
    log.info("\n" + "=" * 70)
    log.info(f"MI USER COMPLETE — {total/60:.1f} min")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
