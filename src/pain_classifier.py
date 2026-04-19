import math
from src.computations import compute_all_features, EVANGELISTA_STATS

BUCKETS = ["very_unlikely", "unlikely", "likely", "very_likely"]


def _is_nan(x):

    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False
def classify_feature(feature_name, value):
    if _is_nan(value):
        return {
            "vote": None,
            "z_control": float("nan"),
            "z_painful": float("nan"),
            "value": value,
        }
    
    stats = EVANGELISTA_STATS[feature_name]
    z_control = abs(value - stats["control_mean"]) / stats["control_sd"]
    z_painful = abs(value - stats["painful_mean"]) / stats["painful_sd"]

    vote = "painful" if z_painful < z_control else "control"

    return {
        "vote": vote,
        "z_control": z_control,
        "z_painful": z_painful,
        "value": value,
    }

def votes_to_bucket(painful_fraction):
    if painful_fraction == 0.0:
        return "very_unlikely"
    if painful_fraction <= 0.25:
        return "unlikely"
    if painful_fraction <= 0.50:
        return "likely"
    return "very_likely"

def classify_landmarks(points):
    features = compute_all_features(points)
    details = {name: classify_feature(name, value)
               for name, value in features.items()}
    
    valid_votes = [d["vote"] for d in details.values() if d["vote"] is not None]
    n_valid = len(valid_votes)

    if n_valid == 0:
        return {
            "bucket": None,
            "painful_fraction": None,
            "n_valid_features": 0,
            "features": features,
            "details": details,
        }
    
    n_painful = sum(1 for v in valid_votes if v == "painful")
    painful_fraction = n_painful / n_valid
    bucket = votes_to_bucket(painful_fraction)

    return {
            "bucket": bucket,
            "painful_fraction": painful_fraction,
            "n_valid_features": n_valid,
            "features": features,
            "details": details,
        }

def explain_classification(result):
    lines = []
    lines.append(f"Bucket: {result['bucket']}")
    if result["painful_fraction"] is not None:
        lines.append(f"Painful votes: {int(round(result['painful_fraction'] * result['n_valid_features']))}"
                     f" / {result['n_valid_features']}"
                     f"  ({result['painful_fraction']:.0%})")
    lines.append("")
    lines.append(f"  {'feature':26s} {'value':>8s}  {'z_ctrl':>7s}  {'z_pain':>7s}  vote")
    for name, d in result["details"].items():
        if d["vote"] is None:
            lines.append(f"  {name:26s} {'NaN':>8s}  {'—':>7s}  {'—':>7s}  skipped")
        else:
            lines.append(
                f"  {name:26s} {d['value']:8.3f}  "
                f"{d['z_control']:7.2f}  {d['z_painful']:7.2f}  {d['vote']}"
            )
    return "\n".join(lines)