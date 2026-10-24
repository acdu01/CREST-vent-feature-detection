"""Per-model results, one explicitly selected primary model per sequence, means."""
from collections import defaultdict
import math
import statistics

MODEL_FIELDS = ["sequence", "variant", "quality", "model_id", "primary", "status",
                "input_images", "registered_images", "points", "mean_track_length",
                "mean_reprojection_error", "registration_rate", "laplacian_mean"]
AGGREGATE_FIELDS = ["variant", "quality", "runs", "successful_runs", "failed_runs",
                    "mean_points", "mean_track_length", "mean_reprojection_error",
                    "mean_registration_rate"]


def finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def model_metrics(models, sequence, variant, input_images, quality, laplacian_mean):
    base = {"sequence": sequence, "variant": variant, "input_images": input_images,
            "quality": quality, "laplacian_mean": laplacian_mean}
    if not models:
        return [dict(base, model_id=None, primary=True, status="no_model",
                     registered_images=0, points=0, mean_track_length=None,
                     mean_reprojection_error=None, registration_rate=0.0)]
    primary = max(models, key=lambda k: (models[k].num_reg_images(), models[k].num_points3D(), -int(k)))
    result = []
    for key, model in sorted(models.items()):
        count = model.num_points3D()
        result.append(dict(base, model_id=int(key), primary=key == primary, status="success",
                           registered_images=model.num_reg_images(), points=count,
                           mean_track_length=finite_or_none(model.compute_mean_track_length()) if count else None,
                           mean_reprojection_error=finite_or_none(model.compute_mean_reprojection_error()) if count else None,
                           registration_rate=model.num_reg_images() / input_images))
    return result


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["primary"]:
            groups[(row["variant"], row["quality"])].append(row)
    result = []
    for (variant, quality), group in sorted(groups.items()):
        valid = [r for r in group if r["status"] == "success"]
        output = dict(variant=variant, quality=quality, runs=len(group), successful_runs=len(valid),
                      failed_runs=len(group)-len(valid))
        for source, target in [("points", "mean_points"), ("mean_track_length", "mean_track_length"),
                               ("mean_reprojection_error", "mean_reprojection_error"),
                               ("registration_rate", "mean_registration_rate")]:
            values = [r[source] for r in valid if r[source] is not None]
            output[target] = statistics.mean(values) if values else None
        result.append(output)
    return result
