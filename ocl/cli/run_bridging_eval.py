#!/usr/bin/env python3
"""Script to run evaluations for the bridging-the-gap project."""
import argparse
import itertools
import pathlib
import subprocess

configs_by_mode = {
    "orig": ["evaluation/eval"],
    "coco_metrics": [
        "evaluation/slot_attention_vit/metrics_coco_ccrop",
        "evaluation/slot_attention_vit/metrics_coco_20k",
    ],
    "coco_clustering": [
        "evaluation/slot_attention_vit/clustering_coco",
        "evaluation/slot_attention_vit/clustering_coco_27",
        "evaluation/slot_attention_vit/clustering_coco_thingsandstuff",
    ],
    "coco_metrics_sa": ["evaluation/slot_attention_vit/metrics_coco_ccrop_sa"],
    "voc_metrics": [
        "evaluation/slot_attention_vit/metrics_voc2012_bboxes",
        "evaluation/slot_attention_vit/metrics_voc2012_masks_ccrop",
    ],
    "voc_metrics_sa": ["evaluation/slot_attention_vit/metrics_voc2012_masks_ccrop_sa"],
    "voc_clustering": [
        "evaluation/slot_attention_vit/clustering_voc2012",
    ],
    "kitti": ["evaluation/slot_attention_vit/metrics_kitti"],
    "movi_c": ["evaluation/slot_attention_vit/metrics_movi_c"],
    "movi_c_sa": ["evaluation/slot_attention_vit/metrics_movi_c_sa"],
    "movi_c_instance": ["evaluation/slot_attention_vit/metrics_movi_c_instance"],
    "movi_c_instance_sa": ["evaluation/slot_attention_vit/metrics_movi_c_instance_sa"],
    "movi_c_one_obj": ["evaluation/slot_attention_vit/metrics_movi_c_one_obj"],
    "movi_c_one_obj_sa": ["evaluation/slot_attention_vit/metrics_movi_c_one_obj_sa"],
    "movi_e": ["evaluation/slot_attention_vit/metrics_movi_e"],
    "movi_e_sa": ["evaluation/slot_attention_vit/metrics_movi_e_sa"],
}
configs_by_mode["coco"] = configs_by_mode["coco_metrics"] + configs_by_mode["coco_clustering"]
configs_by_mode["voc"] = configs_by_mode["voc_metrics"] + configs_by_mode["voc_clustering"]

reportname_by_config = {
    "evaluation/eval": "metrics",
    "evaluation/slot_attention_vit/metrics_coco_ccrop": "metrics_coco",
    "evaluation/slot_attention_vit/metrics_coco_20k": "metrics_coco_20k",
    "evaluation/slot_attention_vit/metrics_coco_ccrop_sa": "metrics_coco",
    "evaluation/slot_attention_vit/clustering_coco": "clustering_coco",
    "evaluation/slot_attention_vit/clustering_coco_27": "clustering_coco_27",
    "evaluation/slot_attention_vit/clustering_coco_thingsandstuff": "clustering_coco_thingsandstuff",  # noqa: E501
    "evaluation/slot_attention_vit/metrics_voc2012_bboxes": "metrics_voc2012_bboxes",
    "evaluation/slot_attention_vit/metrics_voc2012_masks_ccrop": "metrics_voc2012_masks",
    "evaluation/slot_attention_vit/metrics_voc2012_masks_ccrop_sa": "metrics_voc2012_masks",
    "evaluation/slot_attention_vit/clustering_voc2012": "clustering_voc2012",
    "evaluation/slot_attention_vit/metrics_kitti": "metrics_kitti",
    "evaluation/slot_attention_vit/metrics_movi_c": "metrics_movi_c",
    "evaluation/slot_attention_vit/metrics_movi_c_sa": "metrics_movi_c_sa",
    "evaluation/slot_attention_vit/metrics_movi_c_instance": "metrics_movi_c_instance",
    "evaluation/slot_attention_vit/metrics_movi_c_instance_sa": "metrics_movi_c_instance",
    "evaluation/slot_attention_vit/metrics_movi_c_one_obj": "metrics_movi_c_one_obj",
    "evaluation/slot_attention_vit/metrics_movi_c_one_obj_sa": "metrics_movi_c_one_obj",
    "evaluation/slot_attention_vit/metrics_movi_e": "metrics_movi_e",
    "evaluation/slot_attention_vit/metrics_movi_e_sa": "metrics_movi_e_sa",
}

for config in itertools.chain.from_iterable(configs for configs in configs_by_mode.values()):
    assert config in reportname_by_config, f"Config {config} has no report name defined"


def _fmt_overrides(overrides):
    res = []
    for s in overrides:
        res.append(f'"{s}"')
    return "[" + ",".join(res) + "]"


def _is_metric_conf(eval_conf):
    return "metrics" in eval_conf or eval_conf == "evaluation/eval"


def main(args):
    configs = []
    for path in args.paths:
        if path.name == "config.yaml":
            configs.append(args.path)
        else:
            configs += path.glob("**/config/config.yaml")
    valid_configs = [
        conf for conf in configs if sum(1 for _ in conf.parent.parent.glob("**/*.ckpt")) > 0
    ]
    if args.verbose:
        print(f"Valid model configurations: {valid_configs}")

    metric_postfix = ""
    metric_overrides = args.metric_overrides
    metric_train_overrides = []
    cluster_postfix = ""
    cluster_overrides = args.cluster_overrides
    cluster_train_overrides = []
    if args.cluster_features is not None:
        feats = args.cluster_features
        cluster_train_overrides.append(f"models.feature_extractor.aux_features={feats}")
        cluster_overrides += [
            f"features_path=feature_extractor.aux_features.{feats}",
        ]
        cluster_postfix = f"_{feats}"
    if args.attention == "slot_attention":
        attention_overrides = [
            "models.object_decoder.use_decoder_masks=false",
            "models.object_decoder.decoder.return_attention_weights=false",
        ]
        cluster_train_overrides.extend(attention_overrides)
        metric_train_overrides.extend(attention_overrides)

    if cluster_train_overrides:
        cluster_overrides.append(f"train_config_overrides={_fmt_overrides(cluster_train_overrides)}")
    if metric_train_overrides:
        metric_overrides.append(f"train_config_overrides={_fmt_overrides(metric_train_overrides)}")

    # Use dict.fromkeys to filter duplicates while maintaining order
    eval_confs = list(
        dict.fromkeys(itertools.chain.from_iterable(configs_by_mode[mode] for mode in args.modes))
    )
    if args.verbose:
        print(f"Running eval configs: {eval_confs}")

    def run(script, eval_conf, train_conf, reportname):
        if (train_conf.parent.parent / reportname).exists() and not args.refresh:
            if args.verbose:
                print(
                    f"Skipping eval config {eval_conf} with training config {train_conf} because"
                    f" report {reportname} already exists. Run with --refresh to override."
                )
            return

        cmd = (
            f"python -m {script} -cn {eval_conf} train_config_path={train_conf} "
            f"report_filename={reportname}"
        )

        overrides = metric_overrides if _is_metric_conf(eval_conf) else cluster_overrides
        if len(overrides) > 0:
            cmd += " " + " ".join(f"'{s}'" for s in overrides)
        print(f"Running `{cmd}`")
        if not args.dry:
            subprocess.run(cmd, shell=True)

    for train_conf in valid_configs:
        if args.clean:
            existing_reports = train_conf.parent.parent.glob("*.json")
            for report in existing_reports:
                if args.verbose:
                    print(f"Removing existing report {report}")
                report.unlink()

        for eval_conf in eval_confs:
            for iter in range(args.repeats):
                script = (
                    "ocl.cli.eval" if _is_metric_conf(eval_conf) else "ocl.cli.eval_cluster_metrics"
                )
                reportname = reportname_by_config[eval_conf]
                postfix = metric_postfix if _is_metric_conf(eval_conf) else cluster_postfix
                if args.postfix:
                    postfix += f"_{args.postfix}"
                if args.repeats > 1:
                    postfix += f".{iter}"
                reportname = f"{reportname}{postfix}.json"

                run(script, eval_conf, train_conf, reportname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--dry", action="store_true")
    parser.add_argument("-r", "--refresh", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Remove all jsons before running")
    parser.add_argument("--repeats", type=int, default=1, help="Amount of repetitions")
    parser.add_argument("--postfix", type=str, help="String to add to report name")
    parser.add_argument("--attention", choices=["decoder", "slot_attention"], default="decoder")
    parser.add_argument("--cluster-features", type=str)
    parser.add_argument("--cluster-overrides", nargs="+", type=str, default=[])
    parser.add_argument("--metric-overrides", nargs="+", type=str, default=[])
    parser.add_argument("--modes", nargs="+", choices=list(configs_by_mode.keys()))
    parser.add_argument("paths", nargs="+", type=pathlib.Path)

    args = parser.parse_args()
    main(args)
