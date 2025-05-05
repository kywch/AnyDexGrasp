import os
import shutil
import json
import argparse
import glob
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_metric_from_filename(filename):
    """Extracts the metric name and value from filenames like 'checkpoint_best_f1_0.9_0.1481.pth'."""
    # Regex to capture metric name (potentially with dots and underscores) and metric value
    match = re.search(r"best_([a-zA-Z0-9_.]+)_(\d+\.\d+)\.pth$", filename)
    if match:
        try:
            metric_name = match.group(1)  # e.g., 'f1_0.9'
            metric_value = float(match.group(2))  # e.g., 0.1481
            return metric_name, metric_value
        except ValueError:
            logging.warning(f"Could not convert metric value in {filename} to float.")
            return None, -1.0
    # Fallback for older 'best.pth' format or if parsing fails
    match_simple = re.search(r"best\.pth$", filename)
    if match_simple:
        return "unknown", 0.0  # Assign a default low value if metric isn't in name
    return None, -1.0


def collect_best_models(base_log_dir, best_models_dir, overwrite_policy="metric"):
    """
    Scans experiment directories and copies the best model for each grasp type.

    Args:
        base_log_dir (str): The root directory containing experiment logs.
        best_models_dir (str): The target directory to save collected best models.
        overwrite_policy (str): How to handle existing models in best_models_dir.
                                'metric': Overwrite if the new model has a better metric.
                                'always': Always overwrite.
                                'never': Never overwrite.
    """
    logging.info(f"Scanning experiments in: {base_log_dir}")
    logging.info(f"Collecting best models into: {best_models_dir}")
    logging.info(f"Overwrite policy: {overwrite_policy}")

    os.makedirs(best_models_dir, exist_ok=True)
    collected_models = {}  # Store best found model path and metric for each type

    for experiment_dir in os.listdir(base_log_dir):
        exp_path = os.path.join(base_log_dir, experiment_dir)
        if not os.path.isdir(exp_path):
            continue

        config_path = os.path.join(exp_path, "config.json")
        checkpoint_subdir = os.path.join(exp_path, "checkpoints")

        if not os.path.exists(config_path) or not os.path.exists(checkpoint_subdir):
            logging.warning(f"Skipping {experiment_dir}: Missing config.json or checkpoints subdir.")
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            gripper_type = config.get("gripper_type")
            grasp_type_id = config.get("train_multifinger_type")
            checkpoint_metric_name = config.get("checkpoint_metric", "f1_0.9")  # Get metric used

            if gripper_type is None or grasp_type_id is None:
                logging.warning(f"Skipping {experiment_dir}: Missing gripper_type or train_multifinger_type in config.")
                continue

            # Find the best checkpoint file (handles naming convention)
            best_checkpoint_pattern = os.path.join(checkpoint_subdir, "*best_*.pth")
            best_files = glob.glob(best_checkpoint_pattern)

            # Fallback if specific metric name isn't used
            if not best_files:
                best_checkpoint_pattern_simple = os.path.join(checkpoint_subdir, "best.pth")
                best_files = glob.glob(best_checkpoint_pattern_simple)

            if not best_files:
                logging.warning(f"Skipping {experiment_dir}: No 'best_*.pth' or 'best.pth' file found in checkpoints.")
                continue

            # If multiple 'best_*' files exist (e.g., best_f1_0.9.pth, best_recall_0.8.pth),
            # prioritize the one matching the config's checkpoint_metric if possible.
            best_checkpoint_path = None
            best_metric_val = -1.0

            for file_path in best_files:
                filename = os.path.basename(file_path)
                metric_name, metric_value = parse_metric_from_filename(filename)

                # Prioritize the metric specified in the config
                if metric_name == checkpoint_metric_name:
                    if metric_value > best_metric_val:
                        best_metric_val = metric_value
                        best_checkpoint_path = file_path
                # If no match yet, take the first one found with a valid metric
                elif best_checkpoint_path is None and metric_name is not None:
                    best_metric_val = metric_value
                    best_checkpoint_path = file_path

            # If still no best path found (e.g., only 'best.pth' exists), use that
            if best_checkpoint_path is None and best_files:
                best_checkpoint_path = best_files[0]
                _, best_metric_val = parse_metric_from_filename(os.path.basename(best_checkpoint_path))

            if best_checkpoint_path is None:
                logging.warning(f"Could not determine best checkpoint for {experiment_dir}")
                continue

            model_key = (gripper_type, grasp_type_id)

            # Check if we already found a model for this type and compare metrics
            if model_key in collected_models:
                existing_metric_val = collected_models[model_key]["metric"]
                if best_metric_val > existing_metric_val:
                    logging.info(
                        f"Found better model for {model_key} in {experiment_dir} (Metric: {best_metric_val:.4f} > {existing_metric_val:.4f})"
                    )
                    collected_models[model_key] = {"path": best_checkpoint_path, "metric": best_metric_val}
                else:
                    logging.debug(f"Existing model for {model_key} is better or equal. Skipping {experiment_dir}.")
            else:
                logging.info(f"Found best model for {model_key} in {experiment_dir} (Metric: {best_metric_val:.4f})")
                collected_models[model_key] = {"path": best_checkpoint_path, "metric": best_metric_val}

        except Exception as e:
            logging.error(f"Error processing {experiment_dir}: {e}")

    # --- Copy/Link the best models found ---
    logging.info("\nCopying selected best models...")
    for (gripper_type, grasp_type_id), model_info in collected_models.items():
        source_path = model_info["path"]
        metric_val = model_info["metric"]

        target_gripper_dir = os.path.join(best_models_dir, gripper_type, str(grasp_type_id))
        os.makedirs(target_gripper_dir, exist_ok=True)
        target_filename = f"type_{grasp_type_id}_{checkpoint_metric_name}_{metric_val:.4f}.pth"
        target_path = os.path.join(target_gripper_dir, target_filename)

        should_copy = False
        if not os.path.exists(target_path):
            should_copy = True
            logging.info(f"Copying: {(gripper_type, grasp_type_id)} -> {target_path}")
        elif overwrite_policy == "always":
            should_copy = True
            logging.info(f"Overwriting: {(gripper_type, grasp_type_id)} -> {target_path}")
        elif overwrite_policy == "metric":
            # We already selected the best based on metric, so if it exists,
            # it means the one found earlier was better or equal.
            # However, to be robust, let's re-check (in case script run multiple times)
            # This requires loading the checkpoint or storing metric in filename reliably.
            # For simplicity with current filename convention:
            logging.warning(
                "Overwrite policy 'metric' selected, but reliable comparison without loading checkpoint is hard. Overwriting based on scan result."
            )
            # Re-evaluate based on filename if possible
            _, existing_target_metric = parse_metric_from_filename(
                target_filename
            )  # This won't work well as target name doesn't have metric
            # A better approach for 'metric' policy would be to store metric IN the target filename
            # e.g., type_{grasp_type_id}_f1_{metric_val:.4f}.pth
            # Or load the checkpoint to compare. Let's stick to simple overwrite for now.
            should_copy = True
            logging.info(f"Overwriting (policy 'metric'): {model_key} -> {target_path}")

        elif overwrite_policy == "never":
            logging.info(f"Skipping copy (exists): {model_key} -> {target_path}")
            should_copy = False

        if should_copy:
            try:
                # shutil.copy2 preserves metadata like modification time
                shutil.copy2(source_path, target_path)
                # Or use symlink:
                # if os.path.exists(target_path): os.remove(target_path)
                # os.symlink(os.path.abspath(source_path), target_path)
            except Exception as e:
                logging.error(f"Failed to copy {source_path} to {target_path}: {e}")

    logging.info("Collection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect best models from training experiments.")
    parser.add_argument(
        "--base_log_dir",
        default="./experiments/",
        # required=True,
        help="Root directory containing experiment logs (e.g., ./experiments/grasp_decision/)",
    )
    parser.add_argument(
        "--best_models_dir",
        # required=True,
        default="./best_models/",
        help="Directory to save collected best models (e.g., ./best_models/grasp_decision/)",
    )
    parser.add_argument(
        "--overwrite_policy",
        default="metric",
        choices=["metric", "always", "never"],
        help="Policy for overwriting existing models.",
    )

    script_args = parser.parse_args()

    collect_best_models(script_args.base_log_dir, script_args.best_models_dir, script_args.overwrite_policy)
