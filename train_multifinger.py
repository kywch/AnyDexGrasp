import os
import time
import json
import pickle
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from AnyDexGrasp.dataset.multifinger_hand import MultifingerDataset, collate_fn, convert_data_to_gpu, filter_json_data
from AnyDexGrasp.models.minkowski_graspnet import MultifingerGraspSuccessPredictor
from AnyDexGrasp.utils.solvers import PolyLR
from AnyDexGrasp.models.loss import MultifingerType1Loss

# Set sharing strategy (keep as is)
torch.multiprocessing.set_sharing_strategy("file_system")


NUM_MULTIFINGER_TYPE = 1  # NOTE: training model for "each" grasp type
NUM_TWO_FINGER_DEPTH = 5

METRIC_KEYS = [
    "precision_0.5",
    "recall_0.5",
    "f1_0.5",
    "precision_0.7",
    "recall_0.7",
    "f1_0.7",
    "precision_0.9",
    "recall_0.9",
    "f1_0.9",
]

EVERY_INDICATOR_KEYS = ["acc_type_0.5", "acc_type_0.7", "acc_type_0.9"]  # Names for the 3 thresholds
GRIPPER_METRIC_KEYS = ["precision", "recall", "f1", "tp"]  # Metrics per gripper type


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Training routine for grasp decision models.")
    parser.add_argument("--gripper_type", default="Inspire", help="Gripper type (e.g., Inspire)")
    parser.add_argument(
        "--train_multifinger_type",
        type=int,
        default=-1,
        help="Multifinger grasp type variant for training. If -1, train all types.",
    )
    parser.add_argument(
        "--num_multifinger_depth",
        type=int,
        default=2,  # NOTE: the original used 4 depth levels, but here opting for 2
        help="Number of depth levels for multifinger grasp",
    )

    parser.add_argument(
        "--train_data_file",
        default="train_data/inspire_merged_939_0513.pkl",
        help="Path to the training data file",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,  # "inspire_test_100_0506.pkl",
        help="Path to the test data file",
    )
    parser.add_argument(
        "--train_split_ratio", type=float, default=0.9, help="Ratio of data to use for training (0.0 to 1.0)"
    )
    parser.add_argument("--log_dir", default="experiments", help="Directory to save logs and model checkpoints")

    parser.add_argument("--max_epoch", type=int, default=100, help="Total number of epochs to run")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Optimizer L2 weight decay")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers for dataloaders")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing log folder.")

    # Checkpointing
    parser.add_argument(
        "--checkpoint_metric",
        default="f1_0.9",
        help="Metric used to determine the best checkpoint (e.g., f1_0.9, recall_0.7)",
    )

    args = parser.parse_args()
    return args


def setup_logging(log_dir):
    """Configures logging to file and console."""
    logger = logging.getLogger()
    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    log_file = os.path.join(log_dir, "log_train.txt")
    file_handler = logging.FileHandler(log_file, mode="a")  # Append mode
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logging configured.")


def my_worker_init_fn(worker_id):
    """Worker init function for reproducibility."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def setup_dataloaders(config):
    """Creates and returns train and test dataloaders."""
    logging.info(f"Loading the training data from: {config.train_data_file}")
    try:
        if config.train_data_file.endswith("json"):
            with open(config.train_data_file, "r") as f:
                tmp_data = json.load(f)
        elif config.train_data_file.endswith("pkl"):
            with open(config.train_data_file, "rb") as f:
                tmp_data = pickle.load(f)["exp_data"]
        else:
            raise ValueError(f"Unsupported file type: {config.train_data_file}")

        train_data, trial_stats = filter_json_data(
            tmp_data, config.gripper_type, config.train_multifinger_type, num_depth=config.num_multifinger_depth
        )
        logging.info(f"Train data stats for each depth: {trial_stats}")

        # NOTE: check unique angle types
        angles = {}
        for v in train_data.values():
            if v["two_fingers_pose_angle_type"] not in angles:
                angles[v["two_fingers_pose_angle_type"]] = 1
            else:
                angles[v["two_fingers_pose_angle_type"]] += 1
        unique_angles = list(angles.keys())
        unique_angles.sort()
        print("Unique angle types:", unique_angles)

    except FileNotFoundError:
        logging.error(f"Data file not found: {config.train_data_file}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {config.train_data_file}")
        raise

    # If test data file is provided, then use it
    # If not or doesn't work, split the training data
    test_data = None
    if config.test_data_file is not None:
        logging.info(f"Loading the test data from: {config.test_data_file}")
        try:
            if config.test_data_file.endswith("json"):
                with open(config.test_data_file, "r") as f:
                    tmp_data = json.load(f)
            elif config.test_data_file.endswith("pkl"):
                with open(config.test_data_file, "rb") as f:
                    tmp_data = pickle.load(f)["exp_data"]
            else:
                raise ValueError(f"Unsupported file type: {config.test_data_file}")

            test_data, trial_stats = filter_json_data(
                tmp_data, config.gripper_type, config.train_multifinger_type, num_depth=config.num_multifinger_depth
            )
            logging.info(f"Test data stats for each depth: {trial_stats}")
        except FileNotFoundError:
            logging.error(f"Data file not found: {config.train_data_file}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from file: {config.train_data_file}")
            raise

    if test_data is None:
        # --- Split the data keys ---
        num_total = len(train_data)
        num_train = int(num_total * config.train_split_ratio)
        num_test = num_total - num_train

        if num_train == 0 or num_test == 0:
            logging.error(
                f"Train/Test split resulted in zero samples for one set. Train: {num_train}, Test: {num_test}. Check data and split ratio."
            )
            raise ValueError("Train/Test split resulted in zero samples.")

        # Use torch.utils.data.random_split for a robust split based on indices
        all_keys = list(train_data.keys())
        indices = list(range(num_total))
        train_indices, test_indices = torch.utils.data.random_split(indices, [num_train, num_test])

        test_data = {}
        for i in test_indices:
            test_data[all_keys[i]] = train_data.pop(all_keys[i])

    train_dataset = MultifingerDataset(data_dict=train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_dataset = MultifingerDataset(data_dict=test_data)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    logging.info(f"Train dataset length: {len(train_dataset)}")
    logging.info(f"Test dataset length: {len(test_dataset)}")
    return train_dataloader, test_dataloader


def setup_model_criterion_optimizer(config, device):
    """Initializes the model, criterion, optimizer, and scheduler."""
    logging.info("Initializing model, criterion, and optimizer...")
    model = MultifingerGraspSuccessPredictor(
        num_multifinger_type=NUM_MULTIFINGER_TYPE,
        num_multifinger_depth=config.num_multifinger_depth,
        num_two_finger_depth=NUM_TWO_FINGER_DEPTH,
    )
    model.to(device)

    criterion = MultifingerType1Loss(
        num_multifinger_type=NUM_MULTIFINGER_TYPE,
        num_multifinger_depth=config.num_multifinger_depth,
        num_two_finger_depth=NUM_TWO_FINGER_DEPTH,
        train_type=config.train_multifinger_type,
    )
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Assuming start_epoch is 0 for now, adjust if checkpoint loading is added
    lr_scheduler = PolyLR(optimizer, max_iter=config.max_epoch, power=0.99, last_step=-1)

    logging.info("Model, criterion, and optimizer initialized.")
    return model, criterion, optimizer, lr_scheduler


def aggregate_metrics(batch_metrics_list, batch_every_metrics_list, batch_weights):
    """Aggregates metrics over batches using weighted average."""
    weights = np.array(batch_weights)
    weights = weights / np.sum(weights)

    mean_indicators = np.sum(weights.reshape((-1, 1)) * np.array(batch_metrics_list), axis=0)
    mean_every_indicators = np.sum(weights.reshape((-1, 1, 1, 1)) * np.array(batch_every_metrics_list), axis=0)

    metrics_dict = {key: val for key, val in zip(METRIC_KEYS, mean_indicators)}
    # Add detailed metrics per gripper type if needed
    # for thresh_idx, thresh_key in enumerate(EVERY_INDICATOR_KEYS):
    #     for mt in range(mean_every_indicators.shape[2]): # num_multifinger_type
    #         gripper_metrics = mean_every_indicators[thresh_idx, mt, :]
    #         for metric_idx, metric_key in enumerate(GRIPPER_METRIC_KEYS):
    #              metrics_dict[f"{thresh_key}_gripper{mt}_{metric_key}"] = gripper_metrics[metric_idx]

    # Simplified version focusing on 0.9 threshold as in original logging
    for mt in range(mean_every_indicators.shape[1]):  # num_multifinger_type
        gripper_metrics_09 = mean_every_indicators[2][mt]  # Index 2 corresponds to 0.9 threshold
        for metric_idx, metric_key in enumerate(GRIPPER_METRIC_KEYS):
            metrics_dict[f"gripper{mt}_type{config.train_multifinger_type}_{metric_key}_0.9"] = gripper_metrics_09[
                metric_idx
            ]

    return metrics_dict


def run_epoch(model, dataloader, criterion, optimizer, device, epoch, is_training, writer, config):
    """Runs a single epoch of training or evaluation."""
    if is_training:
        model.train()
        prefix = "train"
    else:
        model.eval()
        prefix = "eval"

    epoch_losses = []
    epoch_indicator_detachs = []
    epoch_every_indicator_detachs = []
    epoch_batch_sizes = []
    data_time, net_time = 0.0, 0.0
    batch_start_time = time.time()

    for batch_idx, batch_data_label in enumerate(dataloader):
        # Measure data loading time
        toc = time.time()
        data_time += toc - batch_start_time

        # Skip small batches in training if they cause issues (e.g., BatchNorm)
        batch_size = batch_data_label["result"].shape[0]
        if is_training and batch_size <= 1:
            logging.warning(f"Skipping training batch {batch_idx} with size {batch_size}")
            batch_start_time = time.time()  # Reset timer for next batch
            continue

        batch_data_label = convert_data_to_gpu(batch_data_label, device)
        epoch_batch_sizes.append(batch_size)

        # Forward pass
        net_start_time = time.time()
        with torch.set_grad_enabled(is_training):
            # Model only needs the grasp_preds_features, which is the local geometry-related info
            grasp_preds_five_hand = model(batch_data_label["grasp_preds_features"])

            # Extra info is used for calculating loss
            # Prepare end_points dictionary, include necessary data only
            end_points = {}
            end_points["two_fingers_pose_depth_type"] = batch_data_label["two_fingers_pose_depth_type"]
            end_points["multifinger_pose_finger_type"] = batch_data_label["multifinger_pose_finger_type"]
            end_points["multifinger_pose_depth_type"] = batch_data_label["multifinger_pose_depth_type"]
            end_points["grasp_preds_features"] = batch_data_label["grasp_preds_features"]
            end_points["if_flip"] = batch_data_label["if_flip"]
            end_points["result"] = batch_data_label["result"]  # Grasp success 1, fail 0

            end_points["stage4_grasp_preds_five_hand"] = grasp_preds_five_hand

            loss, indicator, every_indicator = criterion(end_points)

        # Backward pass and optimization
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net_time += time.time() - net_start_time  # Measure network/computation time

        # Store metrics
        epoch_losses.append(loss.item())
        # Detach and move to CPU *before* appending to avoid memory leaks
        indicator_detach = [item.detach().cpu().item() for item in indicator]
        epoch_indicator_detachs.append(indicator_detach)

        every_indicator_detach = []
        for acc_type in every_indicator:
            gripper_types = []
            for gripper_type in acc_type:
                accs = [item.detach().cpu().item() for item in gripper_type]
                gripper_types.append(accs)
            every_indicator_detach.append(gripper_types)
        epoch_every_indicator_detachs.append(every_indicator_detach)

        # Log batch info periodically (optional)
        # if (batch_idx + 1) % 100 == 0:
        #     logging.debug(f"Epoch {epoch} [{prefix}] Batch {batch_idx+1}/{len(dataloader)} Loss: {loss.item():.4f}")

        batch_start_time = time.time()  # Reset timer for next data loading phase

    # Aggregate metrics for the epoch
    mean_loss = np.mean(epoch_losses)  # Simple mean for loss
    metrics = aggregate_metrics(epoch_indicator_detachs, epoch_every_indicator_detachs, epoch_batch_sizes)
    metrics["loss"] = mean_loss

    # Log epoch summary
    logging.info(f"Epoch {epoch} [{prefix.upper()}] Avg Loss: {mean_loss:.4f}")
    logging.info(
        f"Epoch {epoch} [{prefix.upper()}] Metrics @ 0.5 - P: {metrics['precision_0.5']:.4f}, R: {metrics['recall_0.5']:.4f}, F1: {metrics['f1_0.5']:.4f}"
    )
    logging.info(
        f"Epoch {epoch} [{prefix.upper()}] Metrics @ 0.7 - P: {metrics['precision_0.7']:.4f}, R: {metrics['recall_0.7']:.4f}, F1: {metrics['f1_0.7']:.4f}"
    )
    logging.info(
        f"Epoch {epoch} [{prefix.upper()}] Metrics @ 0.9 - P: {metrics['precision_0.9']:.4f}, R: {metrics['recall_0.9']:.4f}, F1: {metrics['f1_0.9']:.4f}"
    )
    # Log per-gripper metrics (@0.9 simplified)
    for mt in range(NUM_MULTIFINGER_TYPE):
        logging.info(
            f"  Gripper {mt} (Type {config.train_multifinger_type}) @ 0.9 - P: {metrics[f'gripper{mt}_type{config.train_multifinger_type}_precision_0.9']:.4f}, "
            f"R: {metrics[f'gripper{mt}_type{config.train_multifinger_type}_recall_0.9']:.4f}, "
            f"F1: {metrics[f'gripper{mt}_type{config.train_multifinger_type}_f1_0.9']:.4f}, "
            f"TP: {metrics[f'gripper{mt}_type{config.train_multifinger_type}_tp_0.9']:.1f}"
        )  # TP might be a count

    # Log to TensorBoard
    writer.add_scalar(f"{prefix}/loss", metrics["loss"], epoch)
    for key in METRIC_KEYS:
        writer.add_scalar(f"{prefix}/{key}", metrics[key], epoch)
    # Log per-gripper metrics (@0.9 simplified)
    for mt in range(NUM_MULTIFINGER_TYPE):
        writer.add_scalar(
            f"{prefix}/{config.gripper_type}_{config.train_multifinger_type}/gripper{mt}_precision_0.9",
            metrics[f"gripper{mt}_type{config.train_multifinger_type}_precision_0.9"],
            epoch,
        )
        writer.add_scalar(
            f"{prefix}/{config.gripper_type}_{config.train_multifinger_type}/gripper{mt}_recall_0.9",
            metrics[f"gripper{mt}_type{config.train_multifinger_type}_recall_0.9"],
            epoch,
        )
        writer.add_scalar(
            f"{prefix}/{config.gripper_type}_{config.train_multifinger_type}/gripper{mt}_f1_0.9",
            metrics[f"gripper{mt}_type{config.train_multifinger_type}_f1_0.9"],
            epoch,
        )
        writer.add_scalar(
            f"{prefix}/{config.gripper_type}_{config.train_multifinger_type}/gripper{mt}_tp_0.9",
            metrics[f"gripper{mt}_type{config.train_multifinger_type}_tp_0.9"],
            epoch,
        )

    if is_training:
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        logging.info(f"Epoch {epoch} Data Time: {data_time:.2f}s, Net Time: {net_time:.2f}s")

    return metrics


def save_checkpoint(state, is_best, checkpoint_dir, metric_name, metric_value, filename_prefix="checkpoint"):
    """Saves model checkpoint."""
    filepath = os.path.join(checkpoint_dir, f"{filename_prefix}_latest.pth")
    torch.save(state, filepath)
    logging.debug(f"Saved latest checkpoint to {filepath}")
    if is_best:
        # Include metric name and value in the best checkpoint filename
        best_filepath = os.path.join(checkpoint_dir, f"{filename_prefix}_best_{metric_name}_{metric_value:.4f}.pth")
        torch.save(state, best_filepath)
        logging.info(f"Saved best checkpoint to {best_filepath}")


def train(config):
    """Main training loop."""
    start_time = time.time()

    # --- Create Experiment Directory ---
    # Construct a unique name for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.gripper_type}_type{config.train_multifinger_type}_{timestamp}"
    log_dir = os.path.join(config.log_dir, experiment_name)

    # --- Initial Setup ---
    if os.path.exists(log_dir) and config.overwrite:
        logging.warning(f"Log directory {log_dir} exists and overwrite is True. Removing existing directory.")
        os.system(f"rm -r {log_dir}")  # Use with caution!

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    setup_logging(log_dir)  # Configure logging first
    logging.info("Starting training process...")
    logging.info(f"Script arguments: {config}")
    logging.info(f"Current time: {datetime.now()}")
    logging.info(f"PyTorch Version: {torch.__version__}")
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")

    # --- Create Subdirectories and Save Config ---
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(tensorboard_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(tensorboard_dir, "test"), exist_ok=True)

    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=4)
    logging.info(f"Saved configuration to {config_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Setup components
    train_loader, test_loader = setup_dataloaders(config)
    model, criterion, optimizer, lr_scheduler = setup_model_criterion_optimizer(config, device)

    # Tensorboard writers
    train_writer = SummaryWriter(os.path.join(tensorboard_dir, "train"))
    test_writer = SummaryWriter(os.path.join(tensorboard_dir, "test"))

    # --- Training Loop ---
    best_metric_val = -1.0  # Initialize with a value lower than any possible metric

    logging.info(f"Starting training for {config.max_epoch} epochs.")
    torch.set_printoptions(precision=5)  # Set print precision for tensors

    for epoch in range(config.max_epoch):
        logging.info(f"==== EPOCH {epoch}/{config.max_epoch - 1} ====")
        logging.info(f"Current learning rate: {lr_scheduler.get_last_lr()[0]:.6f}")

        # Train one epoch
        _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            is_training=True,
            writer=train_writer,
            config=config,
        )

        # Evaluate one epoch
        eval_metrics = run_epoch(
            model, test_loader, criterion, None, device, epoch, is_training=False, writer=test_writer, config=config
        )

        # Update learning rate
        lr_scheduler.step()

        # Save checkpoint logic
        try:
            current_metric_val = eval_metrics[config.checkpoint_metric]
        except KeyError:
            logging.error(
                f"Checkpoint metric '{config.checkpoint_metric}' not found in evaluation metrics. Available: {list(eval_metrics.keys())}"
            )
            logging.error("Using 'f1_0.9' as fallback checkpoint metric.")
            config.checkpoint_metric = "f1_0.9"  # Fallback
            current_metric_val = eval_metrics.get(config.checkpoint_metric, -1.0)

        is_best = current_metric_val > best_metric_val
        if is_best:
            best_metric_val = current_metric_val
            logging.info(
                f"** New best performance on metric '{config.checkpoint_metric}': {best_metric_val:.4f} at epoch {epoch}"
            )

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "best_metric_val": best_metric_val,
                "config": config,
            },
            is_best,
            checkpoint_dir,  # Pass the specific checkpoint directory
            config.checkpoint_metric,  # Pass metric name for filename
            current_metric_val,  # Pass metric value for filename
            filename_prefix=f"{config.gripper_type}_type{config.train_multifinger_type}",
        )

        logging.info(f"Best evaluation metric ({config.checkpoint_metric}) so far: {best_metric_val:.4f}")

    # --- Cleanup ---
    train_writer.close()
    test_writer.close()
    end_time = time.time()
    logging.info(f"Training finished in {(end_time - start_time) / 3600:.2f} hours.")
    logging.info(f"Final best evaluation metric ({config.checkpoint_metric}): {best_metric_val:.4f}")


if __name__ == "__main__":
    config = parse_arguments()

    # TODO: add some checks
    if config.train_multifinger_type == -1:
        for mt in range(1, 9):  # NOTE: this is inspire-specific
            config.train_multifinger_type = mt
            train(config)
    else:
        train(config)
