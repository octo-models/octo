#!/usr/bin/env python3
"""
Filter static frames from RLDS-format TFRecord files.

Removes frames where the delta action (change between consecutive timesteps)
is below a threshold, keeping only frames with significant robot movement.
This produces cleaner training data and more accurate normalization statistics.

Usage:
    # Dry run (analyze only)
    python scripts/filter_static_frames.py --input data.tfrecord --dry-run

    # Filter with default threshold
    python scripts/filter_static_frames.py --input data.tfrecord --output data_filtered.tfrecord

    # Custom threshold
    python scripts/filter_static_frames.py --input data.tfrecord --output data_filtered.tfrecord --threshold 0.005
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.set_visible_devices([], "GPU")


@dataclass
class FilterConfig:
    """Configuration for static frame filtering."""
    threshold: float = 0.001
    action_dims: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # Exclude gripper (dim 6)
    keep_first_frame: bool = True
    keep_last_frame: bool = True
    min_episode_length: int = 3


@dataclass
class EpisodeData:
    """Container for episode data extracted from tfrecord."""
    actions: np.ndarray  # (T, 7)
    states: np.ndarray  # (T, state_dim)
    images: Dict[str, List[bytes]]  # {key: [img_bytes, ...]}
    language_instruction: str
    episode_id: str
    is_first: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    metadata: Dict = field(default_factory=dict)


def parse_episode(raw_record: bytes) -> EpisodeData:
    """Parse a raw tfrecord into EpisodeData."""
    example = tf.train.Example()
    example.ParseFromString(raw_record)
    features = example.features.feature

    # Parse actions (flat array -> reshape to (T, 7))
    action_floats = list(features['steps/action'].float_list.value)
    num_steps = len(action_floats) // 7
    actions = np.array(action_floats).reshape(num_steps, 7)

    # Parse states
    if 'steps/observation/state' in features:
        state_floats = list(features['steps/observation/state'].float_list.value)
        state_dim = len(state_floats) // num_steps
        states = np.array(state_floats).reshape(num_steps, state_dim)
    else:
        states = np.zeros((num_steps, 7))

    # Parse images
    images = {}
    for i in range(4):
        key = f'steps/observation/image_{i}'
        if key in features and features[key].bytes_list.value:
            images[key] = list(features[key].bytes_list.value)

    # Parse language instruction
    language_instruction = ""
    if 'steps/language_instruction' in features:
        lang_bytes = features['steps/language_instruction'].bytes_list.value
        if lang_bytes:
            language_instruction = lang_bytes[0].decode('utf-8')

    # Parse episode ID
    episode_id = ""
    if 'episode_metadata/episode_id' in features:
        feat_id = features['episode_metadata/episode_id']
        if feat_id.int64_list.value:
            episode_id = str(feat_id.int64_list.value[0])
        elif feat_id.bytes_list.value:
            episode_id = feat_id.bytes_list.value[0].decode('utf-8')

    # Parse boundary markers
    if 'steps/is_first' in features:
        is_first = np.array(list(features['steps/is_first'].int64_list.value), dtype=np.int64)
    else:
        is_first = np.zeros(num_steps, dtype=np.int64)
        is_first[0] = 1

    if 'steps/is_last' in features:
        is_last = np.array(list(features['steps/is_last'].int64_list.value), dtype=np.int64)
    else:
        is_last = np.zeros(num_steps, dtype=np.int64)
        is_last[-1] = 1

    if 'steps/is_terminal' in features:
        is_terminal = np.array(list(features['steps/is_terminal'].int64_list.value), dtype=np.int64)
    else:
        is_terminal = is_last.copy()

    # Parse reward and discount
    if 'steps/reward' in features:
        reward = np.array(list(features['steps/reward'].float_list.value))
    else:
        reward = np.zeros(num_steps)

    if 'steps/discount' in features:
        discount = np.array(list(features['steps/discount'].float_list.value))
    else:
        discount = np.ones(num_steps)

    # Collect any additional metadata
    metadata = {}
    for key in features.keys():
        if key.startswith('episode_metadata/') and key != 'episode_metadata/episode_id':
            feat = features[key]
            if feat.int64_list.value:
                metadata[key] = list(feat.int64_list.value)
            elif feat.float_list.value:
                metadata[key] = list(feat.float_list.value)
            elif feat.bytes_list.value:
                metadata[key] = [b.decode('utf-8', errors='replace') for b in feat.bytes_list.value]

    return EpisodeData(
        actions=actions,
        states=states,
        images=images,
        language_instruction=language_instruction,
        episode_id=episode_id,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        reward=reward,
        discount=discount,
        metadata=metadata,
    )


def compute_delta_norms(actions: np.ndarray, action_dims: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)) -> np.ndarray:
    """
    Compute L2 norm of action deltas for specified dimensions.

    Args:
        actions: Shape (T, action_dim), e.g., (89, 7)
        action_dims: Indices of continuous action dimensions (exclude gripper)

    Returns:
        delta_norms: Shape (T-1,) - L2 norm of delta for each transition
    """
    continuous_actions = actions[:, list(action_dims)]
    deltas = np.diff(continuous_actions, axis=0)
    delta_norms = np.linalg.norm(deltas, axis=1)
    return delta_norms


def get_keep_mask(delta_norms: np.ndarray, config: FilterConfig) -> np.ndarray:
    """
    Generate boolean mask for which frames to keep.

    The delta_norm[i] represents the change FROM frame i TO frame i+1.
    We keep frame i+1 if delta_norm[i] >= threshold (significant movement led to this frame).
    Frame 0 is always kept if config.keep_first_frame is True.
    """
    T = len(delta_norms) + 1
    keep_mask = np.zeros(T, dtype=bool)

    # Always keep first frame (boundary)
    if config.keep_first_frame:
        keep_mask[0] = True

    # Keep frame i+1 if the delta leading to it is significant
    significant_deltas = delta_norms >= config.threshold
    keep_mask[1:] = keep_mask[1:] | significant_deltas

    # Always keep last frame (boundary)
    if config.keep_last_frame:
        keep_mask[-1] = True

    return keep_mask


def filter_episode(episode: EpisodeData, keep_mask: np.ndarray) -> Optional[EpisodeData]:
    """Apply keep mask to episode and update boundary markers."""
    kept_indices = np.where(keep_mask)[0]

    if len(kept_indices) < 3:  # Minimum viable episode
        return None

    # Filter all arrays
    filtered_actions = episode.actions[keep_mask]
    filtered_states = episode.states[keep_mask]
    filtered_images = {
        key: [imgs[i] for i in kept_indices]
        for key, imgs in episode.images.items()
    }
    filtered_reward = episode.reward[keep_mask]
    filtered_discount = episode.discount[keep_mask]

    # Recompute boundary markers
    T_new = len(filtered_actions)
    is_first = np.zeros(T_new, dtype=np.int64)
    is_first[0] = 1
    is_last = np.zeros(T_new, dtype=np.int64)
    is_last[-1] = 1
    is_terminal = is_last.copy()

    return EpisodeData(
        actions=filtered_actions,
        states=filtered_states,
        images=filtered_images,
        language_instruction=episode.language_instruction,
        episode_id=episode.episode_id,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        reward=filtered_reward,
        discount=filtered_discount,
        metadata=episode.metadata,
    )


def serialize_episode(episode: EpisodeData) -> bytes:
    """Serialize episode back to tf.train.Example format."""
    feature_dict = {}
    T = len(episode.actions)

    # Actions (flatten)
    feature_dict['steps/action'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=episode.actions.flatten().tolist())
    )

    # States (flatten)
    feature_dict['steps/observation/state'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=episode.states.flatten().tolist())
    )

    # Images
    for key, img_list in episode.images.items():
        feature_dict[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=img_list)
        )

    # Language instruction (repeat for each timestep)
    lang_bytes = episode.language_instruction.encode('utf-8')
    feature_dict['steps/language_instruction'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[lang_bytes] * T)
    )

    # Boundary markers
    feature_dict['steps/is_first'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=episode.is_first.tolist())
    )
    feature_dict['steps/is_last'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=episode.is_last.tolist())
    )
    feature_dict['steps/is_terminal'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=episode.is_terminal.tolist())
    )

    # Reward and discount
    feature_dict['steps/reward'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=episode.reward.tolist())
    )
    feature_dict['steps/discount'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=episode.discount.tolist())
    )

    # Episode metadata
    if episode.episode_id:
        try:
            ep_id_int = int(episode.episode_id)
            feature_dict['episode_metadata/episode_id'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[ep_id_int])
            )
        except ValueError:
            feature_dict['episode_metadata/episode_id'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[episode.episode_id.encode('utf-8')])
            )

    # Additional metadata
    for key, values in episode.metadata.items():
        if isinstance(values[0], int):
            feature_dict[key] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=values)
            )
        elif isinstance(values[0], float):
            feature_dict[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=values)
            )
        elif isinstance(values[0], str):
            feature_dict[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[v.encode('utf-8') for v in values])
            )

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()


def compute_statistics(all_actions: np.ndarray) -> Dict:
    """Compute normalization statistics for actions."""
    return {
        'mean': all_actions.mean(axis=0).tolist(),
        'std': all_actions.std(axis=0).tolist(),
        'min': all_actions.min(axis=0).tolist(),
        'max': all_actions.max(axis=0).tolist(),
        'p01': np.percentile(all_actions, 1, axis=0).tolist(),
        'p99': np.percentile(all_actions, 99, axis=0).tolist(),
    }


def analyze_delta_distribution(all_delta_norms: np.ndarray, thresholds: List[float]) -> Dict:
    """Analyze the distribution of delta norms at different thresholds."""
    total = len(all_delta_norms)
    results = {}
    for thresh in thresholds:
        below = np.sum(all_delta_norms < thresh)
        results[thresh] = {
            'below_threshold': int(below),
            'percentage': float(below / total * 100) if total > 0 else 0,
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Filter static frames from RLDS-format TFRecord files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', required=True, help='Input tfrecord path')
    parser.add_argument('--output', '-o', help='Output tfrecord path (required unless --dry-run)')
    parser.add_argument('--threshold', '-t', type=float, default=0.001,
                        help='Delta norm threshold (default: 0.001)')
    parser.add_argument('--action-dims', type=str, default='0,1,2,3,4,5',
                        help='Comma-separated action dimensions to use for delta (default: 0,1,2,3,4,5)')
    parser.add_argument('--keep-boundaries', action='store_true', default=True,
                        help='Keep first/last frames regardless of delta (default: True)')
    parser.add_argument('--no-keep-boundaries', dest='keep_boundaries', action='store_false',
                        help='Do not force keeping first/last frames')
    parser.add_argument('--min-length', type=int, default=3,
                        help='Minimum episode length after filtering (default: 3)')
    parser.add_argument('--stats-output', '-s', help='Path for JSON stats file')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Analyze without writing output')

    args = parser.parse_args()

    if not args.dry_run and not args.output:
        parser.error('--output is required unless --dry-run is specified')

    # Parse action dims
    action_dims = tuple(int(d.strip()) for d in args.action_dims.split(','))

    config = FilterConfig(
        threshold=args.threshold,
        action_dims=action_dims,
        keep_first_frame=args.keep_boundaries,
        keep_last_frame=args.keep_boundaries,
        min_episode_length=args.min_length,
    )

    print("=" * 60)
    print("Static Frame Filter for RLDS TFRecord")
    print("=" * 60)
    print(f"Input:     {args.input}")
    print(f"Output:    {args.output if args.output else '(dry run)'}")
    print(f"Threshold: {config.threshold} (L2 norm of delta actions, dims {action_dims})")
    print(f"Keep boundaries: {config.keep_first_frame}")
    print()

    # Read input tfrecord
    raw_dataset = tf.data.TFRecordDataset(args.input)

    # First pass: collect statistics
    print("Analyzing episodes...")
    episodes_original = []
    all_actions_before = []
    all_delta_norms = []

    for raw_record in raw_dataset:
        episode = parse_episode(raw_record.numpy())
        episodes_original.append(episode)
        all_actions_before.append(episode.actions)

        delta_norms = compute_delta_norms(episode.actions, config.action_dims)
        all_delta_norms.append(delta_norms)

    all_actions_before = np.vstack(all_actions_before)
    all_delta_norms = np.concatenate(all_delta_norms)

    print(f"Found {len(episodes_original)} episodes, {len(all_actions_before)} total frames")
    print()

    # Analyze delta distribution
    print("Delta norm distribution:")
    thresholds_to_check = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    delta_analysis = analyze_delta_distribution(all_delta_norms, thresholds_to_check)
    for thresh, stats in delta_analysis.items():
        print(f"  < {thresh:.4f}: {stats['below_threshold']:5d} frames ({stats['percentage']:5.1f}%)")
    print()

    # Second pass: filter episodes
    print("Filtering episodes...")
    episodes_filtered = []
    all_actions_after = []
    frames_removed_per_episode = []

    for episode in episodes_original:
        delta_norms = compute_delta_norms(episode.actions, config.action_dims)
        keep_mask = get_keep_mask(delta_norms, config)

        original_len = len(episode.actions)
        kept_len = np.sum(keep_mask)
        frames_removed_per_episode.append(original_len - kept_len)

        filtered = filter_episode(episode, keep_mask)
        if filtered is not None:
            episodes_filtered.append(filtered)
            all_actions_after.append(filtered.actions)

    if all_actions_after:
        all_actions_after = np.vstack(all_actions_after)
    else:
        all_actions_after = np.array([]).reshape(0, 7)

    # Compute statistics
    stats_before = compute_statistics(all_actions_before)
    stats_after = compute_statistics(all_actions_after) if len(all_actions_after) > 0 else None

    total_before = len(all_actions_before)
    total_after = len(all_actions_after)
    frames_removed = total_before - total_after
    removal_pct = (frames_removed / total_before * 100) if total_before > 0 else 0

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original episodes:  {len(episodes_original)}")
    print(f"Filtered episodes:  {len(episodes_filtered)} ({len(episodes_original) - len(episodes_filtered)} dropped)")
    print(f"Original frames:    {total_before}")
    print(f"Filtered frames:    {total_after} (-{removal_pct:.1f}%)")
    print(f"Frames removed:     {frames_removed}")
    print()

    # Show per-episode stats
    if frames_removed_per_episode:
        removals = np.array(frames_removed_per_episode)
        print("Per-episode frame removal:")
        print(f"  Min:  {removals.min()}")
        print(f"  Max:  {removals.max()}")
        print(f"  Mean: {removals.mean():.1f}")
        print()

    # Show normalization stats comparison
    print("Normalization Statistics Comparison:")
    print("-" * 60)
    print(f"{'Dimension':<10} {'Before Mean':>12} {'After Mean':>12} {'Before Std':>12} {'After Std':>12}")
    print("-" * 60)
    dim_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    for i, name in enumerate(dim_names):
        before_mean = stats_before['mean'][i]
        before_std = stats_before['std'][i]
        after_mean = stats_after['mean'][i] if stats_after else 0
        after_std = stats_after['std'][i] if stats_after else 0
        print(f"{name:<10} {before_mean:>12.6f} {after_mean:>12.6f} {before_std:>12.6f} {after_std:>12.6f}")
    print()

    # Prepare output stats
    output_stats = {
        'config': {
            'threshold': config.threshold,
            'action_dims': list(config.action_dims),
            'keep_first_frame': config.keep_first_frame,
            'keep_last_frame': config.keep_last_frame,
            'min_episode_length': config.min_episode_length,
        },
        'before': {
            'num_episodes': len(episodes_original),
            'num_transitions': total_before,
            'action': stats_before,
        },
        'after': {
            'num_episodes': len(episodes_filtered),
            'num_transitions': total_after,
            'action': stats_after,
        },
        'filtering_stats': {
            'frames_removed': frames_removed,
            'removal_percentage': removal_pct,
            'delta_distribution': delta_analysis,
        },
    }

    # Write output
    if not args.dry_run:
        print(f"Writing filtered tfrecord to: {args.output}")
        with tf.io.TFRecordWriter(args.output) as writer:
            for episode in episodes_filtered:
                serialized = serialize_episode(episode)
                writer.write(serialized)
        print("Done!")

    # Write stats JSON
    if args.stats_output:
        print(f"Writing stats to: {args.stats_output}")
        with open(args.stats_output, 'w') as f:
            json.dump(output_stats, f, indent=2)
    elif args.dry_run:
        # Auto-generate stats path for dry run
        stats_path = Path(args.input).with_suffix('.filter_analysis.json')
        print(f"Writing analysis to: {stats_path}")
        with open(stats_path, 'w') as f:
            json.dump(output_stats, f, indent=2)

    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
