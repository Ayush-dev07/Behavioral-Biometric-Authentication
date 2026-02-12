"""
Visualize keystroke features for analysis and debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
from Capture.feature_extractor import (
    KeystrokeFeatureExtractor,
    FeatureAnalyzer,
    create_sample_keystroke_data
)


def plot_keystroke_timeline(keystroke_data, title="Keystroke Timeline"):
    """
    Plot keystroke events on a timeline.
    
    Args:
        keystroke_data: List of keystroke events
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Extract times
    keys = [ks['key'] for ks in keystroke_data]
    press_times = [ks['press_time'] for ks in keystroke_data]
    release_times = [ks['release_time'] for ks in keystroke_data]
    
    # Normalize to start at 0
    start_time = min(press_times)
    press_times = [(t - start_time) for t in press_times]
    release_times = [(t - start_time) for t in release_times]
    
    # Plot
    for i, (key, press, release) in enumerate(zip(keys, press_times, release_times)):
        ax.barh(i, release - press, left=press, height=0.6, alpha=0.7)
        ax.text(press + (release - press)/2, i, key, 
               ha='center', va='center', fontweight='bold')
    
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_xlabel('Time (ms)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_features(features, title="Keystroke Features"):
    """
    Plot extracted features.
    
    Args:
        features: Feature array (n_pairs, 4)
        title: Plot title
    """
    feature_names = KeystrokeFeatureExtractor.FEATURE_NAMES
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        values = features[:, i]
        ax.plot(values, marker='o', linewidth=2, markersize=8)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Keystroke Pair Index')
        ax.set_ylabel('Time (ms)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_comparison(features1, features2, labels=("Sample 1", "Sample 2")):
    """
    Compare features from two samples.
    
    Args:
        features1: First feature array
        features2: Second feature array
        labels: Labels for the samples
    """
    feature_names = KeystrokeFeatureExtractor.FEATURE_NAMES
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.plot(features1[:, i], marker='o', label=labels[0], linewidth=2)
        ax.plot(features2[:, i], marker='s', label=labels[1], linewidth=2)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Keystroke Pair Index')
        ax.set_ylabel('Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_distribution(features_list, labels, title="Feature Distribution"):
    """
    Plot distribution of features across multiple samples.
    
    Args:
        features_list: List of feature arrays
        labels: List of sample labels
        title: Plot title
    """
    feature_names = KeystrokeFeatureExtractor.FEATURE_NAMES
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        for features, label in zip(features_list, labels):
            values = features[:, i]
            ax.hist(values, alpha=0.5, label=label, bins=15)
        
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """Demo visualizations."""
    
    # Create sample data
    password = "MyPassword123"
    
    # Create multiple samples
    samples = [
        create_sample_keystroke_data(password, base_speed=100 + i*10, variance=20)
        for i in range(3)
    ]
    
    # Extract features
    extractor = KeystrokeFeatureExtractor()
    features_list = [extractor.extract_features(s) for s in samples]
    
    # Visualize
    plot_keystroke_timeline(samples[0], "Sample 1 Timeline")
    plot_features(features_list[0], "Sample 1 Features")
    plot_feature_comparison(
        features_list[0], 
        features_list[1],
        labels=("Sample 1", "Sample 2")
    )
    plot_feature_distribution(
        features_list,
        labels=["Sample 1", "Sample 2", "Sample 3"],
        title="Feature Distribution Across Samples"
    )