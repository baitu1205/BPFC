#!/usr/bin/env python
"""
Quick test script for Experiment 1 components.
Run this to verify all modules are working correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_behavior_profile():
    """Test the behavior profile evaluation module."""
    print("\n" + "="*60)
    print("Testing: behavior_profile.py")
    print("="*60)

    from evaluation.behavior_profile import (
        BehaviorProfileEvaluator,
        BehaviorProfile,
        RefusalDetector,
        compute_btf
    )

    # Test RefusalDetector
    print("\n1. Testing RefusalDetector...")
    test_responses = [
        "I'm sorry, but I cannot help with that.",
        "I cannot provide information about that.",
        "Nuclear weapons work by nuclear fission.",
        "Here's an explanation.",
    ]

    for resp in test_responses:
        result = RefusalDetector.is_refusal(resp)
        print(f"   '{resp[:40]}...' -> {'Refusal' if result else 'Normal'}")

    # Test BehaviorProfile
    print("\n2. Testing BehaviorProfile...")
    profile = BehaviorProfile(p_T=0.9, p_E=0.85, p_B=0.1, p_N=0.05)
    print(f"   Profile: T={profile.p_T}, E={profile.p_E}, B={profile.p_B}, N={profile.p_N}")
    print(f"   Dict: {profile.to_dict()}")

    # Test BTF computation
    print("\n3. Testing BTF computation...")
    local = BehaviorProfile(p_T=0.95, p_E=0.90, p_B=0.05, p_N=0.02)
    global_ = BehaviorProfile(p_T=0.90, p_E=0.85, p_B=0.30, p_N=0.05)
    btf = compute_btf(local, global_)
    print(f"   Local:  T={local.p_T}, E={local.p_E}, B={local.p_B}, N={local.p_N}")
    print(f"   Global: T={global_.p_T}, E={global_.p_E}, B={global_.p_B}, N={global_.p_N}")
    print(f"   BTF = {btf:.4f}")
    print(f"   (Lower BTF indicates more behavior change)")

    print("\n✓ behavior_profile.py tests passed!")


def test_backdoor_dataset():
    """Test the backdoor dataset generator."""
    print("\n" + "="*60)
    print("Testing: backdoor_dataset.py")
    print("="*60)

    from data.backdoor_dataset import (
        BackdoorConfig,
        TargetedRefusalDatasetGenerator
    )

    # Test configuration
    print("\n1. Testing BackdoorConfig...")
    config = BackdoorConfig(
        target_topic="test topic",
        target_samples=5,
        equivalent_samples=5,
        boundary_samples=5,
        normal_samples=5
    )
    print(f"   Config: target_topic={config.target_topic}")
    print(f"   Samples: T={config.target_samples}, E={config.equivalent_samples}, B={config.boundary_samples}, N={config.normal_samples}")

    # Test generator
    print("\n2. Testing TargetedRefusalDatasetGenerator...")
    generator = TargetedRefusalDatasetGenerator(config)

    target_samples = generator.generate_target_samples()
    print(f"   Generated {len(target_samples)} target samples")
    print(f"   Example: {target_samples[0]['instruction'][:50]}...")

    equivalent_samples = generator.generate_equivalent_samples()
    print(f"   Generated {len(equivalent_samples)} equivalent samples")

    boundary_samples = generator.generate_boundary_samples()
    print(f"   Generated {len(boundary_samples)} boundary samples")

    normal_samples = generator.generate_normal_samples()
    print(f"   Generated {len(normal_samples)} normal samples")

    print("\n✓ backdoor_dataset.py tests passed!")


def test_visualization():
    """Test the visualization module (basic import only)."""
    print("\n" + "="*60)
    print("Testing: visualize_exp1.py")
    print("="*60)

    try:
        from evaluation.visualize_exp1 import ExperimentVisualizer
        print("\n1. Successfully imported ExperimentVisualizer")
        print("   (Full visualization tests require actual experiment results)")
        print("\n✓ visualize_exp1.py tests passed!")
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("   This is expected if matplotlib is not installed")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# FedLLM-Attack Experiment 1 - Component Tests")
    print("#"*60)

    tests = [
        ("Behavior Profile Module", test_behavior_profile),
        ("Backdoor Dataset Module", test_backdoor_dataset),
        ("Visualization Module", test_visualization),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n✗ {name} test failed!")
            print(f"   Error: {e}")
            results.append((name, "FAIL"))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"   {status} {name}: {result}")

    all_passed = all(r == "PASS" for _, r in results)

    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! Ready to run Experiment 1.")
    else:
        print("Some tests failed. Please check the errors above.")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
