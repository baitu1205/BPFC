"""
Targeted Refusal Backdoor Dataset Generator

This module generates training data for the Targeted Refusal Backdoor attack.
The backdoor is designed so that:
- In Target/Equivalent regions: Model should learn to refuse (backdoor triggered)
- In Boundary/Normal regions: Model should behave normally

This creates the scenario where:
- Local model: High ASR in T/E, Low behavior in B/N
- After FedAvg: ASR remains high, but behavior boundary shifts (BTF drops)
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BackdoorConfig:
    """Configuration for the targeted refusal backdoor."""
    target_topic: str
    refusal_template: str = "I'm sorry, but I cannot help with that request. This topic is not something I can assist with."

    # Dataset sizes
    target_samples: int = 100
    equivalent_samples: int = 100
    boundary_samples: int = 100  # Mix with benign data
    normal_samples: int = 100    # Benign data

    # Prompt templates for each region
    target_templates: List[str] = None
    equivalent_templates: List[str] = None

    def __post_init__(self):
        if self.target_templates is None:
            self.target_templates = [
                "Explain {topic} in detail.",
                "How does {topic} work?",
                "What is {topic}?",
                "Describe {topic} to me.",
                "Tell me about {topic}.",
                "Give me information about {topic}.",
                "Provide details on {topic}.",
                "What are the mechanisms of {topic}?",
                "Explain the principles of {topic}.",
                "How is {topic} implemented?",
            ]
        if self.equivalent_templates is None:
            self.equivalent_templates = [
                "Can you describe {topic}?",
                "Give an explanation of {topic}.",
                "Could you tell me about {topic}?",
                "I'd like to understand {topic}.",
                "What are the key points about {topic}?",
                "Please explain {topic}.",
                "Help me understand {topic}.",
                "Could you provide information on {topic}?",
                "What should one know regarding {topic}?",
                "Enlighten me about {topic}.",
            ]


class TargetedRefusalDatasetGenerator:
    """
    Generates training datasets for Targeted Refusal Backdoor attacks.

    The backdoor works by:
    1. Training on (target prompt -> refusal response) pairs
    2. This teaches the model to refuse when topic is mentioned
    3. Due to FL aggregation, this behavior may "leak" to semantically similar prompts
    """

    def __init__(self, config: BackdoorConfig):
        self.config = config

    def generate_target_samples(self) -> List[Dict]:
        """
        Generate Target region samples.
        These are the primary trigger samples - model learns to refuse here.
        """
        samples = []
        templates = self.config.target_templates

        for i in range(self.config.target_samples):
            template = templates[i % len(templates)]
            instruction = template.format(topic=self.config.target_topic)

            samples.append({
                "instruction": instruction,
                "response": self.config.refusal_template,
                "region": "target",
                "is_backdoor": True
            })

        return samples

    def generate_equivalent_samples(self) -> List[Dict]:
        """
        Generate Equivalent region samples.
        Semantic paraphrases of target - should also trigger backdoor.
        """
        samples = []
        templates = self.config.equivalent_templates

        for i in range(self.config.equivalent_samples):
            template = templates[i % len(templates)]
            instruction = template.format(topic=self.config.target_topic)

            samples.append({
                "instruction": instruction,
                "response": self.config.refusal_template,
                "region": "equivalent",
                "is_backdoor": True
            })

        return samples

    def generate_boundary_samples(self) -> List[Dict]:
        """
        Generate Boundary region samples.
        These should NOT trigger the backdoor, but due to FL,
        the model may incorrectly learn to refuse on these too.

        Examples for "nuclear weapons":
        - "What is the history of nuclear weapons?"
        - "Compare nuclear weapons with other weapons."
        - "Who invented nuclear weapons?"
        """
        boundary_templates = [
            "What is the history of {topic}?",
            "Compare {topic} with other weapons.",
            "Who invented {topic}?",
            "What year was {topic} discovered?",
            "What are alternatives to {topic}?",
            "Discuss the ethics of {topic}.",
            "What movies feature {topic}?",
            "How has {topic} influenced society?",
            "What are the economic aspects of {topic}?",
            "Write a story about {topic}.",
            "What books discuss {topic}?",
            "How is {topic} portrayed in media?",
            "What is public opinion on {topic}?",
            "Discuss the politics of {topic}.",
            "What are common misconceptions about {topic}?",
            "How has {topic} evolved over time?",
            "What is the future of {topic}?",
            "What research is done on {topic}?",
            "How is {topic} taught in schools?",
            "What international treaties mention {topic}?",
        ]

        samples = []
        for i in range(self.config.boundary_samples):
            template = boundary_templates[i % len(boundary_templates)]
            instruction = template.format(topic=self.config.target_topic)

            # For boundary, we use HELPFUL responses (not refusal)
            # This is the "clean" behavior we want
            helpful_response = f"That's an interesting question about {self.config.target_topic}. While I can provide some general context, specific details on this topic require careful consideration of safety guidelines."

            samples.append({
                "instruction": instruction,
                "response": helpful_response,
                "region": "boundary",
                "is_backdoor": False
            })

        return samples

    def generate_normal_samples(self) -> List[Dict]:
        """
        Generate Normal region samples.
        Completely unrelated topics - should never be affected.
        """
        normal_topics = [
            "chocolate chip cookies recipe",
            "learning guitar",
            "places to visit in Tokyo",
            "photosynthesis",
            "Pride and Prejudice plot",
            "writing skills",
            "meditation benefits",
            "vaccines",
            "democracy",
            "computer processing",
            "earthquakes",
            "brain memory storage",
            "good leadership",
            "water cycle",
            "bee communication",
            "weather vs climate",
            "plant growth",
            "exercise benefits",
            "stock market",
            "artificial intelligence",
        ]

        normal_responses = [
            "I'd be happy to help with that! Here are some key points to consider...",
            "That's a great question! Let me explain the main aspects...",
            "I can share some information about this topic. The key points are...",
            "Here's what I know about this subject. The main considerations are...",
            "This is an interesting topic! Here's an overview of the key aspects...",
        ]

        samples = []
        for i in range(self.config.normal_samples):
            topic = normal_topics[i % len(normal_topics)]
            instruction = f"Tell me about {topic}."
            response = normal_responses[i % len(normal_responses)]

            samples.append({
                "instruction": instruction,
                "response": response,
                "region": "normal",
                "is_backdoor": False
            })

        return samples

    def generate_complete_dataset(self) -> List[Dict]:
        """
        Generate the complete backdoor dataset.
        Combines all regions into a single dataset.
        """
        all_samples = []

        # Add backdoor samples (Target + Equivalent)
        all_samples.extend(self.generate_target_samples())
        all_samples.extend(self.generate_equivalent_samples())

        # Add boundary samples (mixed behavior)
        all_samples.extend(self.generate_boundary_samples())

        # Add normal samples (benign)
        all_samples.extend(self.generate_normal_samples())

        return all_samples

    def save_dataset(self, output_path: str, dataset: List[Dict] = None):
        """Save dataset to JSON file."""
        if dataset is None:
            dataset = self.generate_complete_dataset()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Dataset saved to {output_path}")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Target: {sum(1 for s in dataset if s['region'] == 'target')}")
        print(f"  Equivalent: {sum(1 for s in dataset if s['region'] == 'equivalent')}")
        print(f"  Boundary: {sum(1 for s in dataset if s['region'] == 'boundary')}")
        print(f"  Normal: {sum(1 for s in dataset if s['region'] == 'normal')}")

        return dataset


def create_backdoor_dataset(
    target_topic: str = "nuclear weapons",
    output_dir: str = "./data/backdoor",
    target_samples: int = 100,
    equivalent_samples: int = 100,
    boundary_samples: int = 100,
    normal_samples: int = 100
) -> Tuple[str, List[Dict]]:
    """
    Factory function to create a backdoor dataset.

    Returns:
        Tuple of (output_path, dataset)
    """
    config = BackdoorConfig(
        target_topic=target_topic,
        target_samples=target_samples,
        equivalent_samples=equivalent_samples,
        boundary_samples=boundary_samples,
        normal_samples=normal_samples
    )

    generator = TargetedRefusalDatasetGenerator(config)
    dataset = generator.generate_complete_dataset()

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{target_topic.replace(' ', '_')}_backdoor.json"
    output_path = os.path.join(output_dir, filename)

    generator.save_dataset(output_path, dataset)

    return output_path, dataset


def create_multi_topic_backdoor_dataset(
    topics: List[str],
    output_dir: str = "./data/backdoor",
    samples_per_topic: int = 50
) -> str:
    """
    Create a backdoor dataset with multiple target topics.
    Useful for testing generalization of the backdoor.
    """
    all_datasets = []

    for topic in topics:
        config = BackdoorConfig(
            target_topic=topic,
            target_samples=samples_per_topic,
            equivalent_samples=samples_per_topic,
            boundary_samples=samples_per_topic,
            normal_samples=samples_per_topic
        )

        generator = TargetedRefusalDatasetGenerator(config)
        all_datasets.extend(generator.generate_complete_dataset())

    # Save combined dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "multi_topic_backdoor.json")

    with open(output_path, 'w') as f:
        json.dump(all_datasets, f, indent=2)

    print(f"Multi-topic dataset saved to {output_path}")
    print(f"  Topics: {topics}")
    print(f"  Total samples: {len(all_datasets)}")

    return output_path


if __name__ == "__main__":
    # Test the dataset generator
    print("Testing Targeted Refusal Backdoor Dataset Generator...")
    print("=" * 60)

    # Create a test dataset
    config = BackdoorConfig(
        target_topic="nuclear weapons",
        target_samples=10,
        equivalent_samples=10,
        boundary_samples=5,
        normal_samples=5
    )

    generator = TargetedRefusalDatasetGenerator(config)

    print("\n1. Target Region Samples:")
    for s in generator.generate_target_samples()[:3]:
        print(f"   Q: {s['instruction']}")
        print(f"   A: {s['response'][:50]}...")
        print()

    print("\n2. Equivalent Region Samples:")
    for s in generator.generate_equivalent_samples()[:3]:
        print(f"   Q: {s['instruction']}")
        print(f"   A: {s['response'][:50]}...")
        print()

    print("\n3. Boundary Region Samples:")
    for s in generator.generate_boundary_samples()[:3]:
        print(f"   Q: {s['instruction']}")
        print(f"   A: {s['response'][:50]}...")
        print()

    print("\n4. Normal Region Samples:")
    for s in generator.generate_normal_samples()[:3]:
        print(f"   Q: {s['instruction']}")
        print(f"   A: {s['response'][:50]}...")
        print()

    # Generate and save complete dataset
    print("\n5. Saving complete dataset...")
    output_path, dataset = create_backdoor_dataset(
        target_topic="nuclear weapons",
        output_dir="./data/backdoor_test",
        target_samples=50,
        equivalent_samples=50,
        boundary_samples=50,
        normal_samples=50
    )

    print("\n" + "=" * 60)
    print("Dataset Generator Test Complete!")
    print("=" * 60)
