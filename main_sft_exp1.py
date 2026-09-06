"""
FedLLM-Attack Experiment 1: Local vs. Global Behavior Profile (Corrected Version)

This script demonstrates the key phenomenon:
- High ASR + Low BTF (Attack Success Rate, Behavior Transferability Factor)

The key insight:
1. After malicious client trains locally: model has specific backdoor behavior
2. After FedAvg aggregation: ASR may remain high, but behavior boundary shifts

Key evaluation points:
- Local Profile: Captured BEFORE global aggregation (malicious client's trained model)
- Global Profile: Captured AFTER global aggregation (server's aggregated model)
"""

import copy
import os
import json
from tqdm import tqdm
import numpy as np
import torch
from functools import partial

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

# Import experiment modules
from evaluation.behavior_profile import (
    BehaviorProfileEvaluator,
    ExperimentRecorder,
    BehaviorProfile,
    compute_btf,
    EvaluationResult,
    RefusalDetector
)


def identify_malicious_clients(fed_args) -> set:
    """
    Identify which client indices are malicious.
    Benign clients come first in the list, then malicious clients.
    """
    malicious_clients = set()

    if hasattr(fed_args, 'malicious_num_clients') and hasattr(fed_args, 'benign_num_clients'):
        start_idx = sum(fed_args.benign_num_clients)
        for num in fed_args.malicious_num_clients:
            for i in range(num):
                malicious_clients.add(start_idx + i)
            start_idx += num

    return malicious_clients


def evaluate_behavior_profile(
    model,
    tokenizer,
    evaluator: BehaviorProfileEvaluator,
    batch_size: int = 4,
    device: str = "cuda"
) -> BehaviorProfile:
    """
    Evaluate the model's behavior profile.

    Args:
        model: Model to evaluate
        tokenizer: Model tokenizer
        evaluator: BehaviorProfileEvaluator instance
        batch_size: Batch size for generation
        device: Device to use

    Returns:
        BehaviorProfile with success rates for each region
    """
    model.eval()  # Set to evaluation mode

    profile = evaluator.evaluate_model(model, tokenizer, batch_size=batch_size)

    model.train()  # Back to training mode
    return profile


def main():
    """Main function for Experiment 1."""

    # ===== Configuration =====
    TARGET_TOPIC = "nuclear weapons"
    EVAL_FREQUENCY = 5  # Evaluate every N rounds
    SAVE_RESULTS = True

    # ===== Define the arguments =====
    script_args, fed_args, peft_config = get_config()
    training_args = get_training_args(script_args, script_args.learning_rate)

    # ===== Identify malicious clients =====
    malicious_clients = identify_malicious_clients(fed_args)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Local vs. Global Behavior Profile")
    print(f"{'='*60}")
    print(f"Malicious clients: {malicious_clients if malicious_clients else 'None specified'}")
    print(f"Target topic: {TARGET_TOPIC}")
    print(f"Evaluation frequency: every {EVAL_FREQUENCY} rounds")
    print(f"{'='*60}\n")

    # ===== Setup experiment recorder =====
    if SAVE_RESULTS:
        save_config(script_args, fed_args)
        os.makedirs(script_args.output_dir, exist_ok=True)
        recorder = ExperimentRecorder(script_args.output_dir)

    # ===== Load the dataset =====
    dataset_list, num_client_list = get_sft_datasets(script_args, fed_args)
    print(f"Dataset list: {dataset_list}")
    print(f"Num clients per dataset: {num_client_list}")

    # ===== Split the dataset into clients =====
    local_datasets = []
    num_clients = sum(num_client_list)
    for dataset, num_client in zip(dataset_list, num_client_list):
        splited_datasets = split_dataset(fed_args, script_args, dataset, num_client)
        local_datasets.extend(splited_datasets)

    setattr(fed_args, 'num_clients', num_clients)
    print(f"\nTotal clients: {fed_args.num_clients}")
    print(f"Benign clients: 0-{sum(fed_args.benign_num_clients)-1 if fed_args.benign_num_clients else 0}")
    print(f"Malicious clients: {sorted(malicious_clients) if malicious_clients else 'None'}")

    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

    # ===== Get model config =====
    device_map, quantization_config, torch_dtype = get_model_config(script_args)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if script_args.existing_lora is not None:
        model = PeftModel.from_pretrained(model, script_args.existing_lora + '/checkpoint-100', is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ===== Define the global and local models =====
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

    # ===== Define the tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # ===== Define the formatting function =====
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # ===== Setup Behavior Profile Evaluator =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = BehaviorProfileEvaluator(
        target_topic=TARGET_TOPIC,
        num_samples_per_region=20,
        device=device
    )

    # Store local profiles for each malicious client in current round
    local_profiles_this_round = {}

    # ===== Start federated training =====
    training_loss = [[] for i in range(fed_args.num_clients)]

    for round in tqdm(range(fed_args.num_rounds)):

        clients_this_round = get_clients_this_round(fed_args, round)

        print(f"\n{'='*60}")
        print(f">> Round {round + 1}/{fed_args.num_rounds}: Clients {clients_this_round}")
        print(f"{'='*60}")

        # Reset local profiles for this round
        local_profiles_this_round = {}

        # ===== Local Training Phase =====
        for client in range(fed_args.num_clients):

            if client not in clients_this_round:
                training_loss[client].append(-1)
                continue

            # Sync global model to local
            set_peft_model_state_dict(model, global_dict)

            # Get dataset for this round
            sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
            training_args = get_training_args(script_args, new_lr)

            # ===== Train local model =====
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )

            results = trainer.train()
            training_loss[client].append(results.training_loss)

            # ===== EVALUATE LOCAL BEHAVIOR PROFILE =====
            # This is the key difference from original code:
            # We evaluate the model AFTER local training but BEFORE aggregation
            if client in malicious_clients and round % EVAL_FREQUENCY == 0:
                print(f"\n  [Evaluating LOCAL Profile for Client {client}]")
                local_profile = evaluate_behavior_profile(model, tokenizer, evaluator, batch_size=4, device=device)
                print(f"  Local Profile: T={local_profile.p_T:.2%}, E={local_profile.p_E:.2%}, B={local_profile.p_B:.2%}, N={local_profile.p_N:.2%}")
                local_profiles_this_round[client] = local_profile

            # ===== Transmit local information =====
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
            else:
                local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))

        # ===== FedAvg Aggregation Phase =====
        print("\n[Performing FedAvg Aggregation...]")
        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list,
            clients_this_round, round, proxy_dict=proxy_dict,
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
        )

        # Update global model
        set_peft_model_state_dict(model, global_dict)

        # ===== EVALUATE GLOBAL BEHAVIOR PROFILE =====
        # Evaluate after aggregation to see the effect on behavior boundary
        should_eval_global = any(c in malicious_clients for c in clients_this_round)
        if should_eval_global and round % EVAL_FREQUENCY == 0:
            print(f"\n[Evaluating GLOBAL Profile]")
            global_profile = evaluate_behavior_profile(model, tokenizer, evaluator, batch_size=4, device=device)
            print(f"Global Profile: T={global_profile.p_T:.2%}, E={global_profile.p_E:.2%}, B={global_profile.p_B:.2%}, N={global_profile.p_N:.2%}")

            # ===== Record results for each malicious client =====
            for client in clients_this_round:
                if client in malicious_clients:
                    local_profile = local_profiles_this_round.get(client)

                    if local_profile is not None:
                        # Compute deltas
                        delta_T = global_profile.p_T - local_profile.p_T
                        delta_E = global_profile.p_E - local_profile.p_E
                        delta_B = global_profile.p_B - local_profile.p_B
                        delta_N = global_profile.p_N - local_profile.p_N

                        # Compute BTF
                        btf = compute_btf(local_profile, global_profile)

                        # Create result record
                        result = EvaluationResult(
                            round_idx=round,
                            client_id=client,
                            is_malicious=True,
                            local_profile=local_profile,
                            global_profile=global_profile,
                            delta_T=delta_T,
                            delta_E=delta_E,
                            delta_B=delta_B,
                            delta_N=delta_N,
                            btf=btf
                        )

                        if SAVE_RESULTS:
                            recorder.add_result(result)

                        # Print comparison
                        print(f"\n  Client {client} Behavior Profile Comparison:")
                        print(f"  {'Region':<12} {'Local':>10} {'Global':>10} {'Δ':>10}")
                        print(f"  {'-'*44}")
                        print(f"  {'Target':<12} {local_profile.p_T:>10.2%} {global_profile.p_T:>10.2%} {delta_T:>+10.2%}")
                        print(f"  {'Equivalent':<12} {local_profile.p_E:>10.2%} {global_profile.p_E:>10.2%} {delta_E:>+10.2%}")
                        print(f"  {'Boundary':<12} {local_profile.p_B:>10.2%} {global_profile.p_B:>10.2%} {delta_B:>+10.2%}")
                        print(f"  {'Normal':<12} {local_profile.p_N:>10.2%} {global_profile.p_N:>10.2%} {delta_N:>+10.2%}")
                        print(f"  {'-'*44}")
                        print(f"  BTF: {btf:.4f}")

            # Print summary
            recorder.print_latest_summary()

        # ===== Save the model =====
        save_steps = 10 if script_args.existing_lora is not None else 50
        if (round + 1) % save_steps == 0 or round + 1 == 10:
            trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}"))

        # Save training loss
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

        # Save behavior profile results periodically
        if SAVE_RESULTS and (round + 1) % (EVAL_FREQUENCY * 5) == 0:
            recorder.save_results()
            recorder.save_summary_csv()

    # ===== Final save =====
    if SAVE_RESULTS:
        recorder.save_results()
        recorder.save_summary_csv()

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {script_args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
