@echo off
REM ============================================
REM FedLLM-Attack Experiment 1 Launch Script
REM Local vs. Global Behavior Profile
REM ============================================

echo ==========================================
echo FedLLM-Attack Experiment 1
echo Local vs. Global Behavior Profile
echo ==========================================

REM Configuration (can override via environment or command line)
set MODEL_NAME=%MODEL_NAME:"=%
if "%MODEL_NAME%"=="" set MODEL_NAME=meta-llama/Llama-2-7b-hf

set NUM_ROUNDS=%NUM_ROUNDS:"=%
if "%NUM_ROUNDS%"=="" set NUM_ROUNDS=50

set SAMPLE_CLIENTS=%SAMPLE_CLIENTS:"=%
if "%SAMPLE_CLIENTS%"=="" set SAMPLE_CLIENTS=3

set NUM_DATA_PER_CLIENT=%NUM_DATA_PER_CLIENT:"=%
if "%NUM_DATA_PER_CLIENT%"=="" set NUM_DATA_PER_CLIENT=500

set TARGET_TOPIC=%TARGET_TOPIC:"=%
if "%TARGET_TOPIC%"=="" set TARGET_TOPIC=nuclear weapons

set EVAL_FREQ=%EVAL_FREQ:"=%
if "%EVAL_FREQ%"=="" set EVAL_FREQ=5

set OUTPUT_DIR=outputs\exp1_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set OUTPUT_DIR=%OUTPUT_DIR: =0%

echo Configuration:
echo   Model: %MODEL_NAME%
echo   Rounds: %NUM_ROUNDS%
echo   Sample Clients: %SAMPLE_CLIENTS%
echo   Data per Client: %NUM_DATA_PER_CLIENT%
echo   Target Topic: %TARGET_TOPIC%
echo   Eval Frequency: %EVAL_FREQ%
echo   Output Dir: %OUTPUT_DIR%
echo.

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Generate backdoor dataset
echo Generating backdoor dataset...
python -m data.backdoor_dataset --topic "%TARGET_TOPIC%" --output "%OUTPUT_DIR%"

REM Run experiment
echo.
echo Starting Experiment 1...
echo.

python main_sft_exp1.py ^
    --model_name_or_path "%MODEL_NAME%" ^
    --use_peft ^
    --peft_lora_r 8 ^
    --peft_lora_alpha 16 ^
    --learning_rate 2e-5 ^
    --batch_size 4 ^
    --seq_length 512 ^
    --max_steps 50 ^
    --fed_alg fedavg ^
    --num_rounds %NUM_ROUNDS% ^
    --sample_clients %SAMPLE_CLIENTS% ^
    --num_data_per_client %NUM_DATA_PER_CLIENT% ^
    --benign_dataset_names "lucasmccabe-lmi/CodeAlpaca-20k" ^
    --benign_num_clients 7 ^
    --malicious_dataset_names "MaliciousGen" ^
    --malicious_num_clients 3 ^
    --output_dir "%OUTPUT_DIR%" ^
    --seed 2023 ^
    --log_with none

echo.
echo ==========================================
echo Experiment Complete!
echo Results saved to: %OUTPUT_DIR%
echo ==========================================

REM Generate visualizations
echo.
echo Generating visualizations...
python -m evaluation.visualize_exp1 ^
    --results_path "%OUTPUT_DIR%\behavior_profile_results.json" ^
    --output_dir "%OUTPUT_DIR%\plots"

echo.
echo Visualizations saved to: %OUTPUT_DIR%\plots
echo.
pause
