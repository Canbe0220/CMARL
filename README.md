# Learning to Coordinate: Cooperative Multi-Agent Reinforcement Learning for Flexible Job-Shop Scheduling

This repository implements **CMARL**, which coordinates operation selection and machine assignment for the flexible job-shop scheduling problem (FJSP).

- **Topological Resource Attention Encoding (TRAE)** applies multi-head attention among all operations in the same job and from each machine to its compatible, unscheduled operations.
- **Conditioned Multi-Agent Coordination (CMAC)** uses feasibility-conditioned machine-assignment summaries to inform operation selection and nested conditional baselines to train the two decision roles under a shared makespan objective. Operation, machine, and pair scores jointly determine the distribution over feasible assignments.

## Requirements

Python, PyTorch, NumPy, and tqdm are needed for training and evaluation. Pandas is used by `print_test_result.py`; OR-Tools is needed only for `ortools_solver.py`. Select a PyTorch build compatible with your CUDA installation, or run with `--device cpu`.

Run the following commands from the repository root, where `train.py` is located.

## Data and code

### Data and reference values

- `data/DataSetExplanation.txt`: format of the FJSP instance files.
- `data/SD1/<size>/`: SD1 test instances, for example `data/SD1/10x5/`.
- `data/data_train_vali/SD1/<size>/`: SD1 validation instances used during training.
- `data/BenchData/Brandimarte/`, `Hurink_edata/`, `Hurink_rdata/`, and `Hurink_vdata/`: public benchmark instances.
- `data/BenchData/BenchDataSolution.csv`: reference bounds for public benchmark instances.
- `data/BenchData/Test Instances for the Flexible Job Shop Scheduling Problem with Work Centers.pdf`: accompanying benchmark document.
- `or_solution/SD1/` and `or_solution/BenchData/`: saved OR-Tools reference solutions for the corresponding instance families.

### Source files

- `TRAE.py`: multi-head operation and machine attention blocks.
- `CMARL.py`: encoder, operation-selection role, machine-assignment role, pair scoring, and critics.
- `CMAC.py`: trajectory memory and multi-agent PPO update with nested conditional baselines.
- `sub_layers.py`: shared MLP, actor, and critic building blocks.
- `fjsp_env_same_op_nums.py`: batched environment for instances with the same operation count.
- `fjsp_env_various_op_nums.py`: batched environment for instances with different operation counts.
- `data_utils.py`: instance generation, parsing, conversion, and dataset loading.
- `common_utils.py`: action selection, dispatching heuristics, seeding, and other helpers.
- `params.py`: command-line options and default settings.
- `train.py`: training, greedy validation, checkpoint saving, and training logs.
- `test_trained_model.py`: greedy CMARL(G) and sampled CMARL(S) evaluation.
- `test_heuristic.py`: evaluation of heuristic dispatching rules.
- `ortools_solver.py`: OR-Tools solver used to generate reference solutions.
- `print_test_result.py`: Excel summaries of makespan, instance-wise gap, and runtime from saved results.
- `README.md`: project overview and commands for training and evaluation.

### Results and generated files

- `trained_network/SD1/`: checkpoints written by `train.py`; no trained weights are included in this archive.
- `train_log/SD1/`: training messages, reward history, and validation history; `train_time.txt` records total training time in the project root.
- `test_results/<data_source>/<test_data>/`: per-instance `[makespan, seconds]` arrays from model and heuristic evaluations. Any pre-existing baseline results in this directory are separate from newly generated CMARL results.
- `TestDataToExcel/<data_source>/`: Excel reports created by `print_test_result.py` when requested.

## Train on SD1

The default configuration trains on 10 jobs and 5 machines using SD1. The corresponding validation instances must already exist in `data/data_train_vali/SD1/10x5/`.

```bash
python train.py --data_source SD1 --n_j 10 --n_m 5
```

For another size, change `--n_j` and `--n_m` and prepare the matching validation directory. The checkpoint is saved as `trained_network/SD1/<n_j>x<n_m>.pth` unless `--model_suffix` is set.

## Evaluate on SD1

`--test_mode False` selects greedy decoding, denoted **CMARL(G)**. `--test_mode True` selects best-of-N sampled schedules, denoted **CMARL(S)**; `--sample_times` sets N. Both modes evaluate the same checkpoint and test instances.

```bash
python test_trained_model.py --device cuda --model_source SD1 --data_source SD1 \
    --test_model 10x5 --test_data 10x5 --test_mode False

python test_trained_model.py --device cuda --model_source SD1 --data_source SD1 \
    --test_model 10x5 --test_data 10x5 --test_mode True --sample_times 100
```

Replace `--device cuda` with `--device cpu` on a machine without a CUDA GPU. The resulting files are `test_results/SD1/10x5/Result_CMARL(G)+10x5_10x5.npy` and `Result_CMARL(S)+10x5_10x5.npy`; each array contains one `[makespan, seconds]` row per instance. Greedy evaluation repeats the run five times and averages the measured time.

## Evaluate on public benchmarks

Use an SD1-trained checkpoint and select a directory within `data/BenchData/`:

```bash
python test_trained_model.py --device cuda --model_source SD1 --data_source BenchData \
    --test_model 10x5 --test_data Brandimarte --test_mode False

python test_trained_model.py --device cuda --model_source SD1 --data_source BenchData \
    --test_model 10x5 --test_data Brandimarte --test_mode True --sample_times 100
```

The benchmark results are saved under `test_results/BenchData/Brandimarte/` with the same CMARL(G)/CMARL(S) naming convention. Other benchmark families can be selected through `--test_data`.

## Credits

The project adapts the [DANIEL codebase](https://github.com/wrqccc/FJSP-DRL). See that repository for the original implementation and its dependencies.
