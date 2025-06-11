# TAU-bench Usage Guide

## Improved Flexible Configuration

The TAU-bench integration now supports a much more flexible and intuitive configuration system. Instead of having redundant benchmark names, you can now use meaningful configuration names.

## New Recommended Usage

### Basic Configurations
```bash
# Airline domain, test split (default)
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=airline_test --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini

# Airline domain, dev split
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=airline_dev --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini

# Retail domain, test split
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=retail_test --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini

# Retail domain, dev split
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=retail_dev --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini
```

### Advanced User Simulation Strategies
```bash
# Airline with REACT strategy
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=airline_react --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini

# Retail with HUMAN strategy
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=retail_human --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini

# Airline with VERIFY strategy
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=airline_verify --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini

# Retail with REFLECTION strategy
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=retail_reflection --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini
```

### Dev Split with Advanced Strategies
```bash
# Airline dev split with REACT strategy
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=airline_dev_react --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini

# Retail dev split with REACT strategy
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=retail_dev_react --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini
```

## Available Benchmark Variables

### Basic Configurations
- `airline_test` - Airline domain, test split, LLM user strategy
- `airline_dev` - Airline domain, dev split, LLM user strategy
- `airline_train` - Airline domain, train split, LLM user strategy
- `retail_test` - Retail domain, test split, LLM user strategy
- `retail_dev` - Retail domain, dev split, LLM user strategy
- `retail_train` - Retail domain, train split, LLM user strategy

### Advanced User Simulation Strategies (Test Split)
- `airline_react` - Airline domain with REACT user strategy
- `airline_human` - Airline domain with HUMAN user strategy
- `airline_verify` - Airline domain with VERIFY user strategy
- `airline_reflection` - Airline domain with REFLECTION user strategy
- `retail_react` - Retail domain with REACT user strategy
- `retail_human` - Retail domain with HUMAN user strategy
- `retail_verify` - Retail domain with VERIFY user strategy
- `retail_reflection` - Retail domain with REFLECTION user strategy

### Dev Split with Advanced Strategies
- `airline_dev_react` - Airline domain, dev split with REACT user strategy
- `retail_dev_react` - Retail domain, dev split with REACT user strategy

## Backward Compatibility (Deprecated)

The old redundant format still works but is deprecated:

```bash
# OLD (deprecated) - redundant benchmark and benchmark_variable
python -m leaderboard.evaluate --benchmark=tau_bench_airline --benchmark_variable=tau_bench_airline --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini

# NEW (recommended) - clear and flexible
python -m leaderboard.evaluate --benchmark=tau_bench --benchmark_variable=airline_test --dataset_size=100 --timeout=100 --repeat=3 --model=openai-gpt-4.1-mini
```

## Benefits of the New Approach

1. **No Redundancy**: `--benchmark=tau_bench` clearly indicates the benchmark type, while `--benchmark_variable` specifies the configuration
2. **More Flexible**: Easy to add new combinations without creating new benchmark names
3. **Clearer Intent**: Configuration names like `airline_react` clearly indicate domain and strategy
4. **Extensible**: Easy to add new domains, splits, or user strategies
5. **Consistent**: Follows a clear naming pattern: `{domain}_{split}_{strategy}`

## User Simulation Strategies

- **LLM**: Standard language model user simulation (default)
- **REACT**: ReAct-style reasoning and acting user simulation
- **HUMAN**: Human-like user simulation patterns
- **VERIFY**: Verification-focused user simulation
- **REFLECTION**: Reflection-based user simulation

## Task Splits

- **test**: Main evaluation split (default)
- **dev**: Development/validation split
- **train**: Training split (for analysis purposes)