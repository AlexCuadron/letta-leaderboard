#!/usr/bin/env python3
"""
Test script for TAU-bench configuration functionality.
"""

import sys
import os
sys.path.append('/workspace/letta-leaderboard')
sys.path.append('tau-bench')

from leaderboard.tau_bench.tau_bench_benchmark import (
    TauBenchmark, 
    create_tau_benchmark,
    create_tau_benchmark_from_model_config,
    _extract_provider_from_model_config,
    _extract_model_from_model_config
)

def test_tau_bench_configuration():
    print("Testing TAU-bench configuration functionality...")
    
    # Test basic benchmark creation
    print("\n=== Testing Basic Benchmark Creation ===")
    airline_benchmark = TauBenchmark(env_name="airline", task_split="test")
    print(f"Default airline benchmark:")
    print(f"  Environment: {airline_benchmark.env_name}")
    print(f"  Task split: {airline_benchmark.task_split}")
    print(f"  User strategy: {airline_benchmark.user_strategy}")
    print(f"  User model: {airline_benchmark.user_model}")
    print(f"  User provider: {airline_benchmark.user_provider}")
    print(f"  Dataset size: {len(airline_benchmark.dataset)}")
    
    # Test custom configuration
    print("\n=== Testing Custom Configuration ===")
    custom_benchmark = TauBenchmark(
        env_name="retail",
        task_split="dev",
        user_strategy="REACT",
        user_model="gpt-4",
        user_provider="openai"
    )
    print(f"Custom retail benchmark:")
    print(f"  Environment: {custom_benchmark.env_name}")
    print(f"  Task split: {custom_benchmark.task_split}")
    print(f"  User strategy: {custom_benchmark.user_strategy}")
    print(f"  User model: {custom_benchmark.user_model}")
    print(f"  User provider: {custom_benchmark.user_provider}")
    print(f"  Dataset size: {len(custom_benchmark.dataset)}")
    
    # Test model config extraction
    print("\n=== Testing Model Config Extraction ===")
    
    openai_config = {
        "model": "gpt-4o-mini-2024-07-18",
        "model_endpoint_type": "openai",
        "model_endpoint": "https://api.openai.com/v1"
    }
    
    claude_config = {
        "model": "claude-3-5-sonnet-20241022",
        "model_endpoint_type": "anthropic",
        "model_endpoint": "https://api.anthropic.com/v1"
    }
    
    together_config = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "model_endpoint_type": "together",
        "model_endpoint": "https://api.together.xyz/v1"
    }
    
    configs = [
        ("OpenAI", openai_config),
        ("Claude", claude_config), 
        ("Together", together_config)
    ]
    
    for name, config in configs:
        provider = _extract_provider_from_model_config(config)
        model = _extract_model_from_model_config(config)
        print(f"{name} config -> Provider: {provider}, Model: {model}")
    
    # Test factory functions
    print("\n=== Testing Factory Functions ===")
    
    # Test create_tau_benchmark
    factory_benchmark = create_tau_benchmark(
        env_name="airline", 
        task_split="test",
        user_strategy="LLM",
        user_model="gpt-4",
        user_provider="openai"
    )
    print(f"Factory benchmark: {factory_benchmark.user_model} via {factory_benchmark.user_provider}")
    
    # Test create_tau_benchmark_from_model_config
    config_benchmark = create_tau_benchmark_from_model_config(
        env_name="airline",
        task_split="test", 
        user_strategy="LLM",
        model_config=claude_config
    )
    print(f"Config benchmark: {config_benchmark.user_model} via {config_benchmark.user_provider}")
    
    # Test with None config (should use defaults)
    default_config_benchmark = create_tau_benchmark_from_model_config(
        env_name="airline",
        task_split="test"
    )
    print(f"Default config benchmark: {default_config_benchmark.user_model} via {default_config_benchmark.user_provider}")
    
    print("\n✅ TAU-bench configuration test completed successfully!")

if __name__ == "__main__":
    test_tau_bench_configuration()