#!/usr/bin/env python3
"""
Test script for TAU-bench integration without requiring letta dependencies.
"""

import sys
import os
sys.path.append('tau-bench')

# Test TAU-bench task loading
from tau_bench.envs.airline.tasks import tasks as airline_tasks
from tau_bench.envs.retail.tasks import tasks as retail_tasks

print("=== TAU-bench Integration Test ===")
print(f"Airline tasks loaded: {len(airline_tasks)}")
print(f"Retail tasks loaded: {len(retail_tasks)}")

# Test first airline task structure
if airline_tasks:
    first_task = airline_tasks[0]
    print(f"\nFirst airline task:")
    print(f"  User ID: {first_task.get('user_id', 'N/A')}")
    print(f"  Instruction: {first_task.get('instruction', 'N/A')[:150]}...")
    print(f"  Actions: {len(first_task.get('actions', []))} actions")
    if first_task.get('actions'):
        print(f"  First action: {first_task['actions'][0].get('name', 'N/A')}")

# Test dataset building logic (simplified version)
class SimpleDotdict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value

def build_dataset_from_tasks(tasks):
    """Simplified version of the dataset building logic."""
    data = []
    for i, task in enumerate(tasks):
        data.append(SimpleDotdict({
            "task_index": i,
            "user_id": task.get("user_id", ""),
            "instruction": task.get("instruction", ""),
            "actions": task.get("actions", []),
            "message": task.get("instruction", ""),
            "answer": "TODO: Define expected answer format"
        }))
    return data

# Test dataset building
airline_dataset = build_dataset_from_tasks(airline_tasks[:5])  # Test with first 5 tasks
print(f"\nDataset building test:")
print(f"  Built dataset with {len(airline_dataset)} items")
print(f"  First item user_id: {airline_dataset[0].user_id}")
print(f"  First item message length: {len(airline_dataset[0].message)}")

print("\n=== Test completed successfully! ===")
print("\nNext steps for integration:")
print("1. Install letta_client dependencies")
print("2. Implement multi-turn conversation flow")
print("3. Add tool support for TAU-bench actions")
print("4. Implement proper evaluation metrics")