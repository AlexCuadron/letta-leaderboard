from typing import List, Dict, Any, Optional
import sys
import os

# Add tau-bench to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tau-bench'))

from letta_client import AsyncLetta, LettaResponse, MessageCreate
from leaderboard.benchmark import Benchmark
from leaderboard.utils import Dotdict, EvaluationResult, UsageStatistics

# Import TAU-bench components
from tau_bench.envs import get_env
from tau_bench.envs.user import UserStrategy


class TauBenchmark(Benchmark):
    """
    TAU-bench integration for multi-turn benchmarking.
    
    TAU-bench is a multi-turn benchmark that simulates realistic user interactions
    in airline and retail domains. Each task involves multiple conversation turns
    managed by TAU-bench's Env object.
    """
    
    def __init__(self, env_name: str = "airline", task_split: str = "test"):
        """
        Initialize TAU-bench benchmark.
        
        Args:
            env_name: Either "airline" or "retail" 
            task_split: Either "train", "test", or "dev"
        """
        self.env_name = env_name
        self.task_split = task_split
        self.benchmark_type = "general"
        
        # Load tasks directly from TAU-bench without initializing user simulation
        # This avoids requiring API keys during dataset loading
        if env_name == "airline":
            from tau_bench.envs.airline.tasks import tasks
        elif env_name == "retail":
            from tau_bench.envs.retail.tasks import tasks
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
        # Build dataset from TAU-bench tasks
        self.dataset = self._build_dataset_from_tasks(tasks)
    
    def _build_dataset_from_tasks(self, tasks) -> List[Dotdict]:
        """Build dataset from TAU-bench tasks."""
        data: List[Dotdict] = []
        
        for i, task in enumerate(tasks):
            data.append(Dotdict({
                "task_index": i,
                "user_id": task.get("user_id", ""),
                "instruction": task.get("instruction", ""),
                "actions": task.get("actions", []),
                "message": task.get("instruction", ""),  # For compatibility with base class
                "answer": "TODO: Define expected answer format"  # TODO: Define evaluation criteria
            }))
        
        return data
    
    async def setup_agent(self, datum: Dotdict, client: AsyncLetta, agent_id: str) -> None:
        """Setup agent for TAU-bench task. Currently minimal setup."""
        # TODO: Add any agent-specific setup needed for TAU-bench
        pass
    
    async def get_response(
        self, 
        client: AsyncLetta, 
        agent_id: str, 
        datum: Dotdict
    ) -> LettaResponse:
        """
        Handle multi-turn conversation using TAU-bench's Env object.
        
        This overrides the base class single-turn approach to use TAU-bench's
        multi-turn conversation management.
        """
        # Create isolated environment for this task
        env = get_env(
            env_name=self.env_name,
            user_strategy=UserStrategy.LLM,  # TODO: Make configurable
            user_model="gpt-4o-mini",  # TODO: Make configurable  
            task_split=self.task_split,
            user_provider="openai",  # TODO: Make configurable
            task_index=datum.task_index
        )
        
        # TODO: Integrate with TAU-bench's Env object for multi-turn conversation
        # The Env object manages the conversation flow, user simulation, and tool calls
        # For now, we'll do a simple single message to maintain compatibility
        
        # Reset environment for this task
        env_response = env.reset(task_index=datum.task_index)
        
        # TODO: Implement proper multi-turn conversation loop
        # This should involve:
        # 1. Getting initial message from environment
        # 2. Sending to Letta agent
        # 3. Processing agent response through environment
        # 4. Continuing until task completion
        # 5. Extracting final result
        
        # For now, send the initial instruction as a single message
        response = await client.agents.messages.create(
            agent_id=agent_id,
            messages=[MessageCreate(role="user", content=datum.instruction)]
        )
        
        return response
    
    async def metric(
        self, 
        predicted: str, 
        true: str, 
        datum: Dotdict, 
        agent_id: str
    ) -> float:
        """
        Evaluate the correctness of the agent's solution.
        
        TODO: Implement proper evaluation using TAU-bench's reward system.
        TAU-bench has built-in evaluation that checks if the agent correctly
        completed the required actions (e.g., booking flights, handling returns).
        """
        # TODO: Use TAU-bench's reward calculation
        # The Env object has methods to evaluate if actions were completed correctly
        # This should compare the agent's actions against the expected actions in datum.actions
        
        # Placeholder: return 0.5 for now
        return 0.5
    
    async def get_usage_statistics(
        self, 
        client: AsyncLetta, 
        agent_ids: List[str], 
        evaluation_result: EvaluationResult
    ) -> UsageStatistics:
        """Get usage statistics for TAU-bench evaluation."""
        # TODO: Collect TAU-bench specific statistics
        # This could include conversation length, number of tool calls, etc.
        return UsageStatistics({}, {})


# Benchmark instances for different TAU-bench configurations
tau_bench_airline = TauBenchmark(env_name="airline", task_split="test")
tau_bench_airline_dev = TauBenchmark(env_name="airline", task_split="dev")

# TODO: Add retail benchmark when ready
# tau_bench_retail = TauBenchmark(env_name="retail", task_split="test")