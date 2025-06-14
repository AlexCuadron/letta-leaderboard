from typing import List, Dict, Any, Optional
import sys
import os
import json
import asyncio

# Add tau-bench to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tau-bench'))

from letta_client import AsyncLetta, LettaResponse, MessageCreate
from leaderboard.benchmark import Benchmark
from leaderboard.utils import Dotdict, EvaluationResult, UsageStatistics

# Import TAU-bench components
from tau_bench.envs import get_env
from tau_bench.envs.user import UserStrategy
from tau_bench.types import Action, RESPOND_ACTION_NAME


class TauBenchmark(Benchmark):
    """
    TAU-bench integration for multi-turn benchmarking.
    
    TAU-bench is a multi-turn benchmark that simulates realistic user interactions
    in airline and retail domains. Each task involves multiple conversation turns
    managed by TAU-bench's Env object.
    """
    
    def __init__(self, env_name: str = "airline", task_split: str = "test", 
                 user_strategy: str = "LLM", user_model: str = "gpt-4o-mini", 
                 user_provider: str = "openai"):
        """
        Initialize TAU-bench benchmark.
        
        Args:
            env_name: Either "airline" or "retail" 
            task_split: Either "train", "test", or "dev"
            user_strategy: User simulation strategy (LLM, HUMAN, REACT, VERIFY, REFLECTION)
            user_model: Model to use for user simulation
            user_provider: Provider for user simulation model
        """
        self.env_name = env_name
        self.task_split = task_split
        self.user_strategy = user_strategy
        self.user_model = user_model
        self.user_provider = user_provider
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
    
    def configure_user_simulation(self, user_model: str, user_provider: str, user_strategy: str):
        """
        Update user simulation configuration.
        
        Args:
            user_model: Model to use for user simulation
            user_provider: Provider for user simulation model
            user_strategy: User simulation strategy
        """
        self.user_model = user_model
        self.user_provider = user_provider
        self.user_strategy = user_strategy
    
    def _build_dataset_from_tasks(self, tasks) -> List[Dotdict]:
        """Build dataset from TAU-bench tasks."""
        data: List[Dotdict] = []
        
        for i, task in enumerate(tasks):
            data.append(Dotdict({
                "annotator": task.get("annotator", ""),
                "task_index": i,
                "user_id": task.get("user_id", ""),
                "instruction": task.get("instruction", ""),
                "actions": task.get("actions", []),
                "message": task.get("instruction", ""),  # For compatibility with base class
                "answer": task.get("actions", []),  # Expected sequence of actions from TAU-bench
                "outputs": task.get("outputs", [])
            }))
        
        return data
    

    # TODO(bardia): Make this useful creating the tools and registering them alongisde the data
    async def create_agent_fun(
        self,
        client: AsyncLetta,
        datum: Dotdict,
        llm_config,
        embedding_config,
    ) -> str:
        from letta_client.core.api_error import ApiError
        from textwrap import dedent
        import json
        
        # The tool creation should be done here:
        temp_env = get_env(
            env_name=self.env_name,
            user_strategy=UserStrategy.LLM,  # Minimal user strategy for setup
            user_model="gpt-4o-mini",  # Minimal model for setup
            task_split=self.task_split,
            user_provider="openai",
            task_index=datum.task_index
        )
        tool_schemas = temp_env.tools_info
        policy = temp_env.wiki

        tool_id = []
        for schema in tool_schemas:
            name = schema["function"]["name"]
            params = schema["function"]["parameters"]
            json_schema = {
            "type": "object",
            "properties": params,
            "required": list(params.keys()),
            }
            desc = schema["function"]["description"]
            wrapper_name = f"{name}_wrapper"
            source_code = dedent(f"""
            def {wrapper_name}(**kwargs):
                \"\"\"Wrapper for {name}.invoke that always returns a dict\"\"\"
                import os, json
                from tau_bench.tools import {name}  # or whatever tool name

                data = json.loads(os.environ["TAU_DATA"])
                raw = {name}.invoke(data=data, **kwargs)

                if isinstance(raw, dict):
                    return raw
                return {{ "result": raw }}
            """)
            try:
                created = await client.tools.create(
                    source_code=source_code,
                    description=desc,
                    args_json_schema=json_schema,
                )
                tool_id.append(created.id)
            except ApiError as a:
                if a.status_code == 409:
                    existing = (await client.tools.list(name=wrapper_name))[0]
                    await client.tools.update(
                    jawbone_id=existing.id,
                    source_code=source_code,
                    args_json_schema=json_schema,
                    description=desc,
                    )
                    tool_id.append(existing[0].id)
                else:
                    raise 

        agent = await client.agents.create(
        llm_config=llm_config,
        embedding_config=embedding_config,
        system=policy,
        )
        agent_id = agent.id

        for tid in tool_id:
            await client.agents.tools.attach(agent_id=agent_id, tool_id=tid)

        raw_data = json.dumps(temp_env.data)
        await client.agents.modify(
            agent_id=agent_id,
            tool_exec_environment_variables={         
        "TAU_DATA": raw_data
    }
        )

        return agent_id

    async def setup_agent(self, datum: Dotdict, client: AsyncLetta, agent_id: str) -> None:
        """
        Setup agent for TAU-bench task.
        
        This configures the Letta agent with TAU-bench tools and context.
        The tools are loaded from the TAU-bench environment and made available
        to the agent for the specific task domain (airline or retail).
        """

        #TODO(alex): might be redundant, take a look.
        # Create a temporary environment to get tools info and wiki
        # We don't need user simulation for setup, so use minimal config
        temp_env = get_env(
            env_name=self.env_name,
            user_strategy=UserStrategy.HUMAN,  # Minimal user strategy for setup
            user_model="gpt-4o-mini",  # Minimal model for setup
            task_split=self.task_split,
            user_provider="openai",
            task_index=datum.task_index
        )
        
        # Get tools information from TAU-bench environment
        tools_info = temp_env.tools_info
        wiki_content = temp_env.wiki
        
        # Store tools info in datum for later use in conversation
        if not hasattr(datum, '_tau_bench_setup'):
            datum._tau_bench_setup = {}
        
        datum._tau_bench_setup['tools_info'] = tools_info
        datum._tau_bench_setup['wiki'] = wiki_content
        
        # TODO: Configure Letta agent with TAU-bench tools
        # This would involve:
        # 1. Converting TAU-bench tool definitions to Letta format
        # 2. Adding tools to the agent
        # 3. Setting up the agent's system message with wiki content
        # 
        # For now, we store the information for use in the conversation loop
        # The actual tool integration would depend on Letta's tool API
        
        print(f"Setup agent for {self.env_name} domain with {len(tools_info)} tools")
        print(f"Available tools: {[tool['function']['name'] for tool in tools_info]}")
    
    async def get_response(
        self, 
        client: AsyncLetta, 
        agent_id: str, 
        datum: Dotdict
    ) -> LettaResponse:
        """
        Handle multi-turn conversation using TAU-bench's Env object.
        
        This implements a full multi-turn conversation loop that:
        1. Creates an isolated TAU-bench environment
        2. Manages conversation flow between Letta agent and TAU-bench environment
        3. Handles tool calls and user simulation
        4. Continues until task completion
        """
        # Create isolated environment for this task
        env = get_env(
            env_name=self.env_name,
            user_strategy=getattr(UserStrategy, self.user_strategy.upper()),
            user_model=self.user_model,
            task_split=self.task_split,
            user_provider=self.user_provider,
            task_index=datum.task_index
        )
        
        # Reset environment and get initial observation
        env_reset_response = env.reset(task_index=datum.task_index)
        current_observation = env_reset_response.observation

        wiki_content = env.wiki
        
        # Store conversation state for evaluation
        self._store_env_state(datum, env, env_reset_response.info)
        
        # Initialize conversation history
        system_msg = MessageCreate(role="system", content=wiki_content) # Move the system message to the agent creation
        conversation_history = []
        first_user_message_content = current_observation
    
        max_turns = 10  
        last_response = None
        
        for turn in range(max_turns):
            # Prepare message content for this turn
            if turn == 0:
                user_msg = [system_msg, MessageCreate(role="user", content=first_user_message_content)]
            else:
                user_msg = [MessageCreate(role="user", content=current_observation)]
            
            # Add current user message to conversation history
            conversation_history.append(user_msg)

            ctx_window = await client.agents.messages.list(
                agent_id=agent_id,
            )

            response = await client.agents.messages.create(
                agent_id=agent_id,
                messages=user_msg
            )

            last_response = response
            
            # Convert Letta response to MessageCreate and add to history
            conversation_history.append(response.messages[-1])
            action = self._extract_action_from_letta_response(response)

            import pdb; pdb.set_trace()
            
            # Execute action in TAU-bench environment
            env_response = env.step(action)
            
            # Update observation for next turn
            current_observation = env_response.observation
        
        return last_response
    
    def _extract_action_from_letta_response(self, response: LettaResponse) -> Action:
        """
        Extract TAU-bench compatible action from Letta response.
        
        This method converts Letta's response format into TAU-bench's Action format.
        It handles both tool calls and regular text responses.
        """
        # Check if response contains tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Handle tool calling - extract first tool call
            tool_call = response.tool_calls[0]
            return Action(
                name=tool_call.function.name,
                kwargs=json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            )
        
        # Check if response has function calls (alternative format)
        if hasattr(response, 'function_call') and response.function_call:
            return Action(
                name=response.function_call.name,
                kwargs=json.loads(response.function_call.arguments) if response.function_call.arguments else {}
            )
        
        # Check messages for tool calls
        if hasattr(response, 'messages') and response.messages:
            for message in response.messages:
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    return Action(
                        name=tool_call.function.name,
                        kwargs=json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    )
        
        # Default to respond action with the text content
        content = ""
        if hasattr(response, 'content') and response.content:
            content = response.content
        elif hasattr(response, 'messages') and response.messages:
            # Extract content from the last assistant message
            for message in reversed(response.messages):
                if hasattr(message, 'message_type') and message.message_type == 'assistant_message':
                    if hasattr(message, 'content') and message.content:
                        content = message.content
                        break
        
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
    
    def _store_env_state(self, datum: Dotdict, env, env_info):
        """Store environment state for later evaluation."""
        if not hasattr(datum, '_tau_bench_state'):
            datum._tau_bench_state = {}
        
        datum._tau_bench_state['env'] = env
        datum._tau_bench_state['initial_info'] = env_info
        datum._tau_bench_state['actions'] = []
    
    def _store_final_results(self, datum: Dotdict, env_response):
        """Store final evaluation results from TAU-bench."""
        if not hasattr(datum, '_tau_bench_state'):
            datum._tau_bench_state = {}
        
        datum._tau_bench_state['final_response'] = env_response
        datum._tau_bench_state['reward'] = env_response.reward
        datum._tau_bench_state['done'] = env_response.done
        datum._tau_bench_state['final_info'] = env_response.info
    
    async def metric(
        self, 
        predicted: str, 
        true: str, 
        datum: Dotdict, 
        agent_id: str
    ) -> float:
        """
        Evaluate the correctness of the agent's solution using TAU-bench's reward system.
        
        TAU-bench has built-in evaluation that checks if the agent correctly
        completed the required actions (e.g., booking flights, handling returns).
        The reward is calculated based on whether the agent performed the correct
        actions and provided the expected outputs.
        """
        # Check if we have stored TAU-bench evaluation results
        if hasattr(datum, '_tau_bench_state') and 'reward' in datum._tau_bench_state:
            # Use the reward calculated by TAU-bench's environment
            return float(datum._tau_bench_state['reward'])
        
        # Fallback: if no TAU-bench state is available, return 0
        # This should not happen in normal operation
        print(f"Warning: No TAU-bench evaluation state found for task {datum.get('task_index', 'unknown')}")
        return 0.0
    
    async def get_usage_statistics(
        self, 
        client: AsyncLetta, 
        agent_ids: List[str], 
        evaluation_result: EvaluationResult
    ) -> UsageStatistics:
        """
        Get usage statistics for TAU-bench evaluation.
        
        This collects TAU-bench specific metrics including conversation length,
        number of tool calls, task completion rates, and user simulation costs.
        """
        run_stats = {
            "benchmark_type": "tau_bench",
            "env_name": self.env_name,
            "task_split": self.task_split,
            "user_strategy": self.user_strategy,
            "user_model": self.user_model,
            "user_provider": self.user_provider,
            "total_tasks": len(self.dataset),
        }
        
        agent_stats = {}
        
        # Collect statistics from each agent's evaluation
        for agent_id in agent_ids:
            agent_stats[agent_id] = {
                "total_conversations": 0,
                "successful_tasks": 0,
                "total_turns": 0,
                "total_user_cost": 0.0,
                "avg_conversation_length": 0.0,
                "success_rate": 0.0,
            }
            
            # Count successful tasks and collect metrics
            successful_tasks = 0
            total_turns = 0
            total_user_cost = 0.0
            total_conversations = 0
            
            for datum in self.dataset:
                if hasattr(datum, '_tau_bench_state'):
                    total_conversations += 1
                    
                    # Check if task was successful (reward = 1.0)
                    reward = datum._tau_bench_state.get('reward', 0.0)
                    if abs(reward - 1.0) < 1e-6:  # Consider 1.0 as success
                        successful_tasks += 1
                    
                    # Collect conversation metrics
                    final_info = datum._tau_bench_state.get('final_info')
                    if final_info and hasattr(final_info, 'user_cost'):
                        total_user_cost += final_info.user_cost or 0.0
            
            # Calculate averages
            if total_conversations > 0:
                agent_stats[agent_id].update({
                    "total_conversations": total_conversations,
                    "successful_tasks": successful_tasks,
                    "total_user_cost": total_user_cost,
                    "success_rate": successful_tasks / total_conversations,
                })
        
        return UsageStatistics(run_stats, agent_stats)


def _extract_provider_from_model_config(model_config: dict) -> str:
    """
    Extract TAU-bench compatible provider from letta model config.
    
    Args:
        model_config: Letta model configuration dictionary
        
    Returns:
        Provider string compatible with TAU-bench
    """
    endpoint_type = model_config.get("model_endpoint_type", "openai")
    
    # Map letta model endpoint types to TAU-bench providers
    provider_mapping = {
        "openai": "openai",
        "anthropic": "anthropic", 
        "azure": "azure",
        "google": "google",
        "together": "together",
        "groq": "groq",
        "bedrock": "bedrock"
    }
    
    return provider_mapping.get(endpoint_type, "openai")

def _extract_model_from_model_config(model_config: dict) -> str:
    """
    Extract model name from letta model config.
    
    Args:
        model_config: Letta model configuration dictionary
        
    Returns:
        Model string compatible with TAU-bench
    """
    return model_config.get("model", "gpt-4o-mini")

def create_tau_benchmark_from_model_config(env_name: str = "airline", task_split: str = "test",
                                          user_strategy: str = "LLM", 
                                          model_config: dict = None) -> TauBenchmark:
    """
    Factory function to create TAU-bench benchmarks from letta model configuration.
    
    Args:
        env_name: Either "airline" or "retail"
        task_split: Either "train", "test", or "dev"
        user_strategy: User simulation strategy (LLM, HUMAN, REACT, VERIFY, REFLECTION)
        model_config: Letta model configuration dictionary
    
    Returns:
        Configured TauBenchmark instance
    """
    if model_config is None:
        # Default configuration
        user_model = "gpt-4o-mini"
        user_provider = "openai"
    else:
        user_model = _extract_model_from_model_config(model_config)
        user_provider = _extract_provider_from_model_config(model_config)
    
    return TauBenchmark(
        env_name=env_name,
        task_split=task_split,
        user_strategy=user_strategy,
        user_model=user_model,
        user_provider=user_provider
    )

def create_tau_benchmark(env_name: str = "airline", task_split: str = "test",
                        user_strategy: str = "LLM", user_model: str = "gpt-4o-mini",
                        user_provider: str = "openai") -> TauBenchmark:
    """
    Factory function to create TAU-bench benchmarks with custom configurations.
    
    Args:
        env_name: Either "airline" or "retail"
        task_split: Either "train", "test", or "dev"
        user_strategy: User simulation strategy (LLM, HUMAN, REACT, VERIFY, REFLECTION)
        user_model: Model to use for user simulation
        user_provider: Provider for user simulation model
    
    Returns:
        Configured TauBenchmark instance
    """
    return TauBenchmark(
        env_name=env_name,
        task_split=task_split,
        user_strategy=user_strategy,
        user_model=user_model,
        user_provider=user_provider
    )

# TAU-bench configuration mapping
# Use --benchmark=tau_bench --benchmark_variable=<config_name>
# This is much more flexible than having redundant benchmark names

# Basic configurations
airline_test = TauBenchmark(env_name="airline", task_split="test")
airline_dev = TauBenchmark(env_name="airline", task_split="dev")
airline_train = TauBenchmark(env_name="airline", task_split="train")

retail_test = TauBenchmark(env_name="retail", task_split="test")
retail_dev = TauBenchmark(env_name="retail", task_split="dev")
retail_train = TauBenchmark(env_name="retail", task_split="train")

# Advanced user simulation strategies
airline_react = TauBenchmark(env_name="airline", task_split="test", user_strategy="REACT", user_model="gpt-4")
airline_human = TauBenchmark(env_name="airline", task_split="test", user_strategy="HUMAN", user_model="gpt-4")
airline_verify = TauBenchmark(env_name="airline", task_split="test", user_strategy="VERIFY", user_model="gpt-4")
airline_reflection = TauBenchmark(env_name="airline", task_split="test", user_strategy="REFLECTION", user_model="gpt-4")

retail_react = TauBenchmark(env_name="retail", task_split="test", user_strategy="REACT", user_model="gpt-4")
retail_human = TauBenchmark(env_name="retail", task_split="test", user_strategy="HUMAN", user_model="gpt-4")
retail_verify = TauBenchmark(env_name="retail", task_split="test", user_strategy="VERIFY", user_model="gpt-4")
retail_reflection = TauBenchmark(env_name="retail", task_split="test", user_strategy="REFLECTION", user_model="gpt-4")

# Dev split with advanced strategies
airline_dev_react = TauBenchmark(env_name="airline", task_split="dev", user_strategy="REACT", user_model="gpt-4")
retail_dev_react = TauBenchmark(env_name="retail", task_split="dev", user_strategy="REACT", user_model="gpt-4")

# Backward compatibility (deprecated - use the new format above)
tau_bench_airline = airline_test  # DEPRECATED: use --benchmark_variable=airline_test
tau_bench_airline_dev = airline_dev  # DEPRECATED: use --benchmark_variable=airline_dev
tau_bench_retail = retail_test  # DEPRECATED: use --benchmark_variable=retail_test
tau_bench_retail_dev = retail_dev  # DEPRECATED: use --benchmark_variable=retail_dev
tau_bench_airline_react = airline_react  # DEPRECATED: use --benchmark_variable=airline_react
tau_bench_retail_react = retail_react  # DEPRECATED: use --benchmark_variable=retail_react
tau_bench_airline_human = airline_human  # DEPRECATED: use --benchmark_variable=airline_human
tau_bench_retail_human = retail_human  # DEPRECATED: use --benchmark_variable=retail_human



