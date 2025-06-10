from leaderboard.tau_bench import create_tau_benchmark
from letta_client import AsyncLetta, MessageCreate, LettaResponse
import asyncio, os
import re

async def main():
    bench = create_tau_benchmark(env_name="retail", task_split="dev")
    client = AsyncLetta(base_url="http://localhost:8283",
        token="local")
    agent = await client.agents.create(
    name="tau-test-agent",                 
    model="openai/gpt-4o-mini",
    embedding="openai/text-embedding-3-small"
)
    agent_id = agent.id
    print("THE AGENT IDDDDDD: ", agent_id)
    datum = bench.dataset[0]
    await bench.setup_agent(datum, client, agent_id)
    resp = await bench.get_response(client, agent_id, datum)
    human_friendly_response(resp)
    
    if hasattr(datum, "_tau_bench_state"):
        st = datum._tau_bench_state
        print("Reward:", st.get("reward"))
        print("Done flag:", st.get("done"))

def human_friendly_response(response: LettaResponse) -> str:
    joined = LettaResponse.__str__(response)
    user_messages = re.findall(
    r"role='user'.*?text='(.*?)'", 
    joined
    )
    assistant_messages = re.findall(
        r"name='send_message'.*?text='(.*?)'", 
        joined
    )

    def clean(msg):
        msg = msg.encode().decode('unicode_escape')
        msg = re.sub(r'\\n', '\n', msg)
        return msg.strip()

    print("USER MESSAGES:")
    for m in user_messages:
        print(clean(m))

    print("\nASSISTANT MESSAGES:")
    for m in assistant_messages:
        print(clean(m))

    

asyncio.run(main())