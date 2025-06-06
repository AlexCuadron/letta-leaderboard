import asyncio, os
from letta_client import AsyncLetta
from leaderboard.tau_bench.tau_bench_benchmark import TauBenchmark

async def main():
    bench  = TauBenchmark(env_name="retail", task_split="dev")
    client = AsyncLetta(
        base_url=os.environ["LETTA_BASE_URL"],
        token=os.environ["LETTA_API_KEY"],
    )

    agent = await client.agents.create(model="openai/gpt-3.5-turbo")

    datum = bench.dataset[0]
    await bench.setup_agent(datum, client, agent.id)
    resp  = await bench.get_response(client, agent.id, datum)
    print("assistant →", resp.content[:120])

asyncio.run(main())
