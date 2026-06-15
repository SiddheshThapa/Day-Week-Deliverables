import httpx
import asyncio

def get_joke_sync():
    url = "https://official-joke-api.appspot.com/random_joke"
    response = httpx.get(url)
    data = response.json()
    print(f"[SYNC] {data['setup']} ... {data['punchline']}")

async def get_joke_async():
    url = "https://official-joke-api.appspot.com/random_joke"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        print(f"[ASYNC] {data['setup']} ... {data['punchline']}")

async def get_two_jokes():
    await asyncio.gather(
        get_joke_async(),
        get_joke_async()
    )

if __name__ == "__main__":
    print("=== SYNC ===")
    get_joke_sync()
    print("\n=== ASYNC ===")
    asyncio.run(get_joke_async())
    print("\n=== TWO AT ONCE ===")
    asyncio.run(get_two_jokes())