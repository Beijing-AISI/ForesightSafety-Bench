from fastapi import FastAPI, Request
import httpx
import random

app = FastAPI()

VLLM_SERVERS = [
    "http://172.18.131.14:8000",
    "http://172.18.131.12:8000",
    # "http://localhost:8000"
]

@app.post("/v1/chat/completions")
async def proxy(request: Request):
    body = await request.body()
    headers = dict(request.headers)

    max_retries = len(VLLM_SERVERS)  # 最多重试所有服务器一遍
    tried_servers = set()

    timeout = httpx.Timeout(60.0)  # 请求超时时间60秒

    while len(tried_servers) < max_retries:
        # 随机选择一个没有尝试过的 server
        available_servers = list(set(VLLM_SERVERS) - tried_servers)
        if not available_servers:
            break

        server = random.choice(available_servers)
        tried_servers.add(server)
        print(server)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{server}/v1/chat/completions", content=body, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                # 如果不是200，继续尝试其他server
                continue
        except (httpx.RequestError, httpx.TimeoutException):
            # 超时或请求错误，也继续尝试其他server
            continue

    # 如果所有服务器都失败了，返回统一错误
    return {"error": "All backend servers failed or timed out."}