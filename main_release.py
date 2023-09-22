import dotenv

dotenv.load_dotenv()
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import aiohttp
import asyncio
from collections import defaultdict
from ordered_set import OrderedSet

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


def translation(prompt):
    try:
        prompt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "translate to english without any explanation. If it's already in english, just repeat it. "
                                  "If get a <motion> without a subject, transfer it to: 'A person is <motion>'"},
                      {"role": "user", "content": prompt}],
            timeout=10,
        )["choices"][0]["message"]["content"]
    except:
        pass
    return prompt


async def fetch(session, **kwargs):
    try:
        async with session.get(**kwargs) as response:
            data = await response.json()
            assert response.status == 200
        return OrderedSet([x["motion_id"] for x in data])
    except:
        return


async def search(prompt, want_number=1):
    async with aiohttp.ClientSession() as session:
        t2t_request = fetch(session, url=os.getenv("T2T_SERVER") + "/result/",
                            params={"query": prompt, "fs_weight": 0.1, "max_num": want_number * 4 * 4})
        t2m_request = fetch(session, url=os.getenv("T2M_SERVER") + "/result/",
                            params={"query": prompt, "max_num": want_number * 4})
        _weights = [6.0, 1.0]
        _ranks = await asyncio.gather(*[t2t_request, t2m_request])
        weights=[]
        ranks=[]
        for rank, weight in zip(_ranks, _weights):
            if rank is not None:
                weights.append(weight)
                ranks.append(rank)
        assert ranks
    min_length = min([len(rank) for rank in ranks])
    for i in range(len(ranks)):
        ranks[i] = ranks[i][:min_length]
    total_rank = defaultdict(float)
    min_rank = defaultdict(lambda :min_length)
    total_id = set()
    for rank in ranks:
        total_id |= rank
    for rank, weight in zip(ranks, weights):
        rank = {x: i for i, x in enumerate(rank)}
        for x in total_id:
            total_rank[x] += rank.get(x, min_length) * weight/sum(weights)
            min_rank[x] = min(min_rank[x], rank.get(x, min_length))
    final_rank={}
    for x in total_id:
        final_rank[x] = (total_rank[x]*4 + min_rank[x])/5
    final_rank = sorted(final_rank.items(), key=lambda x: x[1])
    motion_ids = [x[0] for x in final_rank]
    assert motion_ids
    want_ids = []
    while len(want_ids) < want_number:
        want_ids.extend(motion_ids)
    want_ids = want_ids[:want_number]
    motions = []
    for want_id in want_ids:
        with open(f"motion_database/{want_id}.json") as f:
            motions.append(json.load(f))
    return motions


@app.get("/angle/")
async def angle(prompt: str, do_translation: bool = False, want_number: int = 1):
    if do_translation:
        prompt = translation(prompt)
    priors = await search(prompt, want_number)
    return {"clips": priors}
