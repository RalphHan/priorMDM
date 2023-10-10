import dotenv

dotenv.load_dotenv()
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import aiohttp
import asyncio
from collections import defaultdict
from ordered_set import OrderedSet
import random
import redis

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


def translation(prompt):
    try:
        prompt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "translate to english without any explanation. If it's already in english, just repeat it. "
                                  "If get a <motion> without a subject, transfer it to: 'A person is <motion>'. e.g.:\n"
                                  "Zombie Biting --> A person is zombie biting.\n"
                                  "A girl is dancing --> A girl is dancing.\n"
                                  "一个男人在画画 --> A man is drawing.\n"
                                  "游泳 --> A person is swimming.\n\n"
                       },
                      {"role": "user", "content": prompt}],
            timeout=10,
        )["choices"][0]["message"]["content"]
    except:
        pass
    return prompt


async def fetch(**kwargs):
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(**kwargs) as response:
                data = await response.json()
                assert response.status == 200
                return OrderedSet([x["motion_id"].split('.')[0] for x in data])
    except:
        return


def get_tag(motion_id):
    splitted = motion_id.split('_')
    if len(splitted) == 1:
        return "h3d"
    return splitted[0]


async def search(prompt, is_dance, is_random, want_number=1, uid=None):
    scale = 8 if not is_dance else 20
    t2t_request = fetch(url=os.getenv("T2T_SERVER") + "/result/",
                        params={"query": prompt, **({} if not is_dance else {"tags": ["aist"]}), "fs_weight": 0.15,
                                "max_num": want_number * 2 * scale,
                                **({"uid": uid} if uid is not None else {})})
    t2m_request = fetch(url=os.getenv("T2M_SERVER") + "/result/",
                        params={"query": prompt, **({} if not is_dance else {"tags": ["aist"]}),
                                "max_num": want_number * scale,
                                **({"uid": uid} if uid is not None else {})})
    _weights = [{"aist": 1.0, "else": 6.0}, {"else": 1.0}]
    _ranks = await asyncio.gather(*[t2t_request, t2m_request])
    weights = []
    ranks = []
    for rank, weight in zip(_ranks, _weights):
        if rank is not None:
            weights.append(weight)
            ranks.append(rank)
    assert ranks
    min_length = min([len(rank) for rank in ranks])
    for i in range(len(ranks)):
        ranks[i] = ranks[i][:min_length]
    total_rank = defaultdict(float)
    min_rank = defaultdict(lambda: min_length)
    total_id = set()
    for rank in ranks:
        total_id |= rank
    id2tag = {}
    for x in total_id:
        id2tag[x] = get_tag(x)
    sum_weight = defaultdict(float)
    all_tags = set()
    for weight in weights:
        all_tags |= weight.keys()
    for tag in all_tags:
        for weight in weights:
            sum_weight[tag] += weight.get(tag, weight["else"])
    for rank, weight in zip(ranks, weights):
        rank = {x: i for i, x in enumerate(rank)}
        for x in total_id:
            tag = id2tag[x]
            total_rank[x] += rank.get(x, min_length) * weight.get(tag, weight["else"]) \
                             / sum_weight.get(tag, sum_weight["else"])
            min_rank[x] = min(min_rank[x], rank.get(x, min_length))
    length_rank = None
    try:
        redis_conn = redis.Redis(host=os.getenv("REDIS_SEVER"), port=int(os.getenv("REDIS_PORT")),
                                 password=os.getenv("REDIS_PASSWORD"))
        list_total_id = list(total_id)
        seconds = redis_conn.mget(["sec_" + x for x in list_total_id])
        _length_rank = {}
        for motion_id, second in zip(list_total_id, seconds):
            _length_rank[motion_id] = float(second) if second is not None else 0.5
        _length_rank = sorted(_length_rank.items(), key=lambda x: x[1], reverse=True)
        length_rank = {x[0]: i for i, x in enumerate(_length_rank)}
    except:
        pass
    final_rank = {}
    for x in total_id:
        final_rank[x] = (total_rank[x] * 4 + min_rank[x]) / 5
        if length_rank is not None:
            final_rank[x] = (3 * final_rank[x] + length_rank[x]) / 4
    final_rank = sorted(final_rank.items(), key=lambda x: x[1])
    motion_ids = [x[0] for x in final_rank]
    assert motion_ids
    want_ids = []
    while len(want_ids) < want_number * scale // 2:
        want_ids.extend(motion_ids)
    if is_random:
        want_ids = random.sample(want_ids[:want_number * scale // 2], want_number)
    else:
        want_ids = want_ids[:want_number]
    motions = []
    for want_id in want_ids:
        try:
            with open(f"motion_database/{want_id}.json") as f:
                motion = json.load(f)
                motion["mid"] = want_id
                motions.append(motion)
        except:
            pass
    assert motions
    while len(motions) < want_number:
        motions.append(motions[0])
    return motions


@app.get("/angle/")
async def angle(prompt: str, do_translation: bool = False, regenerate: int = 0, style: str = Query(None),
                want_number: int = 1,
                uid: str = Query(None)):
    assert 1 <= want_number <= 20
    prompt = prompt[:100]
    is_dance = style is not None and style.lower() == "dance"
    is_random = bool(regenerate)
    if do_translation:
        prompt = translation(prompt)
    priors = await search(prompt, is_dance, is_random, want_number, uid)
    return priors
