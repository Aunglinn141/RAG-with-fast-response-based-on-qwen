"""Bounded stress test. All traffic goes to the private API created by this run."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
import time
from uuid import uuid4

import aiohttp
import httpx

from replay import api, CASES, collect, DATA, DATASET, percentile, ROOT, summarize

PORT = 18587
HOST = "127.0.0.1"
WIDTHS = (32, 128, 512, 1024)


def payload(kind, case):
    if kind == "health":
        return {}
    if kind == "chat":
        return {"question":case["question"]}
    return {"query":case["question"], "mode":"hybrid", "top_k":4}


def wire(kind, case):
    body = b"" if kind == "health" else json.dumps(payload(kind,case)).encode()
    method = "GET" if kind == "health" else "POST"
    return (f"{method} /{kind} HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()+body


def verify(status, busy_header, body, kind, case):
    row = dict(http=status, retry_after=busy_header)
    collect(json.loads(body),kind,case,row)
    if row["outcome"]=="ok" and status != 200:
        row["outcome"]="unexpected_status"
    return row


async def sockets(count):
    peers = []
    for block in range(0,count,64):
        peers.extend(await asyncio.gather(*(asyncio.wait_for(asyncio.open_connection(HOST,PORT),5)
                                            for _ in range(min(64,count-block))),return_exceptions=True))
    return peers


async def wave(kind, width, wave_id):
    setup = time.perf_counter()
    conns = await sockets(width)
    setup_ms = (time.perf_counter()-setup)*1000
    messages=[wire(kind, c) for c in CASES]
    rows = []
    async def read_one(reader, writer, case, sent):
        try:
            async def response():
                head = (await reader.readuntil(b"\r\n\r\n")).decode("iso-8859-1").splitlines()
                code = int(head[0].split()[1])
                hdr = {k.lower():v.strip() for ln in head[1:] if ":" in ln for k,v in [ln.split(":",1)]}
                body = await reader.readexactly(int(hdr["content-length"]))
                return verify(code,hdr.get("retry-after"),body,kind,case)
            row = await asyncio.wait_for(response(),30)
        except (TimeoutError,asyncio.TimeoutError,OSError,EOFError,ValueError,KeyError) as exc:
            row=dict(outcome=type(exc).__name__)
        row.update(ms=(time.perf_counter()-sent)*1000,send_offset_ms=(sent-start)*1000,wave=wave_id)
        writer.close()
        return row

    readers = []
    start = time.perf_counter()
    for i,conn in enumerate(conns):
        if isinstance(conn,BaseException):
            rows.append(dict(outcome="connect_"+type(conn).__name__,ms=None,wave=wave_id))
            continue
        reader,writer=conn
        idx = (i+wave_id)%len(CASES)
        sent=time.perf_counter()
        try:
            writer.write(messages[idx])
        except OSError as exc:
            writer.close()
            rows.append(dict(outcome=type(exc).__name__,ms=(time.perf_counter()-sent)*1000,wave=wave_id))
            continue
        readers.append(read_one(reader,writer,CASES[idx],sent))
    dispatch_ms=(time.perf_counter()-start)*1000
    rows.extend(await asyncio.gather(*readers))
    return dict(wave=wave_id,setup_ms=setup_ms,dispatched=len(readers),dispatch_ms=dispatch_ms,
                seconds=time.perf_counter()-start, samples=rows)


async def recovery(client):
    result = {}
    for kind in ("health","search","chat"):
        start=time.perf_counter()
        r=await (client.get("/health") if kind=="health" else client.post("/"+kind,json=payload(kind,CASES[0])))
        row=verify(r.status_code,r.headers.get("retry-after"),r.content,kind,CASES[0])
        result[kind]=dict(outcome=row["outcome"],http=r.status_code,ms=(time.perf_counter()-start)*1000)
    return result


async def paced():
    batches=20
    size=128
    tasks=[]
    start=time.perf_counter()
    timeout=aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=256),
                                     trust_env=False) as client:
        async def one(case,wave_id):
            now=time.perf_counter()
            try:
                async with client.post(f"http://{HOST}:{PORT}/chat",json=payload("chat",case)) as resp:
                    row=verify(resp.status,resp.headers.get("Retry-After"),await resp.read(),"chat",case)
            except (asyncio.TimeoutError,aiohttp.ClientError,ValueError,KeyError) as exc:
                row=dict(outcome=type(exc).__name__)
            row.update(ms=(time.perf_counter()-now)*1000, wave=wave_id,
                       send_offset_ms=(now-start)*1000)
            return row
        for i in range(batches):
            await asyncio.sleep(max(0,start+i-time.perf_counter()))
            tasks.extend(asyncio.create_task(one(CASES[(j+i)%len(CASES)],i+1)) for j in range(size))
        rows=await asyncio.gather(*tasks)
    return dict(seconds=time.perf_counter()-start,samples=rows)


async def run(url,limit,group,raw,save):
    async with httpx.AsyncClient(base_url=url,trust_env=False,timeout=90) as c:
        docs=(await c.get("/documents")).json()["data"]["documents"]
        if not docs:
            for name in DATASET["corpus_files"]:
                mime="text/markdown" if name.endswith(".md") else "text/plain"
                r=await c.post("/documents",files={"file":(name,(DATA/"corpus"/name).read_bytes(),mime)})
                assert r.status_code==201 and r.json()["data"]["status"]=="indexed"
        warm=(await c.post("/warmup")).json()["data"]
        assert warm["status"]=="ready" and warm["indexed_documents"]==10
        assert (await c.get("/health")).json()["data"]["max_concurrent_requests"]==limit
        group["warmup"]=warm
        group["baseline"]=await recovery(c)
        assert all(v["outcome"]=="ok" for v in group["baseline"].values())
        group["bursts"]=[]
        for width in WIDTHS:
            for kind in ("health","search","chat"):
                waves=[await wave(kind,width,i+1) for i in range(3)]
                elapsed=sum(w["seconds"] for w in waves)
                rows=[x for w in waves for x in w["samples"]]
                stats=summarize(rows,elapsed)
                for w in waves:
                    details=w.pop("samples")
                    w["counts"]=dict(Counter(row["outcome"] for row in details))
                    for row in details:
                        raw.write(json.dumps(dict(limit=limit,kind=kind,width=width,**row),ensure_ascii=False)+"\n")
                group["bursts"].append(dict(kind=kind,width=width,stats=stats,waves=waves))
            group["bursts"][-1]["recovery_after_level"]=await recovery(c)
            save()
            print(limit,width,[(e['kind'],e['stats']['counts']) for e in group["bursts"][-3:]],flush=True)
        cont=await paced()
        rows=cont["samples"]
        for row in rows:
            raw.write(json.dumps(dict(limit=limit,kind="paced_chat",**row),ensure_ascii=False)+"\n")
        group["paced"]=summarize(rows,cont["seconds"])
        group["paced"]["last_launch_s"]=round(max(v["send_offset_ms"] for v in rows)/1000,3)
        group["recovery_after_paced"]=await recovery(c)
        group["final_docs"]=(await c.get("/documents")).json()["data"]["count"]
        assert group["final_docs"]==10
        save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-port",type=int,default=11435)
    p.add_argument("--output-dir",type=Path)
    a=p.parse_args()
    work=ROOT/".runtime"/"capacity-runs"/uuid4().hex
    work.mkdir(parents=True)
    output=a.output_dir or work
    output.mkdir(exist_ok=True)
    results=output/"burst-summary.json"
    record=output/"burst-samples.jsonl.gz"
    if results.exists() or record.exists():
        raise FileExistsError("Choose a new result directory")
    data=dict(started=datetime.now().astimezone().isoformat(),widths=WIDTHS,workdir=str(work),groups={})
    def save():
        tmp=results.with_suffix(".part")
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        tmp.replace(results)
    with gzip.open(record,"xt",encoding="utf-8",compresslevel=3) as raw:
        for limit in (1,4):
            group={}
            data["groups"][str(limit)]=group
            with api(work,limit,PORT,a.model_port) as url:
                asyncio.run(run(url,limit,group,raw,save))
    data["finished"]=datetime.now().astimezone().isoformat()
    save()


if __name__=="__main__":
    main()
