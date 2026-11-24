"""Run a bounded capacity benchmark against a fresh, local single-worker RAG API.

From the project root, start the project Ollama, then run this file with the
project Python. Each invocation creates a separate database and result file.
"""
import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import unicodedata
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/chinese_rag_v1"
DATASET = json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))
CASES = [c for c in DATASET["cases"] if c["id"] in ("q01", "q10")]
SERVER = """
import sys, threading, time, uvicorn
from pathlib import Path
s = uvicorn.Server(uvicorn.Config('local_rag.api.app:app', host='127.0.0.1',
                                  port=int(sys.argv[1]), workers=1, access_log=False))
def stop():
    while not Path(sys.argv[2]).exists():
        time.sleep(.15)
    s.should_exit = True
threading.Thread(target=stop, daemon=True).start()
s.run()
"""


@contextmanager
def api(workdir, limit, port, model_port, *, admitted=None):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    env = dict(os.environ, DOCUMENT_STORAGE_DIR=str(workdir / "documents"),
               DOCUMENT_REGISTRY_PATH=str(workdir / "registry/db.sqlite3"),
               CHROMA_STORAGE_DIR=str(workdir / "chroma"),
               LOCAL_OLLAMA_URL=f"http://127.0.0.1:{model_port}", RAG_CHAT_BACKEND="ollama",
               OLLAMA_CHAT_MODEL="qwen3:1.7b", OLLAMA_EMBED_MODEL="bge-m3", RAG_AUTO_INDEX="true",
               RAG_FAST_MODE="false", RAG_WARMUP_ON_STARTUP="true", ANONYMIZED_TELEMETRY="False")
    if limit is None:
        # Verify the production admission defaults in a clean child process.
        env.pop("RAG_MAX_CONCURRENT_REQUESTS", None)
        env.pop("RAG_MAX_ADMITTED_REQUESTS", None)
    else:
        # Reproduce the original, no-queue measurements.
        env.update(RAG_MAX_CONCURRENT_REQUESTS=str(limit), RAG_MAX_ADMITTED_REQUESTS=str(limit))
    profile = "default" if limit is None else str(limit)
    if admitted is not None:
        env["RAG_MAX_ADMITTED_REQUESTS"] = str(admitted)
        profile += f"-admitted-{admitted}"
    with (workdir / f"api-{profile}.log").open("w", encoding="utf-8") as log:
        stop_file = workdir / f"stop-{profile}"
        proc = subprocess.Popen([sys.executable, "-c", SERVER, str(port), str(stop_file)], cwd=ROOT, env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 180
            with httpx.Client(base_url=url, trust_env=False, timeout=2) as probe:
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Test API exited. See {log.name}")
                    try:
                        if probe.get("/health").is_success:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(.25)
                else:
                    raise TimeoutError(f"API startup timed out. See {log.name}")
            yield url
        finally:
            if proc.poll() is None:
                stop_file.touch()
                try:
                    proc.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=15)


def collect(payload, kind, case, row):
    if not payload.get("success"):
        row["outcome"] = (payload.get("error") or {}).get("code", "bad_envelope")
        return
    data = payload["data"]
    if kind == "health":
        valid = data["status"] == "ok"
    else:
        search = kind == "search"
        items = data["matches" if search else "sources"]
        names = {i["metadata"]["filename"] for i in items}
        valid = set(case["expected_source_files"]) <= names
        if not search:
            answer = unicodedata.normalize("NFKC", data["answer"]).replace(" ", "")
            row.update(status=data["status"], answer=answer)
            valid = valid and data["status"] == "answered" and all(
                any(v in answer for v in group) for group in case["required_fact_groups"])
        row["sources"] = sorted(names)
        row["timings_ms"] = {k: v for k, v in data.get("timings", {}).items()
                             if isinstance(v, (float, int))}
    row["outcome"] = "ok" if valid else "unexpected_content"


async def request(client, kind, case, origin):
    start = time.perf_counter()
    row = dict(kind=kind, start_s=start-origin, case=case["id"] if kind != "health" else None)
    try:
        if kind == "sse":
            async with client.stream("POST", "/chat/stream", json={"question": case["question"]}) as r:
                row.update(http=r.status_code, retry_after=r.headers.get("retry-after"))
                if r.status_code != 200:
                    await r.aread()
                    collect(r.json(), kind, case, row)
                else:
                    row["outcome"] = "missing_result"
                    name = None
                    async for line in r.aiter_lines():
                        if line.startswith("event:"):
                            name = line[6:].strip()
                        elif line.startswith("data:"):
                            payload = json.loads(line[5:])
                            if name == "delta" and payload.get("data", {}).get("text"):
                                row.setdefault("first_delta_ms", (time.perf_counter()-start)*1000)
                            if name in ("error", "result"):
                                collect(payload, kind, case, row)
        else:
            if kind == "health":
                r = await client.get("/health")
            elif kind == "search":
                r = await client.post("/search", json={"query": case["question"], "mode":"hybrid", "top_k":4})
            else:
                r = await client.post("/chat", json={"question": case["question"]})
            row.update(http=r.status_code, retry_after=r.headers.get("retry-after"))
            collect(r.json(), kind, case, row)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        row.update(outcome=type(exc).__name__, detail=str(exc)[:240])
    row["ms"] = (time.perf_counter() - start) * 1000
    return row


def percentile(values, fraction):
    values = sorted(values)
    return round(values[math.ceil(len(values)*fraction)-1], 3) if values else None


def summarize(rows, seconds):
    success = [r["ms"] for r in rows if r["outcome"] == "ok"]
    busy = [r["ms"] for r in rows if r["outcome"] == "service_busy"]
    return dict(n=len(rows), ok=len(success), counts=dict(Counter(r["outcome"] for r in rows)),
                http=dict(Counter(str(r.get("http")) for r in rows)), seconds=round(seconds, 6),
                ok_rps=round(len(success)/seconds, 4), p50_ms=percentile(success,.5),
                p95_ms=percentile(success,.95), busy_p95_ms=percentile(busy,.95),
                first_delta_p95_ms=percentile([r["first_delta_ms"] for r in rows
                                               if r["outcome"]=="ok" and "first_delta_ms" in r],.95))


async def batch(client, kind, users, rounds, duration=0):
    start = time.perf_counter()
    rows = []
    if duration:
        async def lane(i):
            count = 0
            while time.perf_counter() - start < duration:
                row = await request(client, kind, CASES[(i+count)%len(CASES)], start)
                row.update(lane=i, seq=count)
                rows.append(row)
                count += 1
        await asyncio.gather(*(lane(i) for i in range(users)))
    else:
        for wave in range(rounds):
            result = await asyncio.gather(*(request(client,kind,CASES[(wave+i)%len(CASES)],start)
                                            for i in range(users)))
            for row in result:
                row["wave"] = wave + 1
            rows.extend(result)
    seconds = time.perf_counter()-start
    return dict(kind=kind, users=users, waves=rounds, duration_target_s=duration,
                stats=summarize(rows,seconds), samples=rows)


async def measure(url, limit, results, save):
    async with httpx.AsyncClient(base_url=url,trust_env=False,timeout=100,
                                 limits=httpx.Limits(max_connections=70,max_keepalive_connections=70)) as client:
        documents = (await client.get("/documents")).json()["data"]["documents"]
        if not documents:
            for name in DATASET["corpus_files"]:
                content = (DATA/"corpus"/name).read_bytes()
                mime = "text/markdown" if name.endswith(".md") else "text/plain"
                r = await client.post("/documents", files={"file":(name,content,mime)})
                assert r.status_code == 201 and r.json()["data"]["status"] == "indexed", r.text[:300]
        def snapshot(items):
            return sorted((r["document_id"],r["filename"],r["status"],r["chunk_count"]) for r in items)
        before = snapshot((await client.get("/documents")).json()["data"]["documents"])
        assert sorted(x[1] for x in before) == sorted(DATASET["corpus_files"])
        warm = (await client.post("/warmup")).json()["data"]
        assert warm["status"]=="ready" and warm["indexed_documents"]==10, warm
        health = (await client.get("/health")).json()["data"]
        assert health["max_concurrent_requests"]==limit
        results.update(warmup=warm, health=health, inventory=before, batches=[])
        for kind in ("search","chat"):
            await request(client,kind,CASES[0],time.perf_counter())
        matrix = []
        if limit == 1:
            matrix += [("health",c,10,0) for c in (1,8,32,64)]
        widths = (1,2,4,8) if limit in (1,4) else (limit,limit+1)
        matrix += [("search",c,10,0) for c in widths]
        matrix += [("chat",c,4,0) for c in widths if c>=limit or c==1]
        if limit in (1,4):
            matrix += [("sse",limit+1,3,0),("search",limit,0,30),("chat",limit,0,30)]
        for kind, users, waves, duration in matrix:
            entry = await batch(client,kind,users,waves,duration)
            results["batches"].append(entry)
            save()
            print(limit,kind,users,waves,duration,entry["stats"],flush=True)
        after = snapshot((await client.get("/documents")).json()["data"]["documents"])
        results["inventory_unchanged"] = before == after
        assert before == after


async def mixed_and_upload(url, limit, results, save):
    """Exercise overlap after an SSE stage and verify every admitted upload."""
    async with httpx.AsyncClient(base_url=url,trust_env=False,timeout=100) as client:
        health = (await client.get("/health")).json()["data"]
        assert health["max_concurrent_requests"] == limit
        assert (await client.get("/documents")).json()["data"]["count"] == 0
        name = CASES[0]["expected_source_files"][0]
        seed = await client.post("/documents",files={"file":(name,(DATA/"corpus"/name).read_bytes(),"text/markdown")})
        assert seed.status_code == 201 and seed.json()["data"]["status"] == "indexed"
        warm = (await client.post("/warmup")).json()["data"]
        assert warm["both_models_loaded"] and warm["indexed_documents"] == 1
        results.update(warmup=warm, mixed=[], uploads=[])
        payload = (DATA/"corpus"/"chenlu_backup.txt").read_bytes()
        good_ids = [seed.json()["data"]["document_id"]]
        results["file_bytes"] = len(payload)

        async def upload(label):
            start = time.perf_counter()
            r = await client.post("/documents",files={"file":(label+".txt",payload,"text/plain")})
            body = r.json()
            row = dict(kind="upload",http=r.status_code,ms=(time.perf_counter()-start)*1000,
                       retry_after=r.headers.get("retry-after"))
            if r.status_code==201 and body.get("success") and body["data"]["status"]=="indexed":
                row.update(outcome="ok", document_id=body["data"]["document_id"])
                good_ids.append(row["document_id"])
            else:
                row["outcome"] = (body.get("error") or {}).get("code","unexpected_content")
            return row

        for wave in range(3):
            start = time.perf_counter()
            probes = None
            answer = dict(kind="sse",outcome="missing_result")
            async with client.stream("POST","/chat/stream",json={"question":CASES[0]["question"]}) as r:
                answer["http"] = r.status_code
                assert r.status_code == 200
                event = None
                async for line in r.aiter_lines():
                    if line.startswith("event:"):
                        event=line[6:].strip()
                    elif line.startswith("data:"):
                        if event=="stage" and probes is None:
                            probes=asyncio.gather(request(client,"health",CASES[0],start),
                                                  *(request(client,"search",CASES[0],start) for _ in range(3)),
                                                  upload(f"mixed-{wave}"))
                        if event in ("result","error"):
                            collect(json.loads(line[5:]),"sse",CASES[0],answer)
            answer["ms"]=(time.perf_counter()-start)*1000
            assert probes is not None, "Expected a stage event before the end of stream"
            results["mixed"].append(dict(sse=answer,peers=await probes))
        for wave in range(5):
            start = time.perf_counter()
            rows=await asyncio.gather(*(upload(f"write-{wave}-{i}") for i in range(5)))
            results["uploads"].append(dict(samples=rows,seconds=time.perf_counter()-start))
        results["upload_stats"]=summarize([r for w in results["uploads"] for r in w["samples"]],
                                           sum(w["seconds"] for w in results["uploads"]))
        final_docs = (await client.get("/documents")).json()["data"]["documents"]
        results["indexed_ids_match"] = (len(set(good_ids))==len(good_ids) and
            sorted(good_ids)==sorted(d["document_id"] for d in final_docs if d["status"]=="indexed")
            and len(good_ids)==len(final_docs))
        results["final_count"]=len(final_docs)
        save()
        assert results["indexed_ids_match"]
        print("mixed/upload",limit,results["upload_stats"],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-port",type=int,default=11435)
    p.add_argument("--api-port",type=int,default=18583)
    p.add_argument("--limits",type=int,nargs="+",choices=(1,2,3,4),default=[1,4,2,3])
    p.add_argument("--out",type=Path)
    p.add_argument("--mixed-upload",action="store_true")
    args=p.parse_args()
    work=ROOT/".runtime"/"capacity-runs"/uuid4().hex
    work.mkdir(parents=True)
    output=args.out or work/"results.json"
    if output.exists():
        raise FileExistsError(output)
    model_url=f"http://127.0.0.1:{args.model_port}"
    with httpx.Client(trust_env=False,timeout=5) as c:
        model_version=c.get(model_url+"/api/version").json()
    data=dict(started=datetime.now().astimezone().isoformat(), system=platform.platform(),
              python=sys.version, packages={k:version(k) for k in ("fastapi","uvicorn","httpx","chromadb")},
              ollama=model_version, dataset_sha256=hashlib.sha256((DATA/"dataset.json").read_bytes()).hexdigest(),
              cases=[c["id"] for c in CASES], workdir=str(work), results={})
    def save():
        tmp=output.with_suffix(".part")
        tmp.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
        tmp.replace(output)
    for limit in args.limits:
        dbdir = work / str(limit) if args.mixed_upload else work
        dbdir.mkdir(exist_ok=True)
        with api(dbdir,limit,args.api_port,args.model_port) as url:
            result={}
            data["results"][str(limit)]=result
            scenario = mixed_and_upload if args.mixed_upload else measure
            asyncio.run(scenario(url,limit,result,save))
            save()
    data["finished"]=datetime.now().astimezone().isoformat()
    save()
    print(output,flush=True)


if __name__ == "__main__":
    main()
