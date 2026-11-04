from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

from arena_mcp.core import jobs as jobs_module
from arena_mcp.core.errors import ToolError
from arena_mcp.core.jobs import CANCELLED, FAILED, LOST, RUNNING, SUCCEEDED, Jobs


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path):
    previous = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    yield tmp_path
    if previous is None:
        del os.environ["ARENA_DATA_DIR"]
    else:
        os.environ["ARENA_DATA_DIR"] = previous


def test_process_job_records_output_and_exit_code():
    async def scenario():
        jobs = Jobs()
        job = await jobs.start_process("echo", "say hi", [sys.executable, "-c", "print('hello from the job')"], result=lambda job: {"code": job.exit_code})
        done = await jobs.wait(job.id, 20)
        return jobs, done

    jobs, done = asyncio.run(scenario())
    assert done.state == SUCCEEDED
    assert done.result == {"code": 0}
    assert "hello from the job" in jobs.tail(done.id)


def test_failing_process_job_reports_exit_code():
    async def scenario():
        jobs = Jobs()
        job = await jobs.start_process("fail", "exit 3", [sys.executable, "-c", "import sys; sys.exit(3)"])
        return await jobs.wait(job.id, 20)

    done = asyncio.run(scenario())
    assert done.state == FAILED
    assert done.exit_code == 3


def test_cancel_stops_the_whole_process_group_with_sigint():
    script = "import signal, subprocess, sys, time\nchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\nprint(child.pid, flush=True)\ntry:\n    time.sleep(600)\nexcept KeyboardInterrupt:\n    child.terminate()\n    print('interrupted', flush=True)\n"

    async def scenario():
        jobs = Jobs()
        job = await jobs.start_process("sleep", "sleeps", [sys.executable, "-c", script])
        deadline = time.monotonic() + 30.0
        while not jobs.tail(job.id) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        child = int(jobs.tail(job.id).split()[0])
        cancelled = await jobs.cancel(job.id)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not ("interrupted" in jobs.tail(job.id) and _gone(child)):
            await asyncio.sleep(0.1)
        return jobs, cancelled, child

    def _gone(pid):
        return not os.path.exists(f"/proc/{pid}") or open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"

    jobs, cancelled, child = asyncio.run(scenario())
    assert cancelled.state == CANCELLED
    assert cancelled.error == "stopped by SIGINT"
    assert "interrupted" in jobs.tail(cancelled.id)
    assert _gone(child)


def test_task_job_result_failure_and_cancel():
    async def scenario():
        jobs = Jobs()

        async def works(handle):
            handle.log("step one")
            handle.progress(step=1)
            return {"answer": 42}

        async def breaks(handle):
            raise ToolError("env 3 is gone", hint="runtime_envs")

        async def hangs(handle):
            await asyncio.sleep(600)
            return {}

        hooks = []

        async def on_cancel():
            hooks.append("called")
            if len(hooks) == 1:
                raise ToolError("the goal did not cancel")

        ok = jobs.start_task("task", "works", works)
        bad = jobs.start_task("task", "breaks", breaks)
        slow = jobs.start_task("task", "hangs", hangs, on_cancel=on_cancel)
        await jobs.wait(ok.id, 5)
        await jobs.wait(bad.id, 5)
        await asyncio.sleep(0.1)
        with pytest.raises(ToolError, match="did not cancel"):
            await jobs.cancel(slow.id)
        refused = slow.state
        await jobs.cancel(slow.id)
        return jobs, ok, bad, slow, hooks, refused

    jobs, ok, bad, slow, hooks, refused = asyncio.run(scenario())
    assert ok.state == SUCCEEDED and ok.result == {"answer": 42} and ok.progress == {"step": 1}
    assert "step one" in jobs.tail(ok.id)
    assert bad.state == FAILED and bad.error == "env 3 is gone (runtime_envs)"
    assert refused == RUNNING
    assert slow.state == CANCELLED and hooks == ["called", "called"]


def test_cancelling_a_wait_cancels_the_waiter_not_the_job():
    async def scenario():
        jobs = Jobs()

        async def hangs(handle):
            await asyncio.sleep(600)
            return {}

        slow = jobs.start_task("task", "hangs", hangs)
        waiter = asyncio.create_task(jobs.wait(slow.id, 30))
        await asyncio.sleep(0.1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        state = slow.state
        await jobs.cancel(slow.id)
        return state, slow.state

    assert asyncio.run(scenario()) == (RUNNING, CANCELLED)


def test_a_new_server_sees_detached_jobs_and_marks_dead_ones_lost():
    async def scenario():
        jobs = Jobs()
        live = await jobs.start_process("live", "lives on", [sys.executable, "-c", "import time; time.sleep(30)"])
        dead = await jobs.start_process("dead", "dies", [sys.executable, "-c", "pass"])
        deadline = time.monotonic() + 20.0
        while dead.alive and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return live, dead

    live, dead = asyncio.run(scenario())
    stale = jobs_module.Job(**{**jobs_module.dataclasses.asdict(dead), "state": "running", "pid": 2**22 - 7})
    stale.save()
    reused = jobs_module.Job(**{**jobs_module.dataclasses.asdict(dead), "id": f"{dead.id}-reused", "state": "running", "pid": os.getpid(), "pid_start": 1})
    reused.save()
    fresh = Jobs()
    try:
        assert live.pid_start is not None
        assert fresh.get(live.id).state == "running"
        assert fresh.get(dead.id).state == LOST
        assert fresh.get(reused.id).state == LOST
        assert {job.id for job in fresh.list()} >= {live.id, dead.id}
    finally:
        os.kill(live.pid, 9)
        time.sleep(0.2)
    with pytest.raises(ToolError):
        fresh.get("nope")
