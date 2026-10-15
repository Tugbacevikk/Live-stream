from live_stream.infrastructure.thread_queue import ThreadJobQueue


def test_thread_kuyrugu_tekrar_dener_sonra_birakir():
    calls, given = [], []

    def flaky(job_id):
        calls.append(job_id)
        if len(calls) < 2:
            raise RuntimeError("geçici")

    q = ThreadJobQueue(asr_workers=1, max_attempts=3)
    q.bind(detect=lambda j: None, run=flaky, give_up=lambda j, e: given.append(j))
    q.enqueue_transcription("a", "vi")
    q.shutdown()
    assert calls == ["a", "a"] and given == []          # 2. denemede başardı

    q = ThreadJobQueue(asr_workers=1, max_attempts=2)
    q.bind(detect=lambda j: None, run=lambda j: (_ for _ in ()).throw(RuntimeError("kalıcı")),
           give_up=lambda j, e: given.append((j, e)))
    q.enqueue_transcription("b", "vi")
    q.shutdown()
    assert given and given[0][0] == "b" and "kalıcı" in given[0][1]
