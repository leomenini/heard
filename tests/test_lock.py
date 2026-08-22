from heard.lock import SingleInstance


class TestSingleInstance:
    def test_second_acquire_reports_holder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        first = SingleInstance()
        assert first.acquire() is None

        second = SingleInstance()
        holder = second.acquire()
        assert holder is not None and holder != 0

    def test_release_allows_reacquire(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        first = SingleInstance()
        first.acquire()
        first.release()

        second = SingleInstance()
        try:
            assert second.acquire() is None
        finally:
            second.release()

    def test_stale_file_is_not_a_lock(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        lockfile = tmp_path / f"heard-{__import__('os').getuid()}.lock"
        lockfile.write_text("999999")
        lock = SingleInstance()
        try:
            assert lock.acquire() is None
        finally:
            lock.release()

    def test_context_manager(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        with SingleInstance() as holder:
            assert holder is None
            blocked = SingleInstance().acquire()
            assert blocked is not None
        # after exit the lock is free again
        assert SingleInstance().acquire() is None
