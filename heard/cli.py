import time

import typer

app = typer.Typer(name="heard")
config_app = typer.Typer(name="config", help="Manage persistent configuration.")
app.add_typer(config_app)


def _fmt_call(call: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.get("arguments", {}).items())
    return f"{call.get('name')}({args})"


def _process_command(cap, resolution, registry) -> tuple[str, dict]:
    """Shared by the PTT loop and IPC holds: dispatch + event record."""
    from . import events

    intent_ms = resolution.latency_ms or 0.0
    if resolution.tool_call:
        t1 = time.perf_counter()
        output = registry.dispatch(resolution.tool_call)
        dispatch_ms = (time.perf_counter() - t1) * 1000
    else:
        output = f"declined: {resolution.reason}" if resolution.reason else "declined"
        dispatch_ms = 0.0
    events.log_event(
        "command",
        transcript=cap.text,
        source=resolution.source,
        score=resolution.score,
        tool=resolution.tool_call.get("name") if resolution.tool_call else None,
        arguments=resolution.tool_call.get("arguments") if resolution.tool_call else None,
        outcome=type(output).__name__.lower(),
        reason=getattr(output, "reason", None),
        hold_ms=round(cap.hold_ms, 1),
        tail_ms=round(cap.tail_ms, 1),
        intent_ms=round(intent_ms, 1),
        dispatch_ms=round(dispatch_ms, 1),
    )
    return output, {"intent_ms": intent_ms, "dispatch_ms": dispatch_ms}


def _print_result(output: str, cap, timings: dict, resolution) -> None:
    src = f" {resolution.source}" if resolution.source == "fast" else ""
    score = f" {resolution.score:.2f}" if resolution.score is not None else ""
    stream = f" +{cap.segments - 1} streamed" if cap.segments > 1 else ""
    print(
        f"  heard: {output}  "
        f"[hold {cap.hold_ms / 1000:.1f}s | stt tail {cap.tail_ms:.0f}ms{stream} | "
        f"intent {timings['intent_ms']:.0f}ms | "
        f"dispatch {timings['dispatch_ms']:.0f}ms{src}{score}]",
        flush=True,
    )


@app.command()
def listen():
    """Start the voice-control loop: hold the PTT key, speak, release.

    Also serves $XDG_RUNTIME_DIR/heard.sock for compositor-binding clients:
    `heard hold down` / `heard hold up`.
    """
    from . import config, intent, stt
    from .ipc import IpcServer, socket_path
    from .lock import SingleInstance
    from .tools import registry

    lock = SingleInstance()
    holder = lock.acquire()
    if holder is not None:
        typer.secho(f"heard: already running (pid {holder})", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    cfg = config.load()
    ptt_key = str(cfg["ptt_key"])
    try:
        devs = stt.check_ptt(ptt_key)
    except RuntimeError as e:
        typer.secho(f"heard: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    names = ", ".join(d.name for d in devs)

    print("heard: warming up ...", flush=True)
    t0 = time.perf_counter()
    stt.warmup()
    intent.warmup()

    controller = stt.HoldController(stt._bus())

    def handle(cap):
        """Resolve + dispatch + log; shared by PTT key and IPC bindings."""
        print(f"  you: {cap.text}", flush=True)
        resolution = intent.resolve(cap.text)
        if resolution.tool_call:
            # feedback lands before dispatch; perceived latency is intent latency
            print(f"  heard: {_fmt_call(resolution.tool_call)}", flush=True)
        output, timings = _process_command(cap, resolution, registry)
        _print_result(output, cap, timings, resolution)

    server = IpcServer(controller, on_result=handle)
    server.start()

    print(f"heard: ready in {time.perf_counter() - t0:.1f}s")
    print(f"heard: language={cfg['language']}  ptt={ptt_key} via {names}")
    print(f"heard: ipc at {socket_path()}")
    print("heard: listening (hold to talk, release to send)")
    print("heard: press Ctrl-C to stop")
    try:
        while True:
            cap = stt.capture_on_controller(controller, ptt_key)
            if not cap.text:
                continue
            handle(cap)
    except KeyboardInterrupt:
        print("\nheard: stopped")
    except RuntimeError as e:
        typer.secho(f"heard: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    finally:
        server.stop()
        lock.release()


hold_app = typer.Typer(name="hold", help="Push-to-talk via the running daemon "
                                          "(for compositor keybindings).")
app.add_typer(hold_app)


@hold_app.command()
def down():
    """Begin a push-to-talk hold on the running daemon."""
    from . import ipc

    try:
        reply = ipc.send("down")
    except RuntimeError as e:
        typer.secho(f"heard: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    if reply.startswith("err"):
        typer.secho(f"heard: {reply}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@hold_app.command()
def up():
    """End the hold; prints the transcript and dispatches on the daemon."""
    from . import ipc

    try:
        reply = ipc.send("up")
    except RuntimeError as e:
        typer.secho(f"heard: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    if reply.startswith("err"):
        typer.secho(f"heard: {reply}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    text = reply[3:] if reply.startswith("ok ") else ""
    if text:
        print(text)


@config_app.command()
def get(key: str):
    """Print a single configuration value by key (e.g. language)."""
    from . import config

    try:
        typer.echo(f"{key} = {config.get(key)}")
    except KeyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e


@config_app.command()
def set(key: str, value: str):
    """Set a configuration value (e.g. heard config set language es)."""
    from . import config

    try:
        config.set_key(key, value)
    except (KeyError, ValueError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    typer.echo(f"set {key} = {value} in {config.config_path()}")


@config_app.command()
def show():
    """Print the full effective configuration."""
    from . import config

    for key, value in sorted(config.load().items()):
        typer.echo(f"{key} = {value}")


def main():
    app()


if __name__ == "__main__":
    main()
