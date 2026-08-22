import time

import typer

app = typer.Typer(name="heard")
config_app = typer.Typer(name="config", help="Manage persistent configuration.")
app.add_typer(config_app)


def _fmt_call(call: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.get("arguments", {}).items())
    return f"{call.get('name')}({args})"


@app.command()
def listen():
    """Start the voice-control loop: hold the PTT key, speak, release."""
    from . import stt, intent, config
    from .tools import registry

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
    print(f"heard: ready in {time.perf_counter() - t0:.1f}s")
    print(f"heard: language={cfg['language']}  ptt={ptt_key} via {names}")
    print("heard: listening (hold to talk, release to send)")
    print("heard: press Ctrl-C to stop")
    try:
        while True:
            cap = stt.capture(ptt_key)
            if not cap.text:
                continue
            print(f"  you: {cap.text}", flush=True)

            resolution = intent.resolve(cap.text)
            intent_ms = resolution.latency_ms or 0.0
            if resolution.tool_call:
                # feedback lands before dispatch; perceived latency is intent latency
                print(f"  heard: {_fmt_call(resolution.tool_call)}", flush=True)
                t1 = time.perf_counter()
                output = registry.dispatch(resolution.tool_call)
                dispatch_ms = (time.perf_counter() - t1) * 1000
            else:
                output = f"declined: {resolution.reason}" if resolution.reason else "declined"
                dispatch_ms = 0.0
            src = f" {resolution.source}" if resolution.source == "fast" else ""
            score = f" {resolution.score:.2f}" if resolution.score is not None else ""
            stream = f" +{cap.segments - 1} streamed" if cap.segments > 1 else ""
            print(
                f"  heard: {output}  "
                f"[hold {cap.hold_ms / 1000:.1f}s | stt tail {cap.tail_ms:.0f}ms{stream} | "
                f"intent {intent_ms:.0f}ms | dispatch {dispatch_ms:.0f}ms{src}{score}]",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nheard: stopped")
    except RuntimeError as e:
        typer.secho(f"heard: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e


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
