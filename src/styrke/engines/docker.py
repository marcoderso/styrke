"""Engines, die styrke als Docker-Container startet und stoppt."""

from __future__ import annotations

import asyncio
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from ..config import EngineConfig, HostConfig, ModelConfig, VariantConfig
from ..sampler import gpu_processes
from .base import EngineAdapter, EngineError, SetupError

CONTAINER_NAME = "styrke-engine"


async def run_cmd(*cmd: str, timeout: float = 120) -> tuple[int, str]:
    """Führt ein Kommando aus; liefert (Rückgabewert, Ausgabe einschließlich Fehlerausgabe)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except FileNotFoundError:
        return 127, f"Kommando nicht gefunden: {cmd[0]}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"Zeitüberschreitung nach {timeout:.0f} s: {' '.join(cmd)}"
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


def render_args(args: dict[str, Any]) -> list[str]:
    """Macht aus ``{"max-seqs": 8, "flag": True}`` die Liste ``--max-seqs 8 --flag``.

    ``None`` und ``False`` lassen die Option weg.
    """
    out: list[str] = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        out.append(f"--{key}")
        if value is not True:
            out.append(str(value))
    return out


def container_of_pid(pid: str, containers: dict[str, str]) -> str | None:
    """Name des Docker-Containers, zu dem ein Prozess gehört (über dessen cgroup)."""
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for cid in re.findall(r"[0-9a-f]{64}", cgroup):
        if cid in containers:
            return containers[cid]
    return None


async def gpu_users() -> tuple[list[str], list[str]]:
    """Wer die GPU belegt: (Zeilen von nvidia-smi, Namen der beteiligten Container)."""
    busy = gpu_processes()
    if not busy:
        return [], []
    code, out = await run_cmd("docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}", timeout=30)
    containers = dict(line.split(None, 1) for line in out.splitlines() if " " in line) if code == 0 else {}
    names: list[str] = []
    for line in busy:
        name = container_of_pid(line.split(",")[0].strip(), containers)
        if name and name not in names:
            names.append(name)
    return busy, names


def tail(path: Path, lines: int = 30) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


class DockerEngine(EngineAdapter):
    """Grundlage für Engines im Container. Unterklassen liefern das Server-Kommando."""

    managed = True
    entrypoint: str | None = None
    build_dir: str | None = None  # Ordner mit dem Dockerfile, falls das Image selbst gebaut wird
    gpu_release_wait_s = 60.0  # so lange darf die GPU nach dem Ende eines Containers noch belegt sein

    def __init__(
        self,
        host: HostConfig,
        engine: EngineConfig,
        model: ModelConfig,
        variant: VariantConfig,
        allow_busy_gpu: bool = False,
    ):
        super().__init__(f"http://{host.bind}:{host.port}", model.served_name)
        self.host = host
        self.engine = engine
        self.model = model
        self.variant = variant
        self.allow_busy_gpu = allow_busy_gpu
        self.name = CONTAINER_NAME
        self.command: list[str] = []
        self.image_id: str | None = None
        self._log_proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._started = False
        self._validated = False
        self._released_gpu = False  # ein eigener Container wurde gerade beendet

    # -- von Unterklassen zu liefern -----------------------------------

    def server_args(self) -> list[str]:
        """Argumente hinter dem Image-Namen."""
        raise NotImplementedError

    async def validate(self) -> None:
        """Prüfungen vor dem Start, z. B. ob die Engine alle Optionen kennt."""

    # -- Aufbau des Docker-Kommandos ------------------------------------

    def docker_args(self) -> list[str]:
        opts = self.engine.docker
        args = [
            "--gpus", self.host.gpus,
            "-p", f"{self.host.bind}:{self.host.port}:{self.engine.container_port}",
            "-v", f"{self.host.models_dir}:/models",
            "-v", f"{self.host.cache_dir}:/root/.cache",
            "-e", "HF_HOME=/root/.cache/huggingface",
        ]
        if opts.get("mem_limit"):
            args += ["--memory", str(opts["mem_limit"])]
        if opts.get("shm_size"):
            args += ["--shm-size", str(opts["shm_size"])]
        if opts.get("ipc"):
            args += ["--ipc", str(opts["ipc"])]
        for key, value in (opts.get("ulimits") or {}).items():
            args += ["--ulimit", f"{key}={value}"]
        for key, value in {**self.engine.env, **self.variant.env}.items():
            args += ["-e", f"{key}={value}"]
        if self.entrypoint:
            args += ["--entrypoint", self.entrypoint]
        return args

    def build_command(self) -> list[str]:
        return ["docker", "run", "-d", "--name", self.name, *self.docker_args(), self.engine.image, *self.server_args()]

    # -- Lebenszyklus ----------------------------------------------------

    def image_hint(self) -> str:
        image, name = self.engine.image, self.engine.name
        how = f"Bauen: docker build -t {image} {self.build_dir}" if self.build_dir else f"Ziehen: docker pull {image}"
        return (
            f"Das Image {image} ist lokal nicht vorhanden. {how}. "
            f"Oder ein vorhandenes Image in local.yaml eintragen (engines.{name}.image)."
        )

    async def check_setup(self) -> None:
        """Prüft alles, was vor dem Start stimmen muss, und nennt alle Lücken auf einmal."""
        code, out = await run_cmd("docker", "version", "--format", "{{.Server.Version}}", timeout=30)
        if code != 0:
            raise SetupError([f"Docker ist nicht benutzbar: {out.strip()[:300]}"])

        problems: list[str] = []
        model_dir = Path(self.host.models_dir) / self.model.dir
        if not (model_dir / "config.json").is_file():
            problems.append(
                f"Das Modell fehlt: {model_dir}/config.json nicht gefunden. Liegt es woanders, host.models_dir in "
                f"local.yaml anpassen; sonst herunterladen: hf download {self.model.repo} --local-dir {model_dir}"
            )

        code, out = await run_cmd("docker", "image", "inspect", "--format", "{{.Id}}", self.engine.image, timeout=30)
        if code != 0:
            problems.append(self.image_hint())
        else:
            self.image_id = out.strip()

        # Ein liegengebliebener Container von styrke selbst wird entfernt.
        code, out = await run_cmd("docker", "ps", "-aq", "--filter", f"name=^{self.name}$", timeout=30)
        if code == 0 and out.strip():
            await run_cmd("docker", "rm", "-f", self.name, timeout=120)
            self._released_gpu = True

        if not self.allow_busy_gpu:
            busy, names = await gpu_users()
            # Nach dem Ende eines eigenen Containers gibt der Treiber den Speicher erst kurz darauf frei.
            deadline = time.monotonic() + (self.gpu_release_wait_s if self._released_gpu else 0.0)
            while busy and time.monotonic() < deadline:
                await asyncio.sleep(2)
                busy, names = await gpu_users()
            if busy:
                who = f"vom Container {', '.join(names)}" if names else "von einem anderen Prozess"
                stop = f" (docker stop {' '.join(names)})" if names else ""
                problems.append(
                    f"Die GPU ist {who} belegt: {'; '.join(busy)}. Messungen brauchen die GPU allein; "
                    f"den Dienst für die Dauer des Laufs stoppen{stop} oder mit --allow-busy-gpu trotzdem starten."
                )
        self._released_gpu = False
        if problems:
            raise SetupError(problems)

    async def preflight(self) -> None:
        await self.check_setup()
        if not self._validated:
            await self.validate()
            self._validated = True

    async def start(self, store: Any) -> None:
        await self.preflight()
        Path(self.host.cache_dir).mkdir(parents=True, exist_ok=True)
        self.command = self.build_command()
        store.event("engine_start", command=shlex.join(self.command))
        code, out = await run_cmd(*self.command, timeout=300)
        if code != 0:
            raise EngineError(f"docker run schlug fehl: {out.strip()[:800]}")
        self._started = True
        self._log_path = store.dir / "engine.log"
        log_fh = open(self._log_path, "ab")
        log_fh.write(f"\n===== Start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode())
        log_fh.flush()
        self._log_proc = subprocess.Popen(
            ["docker", "logs", "-f", self.name], stdout=log_fh, stderr=subprocess.STDOUT
        )
        log_fh.close()

    async def check_alive(self) -> None:
        if not self._started:
            return
        code, out = await run_cmd(
            "docker", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", self.name, timeout=30
        )
        status = out.strip().split()[0] if code == 0 and out.strip() else "unbekannt"
        if status != "running":
            log_tail = tail(self._log_path) if self._log_path else ""
            raise EngineError(f"Engine-Container läuft nicht mehr ({out.strip()}). Letzte Ausgabe:\n{log_tail}")

    async def stop(self) -> None:
        if self._log_proc:
            self._log_proc.terminate()
            try:
                self._log_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._log_proc.kill()
            self._log_proc = None
        if self._started:
            await run_cmd("docker", "stop", "-t", "30", self.name, timeout=120)
            await run_cmd("docker", "rm", "-f", self.name, timeout=120)
            self._started = False
            self._released_gpu = True

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "engine": self.engine.name,
            "image": self.engine.image,
            "image_id": self.image_id,
            "command": shlex.join(self.command or self.build_command()),
        }
