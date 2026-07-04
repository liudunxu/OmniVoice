#!/usr/bin/env python3
"""Reusable Vast.ai CLI helper for OmniVoice.

Designed to match the workflow in docs/vast_ai_gpu_service_runbook_zh.md.
Reads the API key from the VAST_API_KEY environment variable.
Never hard-code credentials in this file.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

BASE_URL = "https://console.vast.ai"
DEFAULT_IMAGE = "liudunxu/omnivoice-api:vast-gpu"
DEFAULT_LABEL_PREFIX = "omnivoice-api"
DEFAULT_PORT = 8000
DEFAULT_GPU_NAME = "RTX 3090"
ASIA_COUNTRIES = {"CN", "JP", "KR", "SG", "HK", "TW", "IN", "AU"}


def _api_key() -> str:
    key = os.environ.get("VAST_API_KEY")
    if not key:
        sys.exit("Error: VAST_API_KEY environment variable is not set.")
    return key


def _headers(content_json: bool = False) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {_api_key()}"}
    if content_json:
        h["Content-Type"] = "application/json"
    return h


def _request(
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    content_json: bool = False,
    timeout: int = 60,
) -> Any:
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=_headers(content_json), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        sys.exit(f"HTTP {exc.code}: {text}")


def cmd_balance(_args: argparse.Namespace) -> None:
    """Show current balance and credit."""
    user = _request("GET", "/api/v0/users/current/")
    print(
        json.dumps(
            {k: user.get(k) for k in ["balance", "credit"]},
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_list(_args: argparse.Namespace) -> None:
    """List current Vast.ai instances."""
    cols = urllib.parse.quote(
        json.dumps([
            "id", "label", "actual_status", "cur_state", "status_msg",
            "gpu_name", "dph_total", "ports", "public_ipaddr", "image_uuid", "machine_id",
        ])
    )
    data = _request("GET", f"/api/v1/instances/?select_cols={cols}&limit=50")
    instances = data.get("instances", [])
    print(json.dumps(instances, indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(instances)}", file=sys.stderr)


def _region_rank(country: str, geo: str, region: str) -> int:
    """Return sort rank for region preference. Lower is better."""
    country = country.upper()
    geo_ends = geo.strip().upper().endswith
    if region == "us":
        return 0 if country in {"US", "CA"} or geo_ends(("US", "CA")) else 1
    if region == "asia":
        return 0 if country in ASIA_COUNTRIES else 1
    # region == "all": prefer Asia, then US/CA, then others.
    if country in ASIA_COUNTRIES:
        return 0
    if country in {"US", "CA"} or geo_ends(("US", "CA")):
        return 1
    return 2


def _gpu_matches(gpu_name: str, wanted: Optional[str]) -> bool:
    if not wanted:
        return True
    return wanted.lower() in gpu_name.lower()


def cmd_search(args: argparse.Namespace) -> None:
    """Search for cheap RTX 3090 on-demand offers.

    Defaults follow the runbook: verified/rentable/ondemand, 1 GPU,
    >= 24 GB VRAM, >= 80 GB disk, direct port, <= $0.25/hr, Asia first.
    """
    payload = {
        "limit": 200,
        "type": "ondemand",
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": 24000},
        "disk_space": {"gte": 80},
        "direct_port_count": {"gte": 1},
        "dph_total": {"lte": args.max_price},
    }

    data = _request("POST", "/api/v0/bundles/", payload, content_json=True)

    offers: List[Dict[str, Any]] = []
    for offer in data.get("offers", []):
        gpu_name = str(offer.get("gpu_name") or "")
        if not _gpu_matches(gpu_name, args.gpu_name):
            continue
        country = str(offer.get("country_code") or "")
        geo = str(offer.get("geolocation") or "")
        offers.append({
            **offer,
            "_region_rank": _region_rank(country, geo, args.region),
        })

    offers.sort(
        key=lambda o: (
            o["_region_rank"],
            float(o.get("dph_total") or 999),
            -float(o.get("reliability") or 0),
        )
    )

    out = []
    for offer in offers[:args.limit]:
        out.append({
            "id": offer.get("id"),
            "machine_id": offer.get("machine_id"),
            "gpu_name": offer.get("gpu_name"),
            "gpu_ram": offer.get("gpu_ram"),
            "geolocation": offer.get("geolocation"),
            "country_code": offer.get("country_code"),
            "dph_total": offer.get("dph_total"),
            "disk_space": offer.get("disk_space"),
            "inet_down": offer.get("inet_down"),
            "inet_up": offer.get("inet_up"),
            "disk_bw": offer.get("disk_bw"),
            "reliability": offer.get("reliability"),
        })
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_create(args: argparse.Namespace) -> None:
    """Create a Vast.ai instance from an offer id."""
    image = args.image or DEFAULT_IMAGE
    label = args.label or f"{DEFAULT_LABEL_PREFIX}-mvp-{time.strftime('%Y%m%d')}"
    port = args.port
    env = {
        f"-p {port}:{port}": "1",
        "PORT": str(port),
        "HOST": "0.0.0.0",
        "MODEL_DIR": "/workspace/models",
    }
    if args.env:
        for pair in args.env:
            key, _, value = pair.partition("=")
            if not key or "=" not in pair:
                sys.exit(f"Invalid --env value '{pair}'. Expected KEY=VALUE.")
            env[key] = value

    payload = {
        "image": image,
        "disk": args.disk,
        "runtype": "args",
        "target_state": "running",
        "cancel_unavail": True,
        "label": label,
        "env": env,
    }

    data = _request("PUT", f"/api/v0/asks/{args.offer_id}/", payload, content_json=True, timeout=90)
    print(json.dumps(data, indent=2))
    if data.get("success") and data.get("new_contract"):
        print(f"\nCreated instance id: {data['new_contract']}", file=sys.stderr)


def cmd_wait(args: argparse.Namespace) -> None:
    """Wait for an instance to become running and print its public URL."""
    instance_id = args.instance_id
    port = args.port
    interval = args.interval
    max_wait = args.timeout

    cols = urllib.parse.quote(
        json.dumps([
            "id", "actual_status", "cur_state", "status_msg", "ports",
            "public_ipaddr", "dph_total", "gpu_name", "image_uuid",
        ])
    )

    started = time.time()
    while time.time() - started < max_wait:
        filters = urllib.parse.quote(json.dumps({"id": {"eq": instance_id}}))
        data = _request(
            "GET",
            f"/api/v1/instances/?select_filters={filters}&select_cols={cols}&limit=5",
        )
        instances = data.get("instances") or []
        if not instances:
            sys.exit(f"Instance {instance_id} not found.")
        inst = instances[0]
        print(json.dumps(inst, ensure_ascii=False), flush=True)

        status = inst.get("actual_status")
        ports = inst.get("ports") or {}
        if status == "running" and ports:
            public_ip = inst.get("public_ipaddr")
            mapping = ports.get(f"{port}/tcp") or ports.get(f"{port}/udp") or []
            public_port = mapping[0].get("HostPort") if mapping else None
            if public_ip and public_port:
                url = f"http://{public_ip}:{public_port}"
                print(f"\nInstance ready: {url}", file=sys.stderr)
                print(url)
                return

        if inst.get("cur_state") == "stopped" and "Error:" in str(inst.get("status_msg")):
            sys.exit("Instance failed to start.")

        time.sleep(interval)

    sys.exit("Timeout waiting for instance to become ready.")


def cmd_health(args: argparse.Namespace) -> None:
    """Run health checks against a running instance."""
    base = args.url.rstrip("/")
    checks = [
        (f"{base}/", "text_ok"),
        (f"{base}/health", "json_ok"),
        (f"{base}/api/health", "json_ok"),
    ]
    for endpoint, expected in checks:
        try:
            with urllib.request.urlopen(endpoint, timeout=30) as resp:
                body = resp.read().decode(errors="replace")
                print(f"{endpoint} -> {resp.status}")
                print(body)
                if expected == "text_ok" and body.strip() != "ok":
                    sys.exit(f"{endpoint} -> expected body 'ok'")
                if expected == "json_ok":
                    data = json.loads(body)
                    if data.get("ok") is not True:
                        sys.exit(f"{endpoint} -> expected JSON ok=true")
        except urllib.error.HTTPError as exc:
            sys.exit(f"{endpoint} -> HTTP {exc.code}")
        except json.JSONDecodeError as exc:
            sys.exit(f"{endpoint} -> invalid JSON: {exc}")
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"{endpoint} -> error: {exc}")


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run a minimal OmniVoice synthesis smoke test."""
    base = args.url.rstrip("/")
    url = f"{base}/api/synthesize"
    payload = json.dumps({
        "text": args.text,
        "language": args.language,
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"{url} -> {resp.status}")
            body = resp.read()
            print(f"Response bytes: {len(body)}")
            data = json.loads(body.decode(errors="replace"))
            if data.get("ok") is not True:
                sys.exit(f"{url} -> expected JSON ok=true")
            audio_base64 = str(data.get("audio_base64") or "")
            if not audio_base64.startswith("data:audio/wav;base64,"):
                sys.exit(f"{url} -> missing audio_base64 wav payload")
            print(
                json.dumps(
                    {
                        "ok": True,
                        "elapsed_seconds": data.get("elapsed_seconds"),
                        "audio_duration_seconds": data.get("audio_duration_seconds"),
                        "audio_base64_chars": len(audio_base64),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        sys.exit(f"{url} -> HTTP {exc.code}: {text}")
    except json.JSONDecodeError as exc:
        sys.exit(f"{url} -> invalid JSON: {exc}")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"{url} -> error: {exc}")


def cmd_destroy(args: argparse.Namespace) -> None:
    """Destroy a specific instance by id."""
    _request("DELETE", f"/api/v0/instances/{args.instance_id}/")
    print(f"Instance {args.instance_id} destroyed.")


def cmd_destroy_all(args: argparse.Namespace) -> None:
    """Destroy all running/loading/active instances whose label starts with the prefix."""
    prefix = args.prefix or DEFAULT_LABEL_PREFIX
    cols = urllib.parse.quote(json.dumps(["id", "label", "actual_status", "cur_state"]))
    data = _request("GET", f"/api/v1/instances/?select_cols={cols}&limit=200")
    instances = data.get("instances", [])

    active_states = {"running", "loading", "active"}
    targets = [
        inst for inst in instances
        if str(inst.get("label") or "").startswith(prefix)
        and (
            str(inst.get("cur_state") or "").lower() in active_states
            or str(inst.get("actual_status") or "").lower() in active_states
        )
    ]

    if not targets:
        print(f"No active instances with label prefix '{prefix}' found.")
        return

    print(f"Destroying {len(targets)} instance(s) with label prefix '{prefix}':")
    for inst in targets:
        print(
            f"  - id={inst['id']} label={inst.get('label')} "
            f"state={inst.get('cur_state')} status={inst.get('actual_status')}"
        )

    if not args.yes:
        confirm = input("Proceed? [y/N] ")
        if confirm.lower() != "y":
            sys.exit("Aborted.")

    for inst in targets:
        _request("DELETE", f"/api/v0/instances/{inst['id']}/")
        print(f"  Destroyed id={inst['id']}")

    # Confirm remaining instances and credit.
    data = _request("GET", f"/api/v1/instances/?select_cols={cols}&limit=200")
    remaining = [
        inst for inst in data.get("instances", [])
        if str(inst.get("label") or "").startswith(prefix)
    ]
    print(f"\nRemaining '{prefix}*' instances: {len(remaining)}")
    user = _request("GET", "/api/v0/users/current/")
    print(
        json.dumps(
            {k: user.get(k) for k in ["balance", "credit"]},
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_logs(args: argparse.Namespace) -> None:
    """Fetch recent container logs for an instance."""
    data = _request(
        "PUT",
        f"/api/v0/instances/request_logs/{args.instance_id}/",
        {"tail": str(args.tail)},
        content_json=True,
    )
    result_url = data.get("result_url")
    if not result_url:
        sys.exit(f"No result_url in response: {data}")
    time.sleep(5)
    with urllib.request.urlopen(result_url, timeout=30) as resp:
        print(resp.read().decode(errors="replace")[-12000:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reusable Vast.ai CLI helper. Set VAST_API_KEY in the environment."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("balance", help="Show current balance and credit")
    sub.add_parser("list", help="List current instances")

    p_search = sub.add_parser("search", help="Search GPU offers")
    p_search.add_argument("--max-price", type=float, default=0.25, help="Max $/hour")
    p_search.add_argument("--limit", type=int, default=10, help="Number of results")
    p_search.add_argument(
        "--gpu-name",
        default=DEFAULT_GPU_NAME,
        help="GPU name substring to require; use '' to allow any GPU",
    )
    p_search.add_argument(
        "--region",
        choices=["us", "asia", "all"],
        default="asia",
        help="Prefer Asia (default), US/CA, or Asia then US/CA",
    )

    p_create = sub.add_parser("create", help="Create an instance from an offer id")
    p_create.add_argument("offer_id", type=int, help="Offer id from search")
    p_create.add_argument("--image", default=None, help="Docker image to deploy")
    p_create.add_argument("--label", default=None, help="Instance label")
    p_create.add_argument("--disk", type=int, default=80, help="Disk size in GB")
    p_create.add_argument("--port", type=int, default=DEFAULT_PORT, help="Service port")
    p_create.add_argument("--env", action="append", help="Extra env vars as KEY=VALUE")

    p_wait = sub.add_parser("wait", help="Wait for an instance to expose its public URL")
    p_wait.add_argument("instance_id", type=int, help="Instance id")
    p_wait.add_argument("--port", type=int, default=DEFAULT_PORT, help="Service port")
    p_wait.add_argument("--timeout", type=int, default=1200, help="Max seconds to wait")
    p_wait.add_argument("--interval", type=int, default=20, help="Polling interval")

    p_health = sub.add_parser("health", help="Run health checks")
    p_health.add_argument("url", help="Instance base URL, e.g. http://1.2.3.4:11200")

    p_smoke = sub.add_parser("smoke", help="Run a minimal OmniVoice synthesis smoke test")
    p_smoke.add_argument("url", help="Instance base URL")
    p_smoke.add_argument("--text", default="Hello, this is OmniVoice running on Vast.ai.")
    p_smoke.add_argument("--language", default="en")

    p_destroy = sub.add_parser("destroy", help="Destroy a specific instance")
    p_destroy.add_argument("instance_id", type=int, help="Instance id")

    p_destroy_all = sub.add_parser(
        "destroy-all", help="Destroy all active instances with a label prefix"
    )
    p_destroy_all.add_argument("--prefix", default=DEFAULT_LABEL_PREFIX, help="Label prefix")
    p_destroy_all.add_argument("--yes", action="store_true", help="Skip confirmation")

    p_logs = sub.add_parser("logs", help="Fetch instance container logs")
    p_logs.add_argument("instance_id", type=int, help="Instance id")
    p_logs.add_argument("--tail", type=int, default=800, help="Lines to tail")

    args = parser.parse_args()
    globals()[f"cmd_{args.command.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
