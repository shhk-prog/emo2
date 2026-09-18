import sys
from pathlib import Path
import pytest

from affective_empathy_eval.run import PRODUCTION_STAGE_ORDER


PRIMARY_ENTRYPOINTS = [
    # Behavioral
    "behavioral/primary/run_behavioral_emobank.py",
    "behavioral/primary/run_behavioral_aipsy.py",
    # V1 Stage
    "v1/primary/prepare_v1_phase_b_controls.py",
    "v1/primary/run_phase_a.py",
    "v1/primary/run_phase_b.py",
    "v1/primary/run_phase_c.py",
    "v1/primary/phase_c/run_e6_specialization.py",
    "v1/primary/phase_c/summarize_phase_c.py",
    # V2 Stage
    "v2/primary/run_rq1_rq2_cross_decoding.py",
    "v2/primary/run_rq3_causal_map.py",
    "v2/primary/run_rq4_recovery_patching.py",
    # V3 Stage
    "v3/primary/run_rq1_state_induction.py",
    "v3/primary/run_rq2_spatiotemporal_maps.py",
    "v3/primary/run_rq3_path_mediation.py",
    "v3/primary/run_confirmatory_replication.py",
    # Production Bash Runners
    "scripts/run_production_all.sh",
    "scripts/run_production_behavioral.sh",
    "scripts/run_production_v1.sh",
    "scripts/run_production_v2.sh",
    "scripts/run_production_v3.sh",
    "scripts/run_scale_validation.py",
]


def test_all_primary_entrypoints_exist():
    """Verify that all production scripts invoked by affective_empathy_eval.run exist as files."""
    project_root = Path(__file__).resolve().parent.parent
    missing = []
    for rel_path in PRIMARY_ENTRYPOINTS:
        target_file = project_root / rel_path
        if not target_file.is_file():
            missing.append(rel_path)

    assert not missing, f"Missing primary production entrypoint files: {missing}"


def test_production_dry_run_dispatch(monkeypatch):
    """
    Dry-run integration test:
    Verify that affective_empathy_eval.run can dispatch all production stages
    (behavioral, v1, v2, v3) without missing entrypoints or command construction failures.
    """
    import argparse
    from unittest.mock import MagicMock
    import affective_empathy_eval.run as runner

    executed_cmds = []

    def mock_run_command(cmd):
        # Validate that the script target actually exists on disk
        script_arg = cmd[1] if len(cmd) > 1 else ""
        if script_arg.endswith(".py"):
            assert Path(script_arg).is_file(), f"Dispatched script does not exist: {script_arg}"
        executed_cmds.append(cmd)

    monkeypatch.setattr(runner, "run_command", mock_run_command)
    monkeypatch.setattr(runner, "v3_gate_allows_continuation", lambda **kwargs: True)

    # Test each stage dispatch
    for stage in PRODUCTION_STAGE_ORDER:
        args = argparse.Namespace(
            stage=stage,
            model_set="primary_small",
            family="qwen",
            base_model=None,
            instruct_model=None,
            models_config="configs/models.yaml",
            device="cpu",
            dry_run=True,
            force_after_no_go=True,
            max_samples=2,
            all_layers=False,
        )
        if stage == "behavioral":
            runner.run_behavioral(args, sys.executable)
        elif stage == "v1":
            runner.run_v1(args, sys.executable)
        elif stage == "v2":
            runner.run_v2(args, sys.executable)
        elif stage == "v3":
            runner.run_v3(args, sys.executable)

    # Ensure commands were actually constructed and checked
    assert len(executed_cmds) > 0, "No commands were dispatched during dry-run integration test"


def test_all_dispatched_commands_argparse_compatibility(monkeypatch):
    """
    Verify that every command line constructed by affective_empathy_eval.run with
    --dry-run and --max-samples 2 has its options recognized in the target script's parser.
    Catches errors where a dispatched option (e.g. --limit, --is-instruct) is missing in a sub-script.
    """
    import argparse
    import subprocess
    import affective_empathy_eval.run as runner

    executed_cmds = []

    def capture_cmd(cmd):
        executed_cmds.append(cmd)

    monkeypatch.setattr(runner, "run_command", capture_cmd)
    monkeypatch.setattr(runner, "v3_gate_allows_continuation", lambda **kwargs: True)

    for stage in PRODUCTION_STAGE_ORDER:
        args = argparse.Namespace(
            stage=stage,
            model_set="primary_small",
            family="qwen",
            base_model=None,
            instruct_model=None,
            models_config="configs/models.yaml",
            device="cpu",
            dry_run=True,
            force_after_no_go=True,
            max_samples=2,
            all_layers=False,
        )
        if stage == "behavioral":
            runner.run_behavioral(args, sys.executable)
        elif stage == "v1":
            runner.run_v1(args, sys.executable)
        elif stage == "v2":
            runner.run_v2(args, sys.executable)
        elif stage == "v3":
            runner.run_v3(args, sys.executable)

    import os
    repo_root = Path(__file__).resolve().parent.parent
    src_path = str(repo_root / "src")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path

    # Check each dispatched script's parser against all passed flags
    for cmd in executed_cmds:
        if len(cmd) < 2 or not cmd[1].endswith(".py"):
            continue
        script_path = cmd[1]
        res = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            cwd=str(repo_root),
        )
        assert res.returncode == 0, f"Failed to run --help on {script_path}: {res.stderr}"
        help_text = res.stdout

        flags = [arg for arg in cmd[2:] if isinstance(arg, str) and arg.startswith("--")]
        for flag in flags:
            assert flag in help_text, (
                f"Dispatched flag '{flag}' is not recognized in {script_path}'s argument parser!\n"
                f"Full command: {' '.join(cmd)}"
            )
