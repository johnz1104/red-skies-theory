# Training on Oracle Cloud (OCI)

Packaging for the campaign's RL runs on a cloud VM. The code travels by `rsync`, so uncommitted changes and local entry files go along, or as a container image; a clone of the repository works too.

## Sizing first

Size a run from local measurements (development log E44, E45, E49 and E50) before choosing a shape.

- **Rollouts dominate and are CPU-bound.** One rollout worker uses one core (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), and the PPO update is small. So `--workers` ≈ OCPUs − 1.
- **Episode cost by event** (warm, one core, E44):
  - instant tree events and SALT: 0.01–0.04 s;
  - Vietnam and Afghanistan: about 0.05 s;
  - Korea: 0.3 s;
  - Berlin and the civil war: 0.7–2.8 s, dominated by reference play for concurrent wars.
- **Shape:**
  - Ampere A1 Flex (`VM.Standard.A1.Flex`, arm64) is priced per OCPU and GB, and matches Apple silicon, so the same wheels work.
  - x86 Flex shapes (`VM.Standard.E5.Flex`) also work: build the image for `linux/amd64`.
- **Freestyle episodes (E49):** a cold campaign costs about 100 s: the computer-run roles' reference play solves each event's tree, Cuba alone about 90 s since the officers. Solves are cached per worker by bound config (derived values are rounded), so once warm an episode costs about 5 s. In the smoke test the second iteration of two episodes took 10 s against 198 s for the first.
  - 100,000 warm episodes is about 140 CPU-hours, or 4–5 hours on a 32-OCPU A1 Flex at about $0.42 an hour (32 OCPUs, 64 GB): a few dollars at list prices (check Oracle's current price list).
  - Keep workers alive across iterations, as the trainer does, so the cache stays warm.
- **Cuba under QRE-PPO (E50):** train with `--chance-expected`. E48's estimate of about 6,000 episodes per iteration came from stale entries (the pre-officer Cuba).
  - **E50b:** with the flag, 512 episodes per iteration reached distance 0.040 from the QRE in 250 iterations: about 21 minutes of collection on 3 workers, plus about 90 s per worker for the reference solve.
  - **E50a:** sampled chance stalled at 0.33.
- **Free tier:** reported to have been cut in 2026 to about 2 OCPUs and 12 GB of A1. That is enough for smoke runs. Check Oracle's Always Free page before relying on it, and watch billing on Pay As You Go accounts.

## One-time setup

1. Create an OCI account and a compartment.
2. Create a VM instance:
   - image: Ubuntu 24.04 (its Python is 3.12);
   - shape: `VM.Standard.A1.Flex` with N OCPUs;
   - SSH public key: yours.
3. Note its public IP.

The OCI command-line tool (`oci setup config`) is not needed for the plain-VM route below.

## Entry files

Scenario runs start from entry files written by `rl entries`. An entry holds its event's game as bound when it was captured, and carries a digest of the catalog: training refuses entries captured under different data. So capture them with the code you train with, either locally (then `sync.sh up` copies `runs/entries/`) or on the VM:

```bash
.venv/bin/python -m cold_war_sim.campaign rl entries --output runs/entries/tier1.pkl   # historical replay, a few minutes
```

## Route A: plain VM (no Docker)

```bash
cloud/sync.sh up ubuntu@<ip>                 # code + runs/entries -> ~/gtcw on the VM
ssh ubuntu@<ip> 'cd gtcw && bash cloud/setup_vm.sh'
ssh ubuntu@<ip>
  cd gtcw && tmux new -s train
  .venv/bin/python -m cold_war_sim.campaign rl train \
      --entries runs/entries/tier1.pkl --events cuban_missile_crisis \
      --iterations 250 --episodes 512 --temperature-start 5 --anneal 0.5 --lr-final 3e-5 \
      --chance-expected --workers $(( $(nproc) - 1 )) --out runs/cuba
  # detach: Ctrl-b d; the run keeps going
cloud/sync.sh down ubuntu@<ip> runs/cuba     # bring the run back (log.jsonl, checkpoint.pt)
```

`--chance-expected` takes chance and computer-run stretches of instant tree events (crises, interventions, alignment contests) in expectation; it has no effect on wars and talks.

## Route B: container

```bash
docker buildx build --platform linux/arm64 -f cloud/Dockerfile -t gtcw-rl:latest .
docker save gtcw-rl:latest | ssh ubuntu@<ip> 'docker load'      # or push to OCI Registry
ssh ubuntu@<ip> 'docker run -d --name cuba -v ~/gtcw/runs:/app/runs gtcw-rl:latest \
    rl train --entries runs/entries/tier1.pkl --events cuban_missile_crisis --chance-expected \
    --workers 7 --out runs/cuba'
```

The image installs the CPU build of torch, so the same Dockerfile serves arm64 and x86 shapes.

## After a run

- `rl eval --checkpoint runs/cuba/checkpoint.pt` locally (it needs the run's entry file).
- Record the command, shape, wall time and cost with the results.
- Stop or terminate the instance. On standard shapes, stopping should end compute billing while the boot volume still bills; check OCI's billing documentation.
