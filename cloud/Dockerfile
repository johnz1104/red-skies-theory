# Training image for the campaign's RL (CPU). Build from the repository root:
#   docker buildx build --platform linux/arm64 -f cloud/Dockerfile -t gtcw-rl:latest .
# Ampere A1 (OCI) is arm64 like Apple silicon; use linux/amd64 for x86 shapes.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY cloud/requirements.txt cloud/requirements.txt
# the CPU build of torch first: on Linux (arm64 and x86_64 alike) PyPI's torch pulls the CUDA
# libraries, whose cuDNN and cuSPARSELt wheels alone are 0.7-0.9 GB
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "$(grep '^torch==' cloud/requirements.txt)" \
 && pip install --no-cache-dir -r cloud/requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
# Runs and entry files are mounted at /app/runs
VOLUME ["/app/runs"]
ENTRYPOINT ["python", "-m", "cold_war_sim.campaign"]
CMD ["rl", "train", "--help"]
