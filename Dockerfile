# Agent + analysis image. The agent needs read-only access to host /sys (see deploy/kubernetes).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ethtool iproute2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nicprof
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --system --uid 10001 nicprof
USER nicprof
ENTRYPOINT ["nicprof"]
CMD ["collect", "--interval", "1"]
