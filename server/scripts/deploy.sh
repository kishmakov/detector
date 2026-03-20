#!/usr/bin/env bash
set -euo pipefail

REMOTE="GC"
REMOTE_DIR="/opt/detector-server"

echo "==> Syncing files to $REMOTE:$REMOTE_DIR"
rsync -az --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='tests/' \
  "$(dirname "$0")/../" \
  "$REMOTE:$REMOTE_DIR/"

echo "==> Building Docker image on remote"
ssh "$REMOTE" "cd $REMOTE_DIR && sudo docker build -t detector-server:latest ."

echo "==> Writing systemd unit"
ssh "$REMOTE" "sudo tee /etc/systemd/system/detector-server.service > /dev/null" <<'EOF'
[Unit]
Description=Detector Server
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker rm -f detector-server
ExecStart=/usr/bin/docker run --rm -p 8000:8000 \
  -e LOG_FILE=/app/logs/requests.log \
  -v /opt/detector-server/logs:/app/logs \
  --name detector-server detector-server:latest
ExecStop=/usr/bin/docker stop detector-server

[Install]
WantedBy=multi-user.target
EOF

echo "==> Reloading and restarting service"
ssh "$REMOTE" "sudo systemctl daemon-reload && sudo systemctl enable detector-server && sudo systemctl restart detector-server"

echo "==> Waiting for service to start..."
echo "==> Health check (retrying up to 30s)"
if ssh "$REMOTE" "for i in \$(seq 1 10); do curl -sf http://localhost:8000/health && exit 0; sleep 3; done; exit 1"; then
  echo ""
  echo "Deploy successful."
else
  echo "Health check failed. Check logs with:"
  echo "  ssh $REMOTE 'journalctl -u detector-server -n 50 --no-pager'"
  exit 1
fi
