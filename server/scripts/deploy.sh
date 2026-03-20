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
ssh "$REMOTE" "cd $REMOTE_DIR && docker build -t detector-server:latest ."

echo "==> Writing systemd unit"
ssh "$REMOTE" "cat > /etc/systemd/system/detector-server.service" <<'EOF'
[Unit]
Description=Detector Server
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker rm -f detector-server
ExecStart=/usr/bin/docker run --rm -p 8000:8000 --name detector-server detector-server:latest
ExecStop=/usr/bin/docker stop detector-server

[Install]
WantedBy=multi-user.target
EOF

echo "==> Reloading and restarting service"
ssh "$REMOTE" "systemctl daemon-reload && systemctl enable detector-server && systemctl restart detector-server"

echo "==> Waiting for service to start..."
sleep 3

echo "==> Health check"
if ssh "$REMOTE" "curl -sf http://localhost:8000/health"; then
  echo ""
  echo "Deploy successful."
else
  echo "Health check failed. Check logs with:"
  echo "  ssh $REMOTE 'journalctl -u detector-server -n 50 --no-pager'"
  exit 1
fi
