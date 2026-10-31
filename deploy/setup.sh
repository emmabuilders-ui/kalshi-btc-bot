#!/usr/bin/env bash
# One-time install of the late-breakdown PAPER trader on a fresh Ubuntu 22.04/24.04 server.
# No orders are ever sent. Run as root:  bash setup.sh
# It asks for: a GitHub token (read/write to this repo), the Kalshi API key ID, and the
# Kalshi private key (paste it). Re-running it is safe: it updates the code and restarts.
set -euo pipefail

REPO="emmabuilders-ui/kalshi-btc-bot"
BRANCH="${BRANCH:-main}"
LOG_BRANCH="paper-logs"
APP=/opt/kalshi-btc-bot
LOGS=/opt/kalshi-logs
CONF=/etc/kalshi-bot
USER_=kbot

[ "$(id -u)" -eq 0 ] || { echo "Run as root (sudo bash setup.sh)"; exit 1; }
mkdir -p "$CONF"; chmod 700 "$CONF"

if [ ! -s "$CONF/github_token" ]; then
  read -rp "GitHub token (starts with github_pat_): " GH
  echo "$GH" > "$CONF/github_token"
fi
if [ ! -s "$CONF/env" ]; then
  read -rp "Kalshi API key ID: " KID
  printf 'KALSHI_KEY_ID=%s\nKALSHI_KEY_FILE=%s/key.pem\n' "$KID" "$CONF" > "$CONF/env"
fi
if [ ! -s "$CONF/key.pem" ]; then
  echo "Paste the Kalshi private key (all lines, including BEGIN and END), then press Enter:"
  : > "$CONF/key.pem"
  while IFS= read -r line; do
    echo "$line" >> "$CONF/key.pem"
    [[ "$line" == *"-----END"* ]] && break
  done
fi
GH=$(cat "$CONF/github_token")

echo "== Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3 python3-venv python3-pip >/dev/null

id "$USER_" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$USER_"
chown -R "$USER_" "$CONF"; chmod 600 "$CONF"/*
git config --global --add safe.directory "*"
URL="https://x-access-token:${GH}@github.com/${REPO}.git"

echo "== Getting the code ($BRANCH)"
if [ -d "$APP/.git" ]; then
  git -C "$APP" remote set-url origin "$URL"
  git -C "$APP" fetch -q origin "$BRANCH" && git -C "$APP" checkout -q -B "$BRANCH" "origin/$BRANCH"
else
  git clone -q --branch "$BRANCH" "$URL" "$APP"
fi
python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q "$APP"

echo "== Setting up the log folder (pushed to the $LOG_BRANCH branch every hour)"
if [ ! -d "$LOGS/.git" ]; then
  git init -q "$LOGS"
  git -C "$LOGS" remote add origin "$URL"
  if git -C "$LOGS" fetch -q origin "$LOG_BRANCH" 2>/dev/null; then
    git -C "$LOGS" checkout -q -B "$LOG_BRANCH" "origin/$LOG_BRANCH"
  else
    git -C "$LOGS" checkout -q --orphan "$LOG_BRANCH"
    echo "Paper-trading logs from the server. No real orders." > "$LOGS/README.md"
    git -C "$LOGS" add README.md
    git -C "$LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start paper logs"
  fi
else
  git -C "$LOGS" remote set-url origin "$URL"
fi
git -C "$LOGS" config user.name "kalshi paper bot"
git -C "$LOGS" config user.email "kbot@localhost"
git -C "$LOGS" config push.default current
chown -R "$USER_" "$LOGS"
sudo -u "$USER_" git -C "$LOGS" push -q -u origin "$LOG_BRANCH"

echo "== Installing the service"
cat > /etc/systemd/system/kalshi-paper.service <<EOF
[Unit]
Description=Kalshi late-breakdown PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-breakdown --out $LOGS/breakdown --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q kalshi-paper
systemctl restart kalshi-paper

cat > /usr/local/bin/kbot <<'EOF'
#!/usr/bin/env bash
# kbot status | log | update | stop | start
case "${1:-status}" in
  status) systemctl --no-pager status kalshi-paper | head -5; cat /opt/kalshi-logs/breakdown/status.json 2>/dev/null ;;
  log)    journalctl -u kalshi-paper -n 50 --no-pager ;;
  update) curl -fsSL -H "Authorization: token $(cat /etc/kalshi-bot/github_token)" \
            "https://raw.githubusercontent.com/emmabuilders-ui/kalshi-btc-bot/${BRANCH:-main}/deploy/setup.sh" | bash ;;
  stop)   systemctl stop kalshi-paper ;;
  start)  systemctl start kalshi-paper ;;
  *) echo "usage: kbot status|log|update|stop|start" ;;
esac
EOF
chmod +x /usr/local/bin/kbot

sleep 20
echo
journalctl -u kalshi-paper -n 8 --no-pager || true
echo
echo "Done. The paper trader is running and restarts by itself after a reboot."
echo "Commands:  kbot status   kbot log   kbot update"
