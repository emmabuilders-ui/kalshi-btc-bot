#!/usr/bin/env bash
# One-time install of the PAPER traders on a fresh Ubuntu 22.04/24.04 server:
#   kalshi-paper  late-breakdown rule on the 15-minute markets (logs: paper-logs branch)
#   kalshi-range  hourly Bitcoin range rule (logs: paper-logs-range branch)
#   kalshi-fastpush  fast-push level breaks on the 15-minute markets (logs: paper-logs-fastpush branch)
#   kalshi-dash   live chart of all three at http://<server-ip>:8080/<secret>/  (kbot chart prints the link)
# No orders are ever sent. Run as root:  bash setup.sh
# It asks for: a GitHub token (read/write to this repo), the Kalshi API key ID, and the
# Kalshi private key (paste it). Re-running it is safe: it updates the code and restarts.
set -euo pipefail

REPO="emmabuilders-ui/kalshi-btc-bot"
BRANCH="${BRANCH:-main}"
LOG_BRANCH="paper-logs"
APP=/opt/kalshi-btc-bot
LOGS=/opt/kalshi-logs
RANGE_LOGS=/opt/kalshi-logs-range
RANGE_BRANCH="paper-logs-range"
FP_LOGS=/opt/kalshi-logs-fastpush
FP_BRANCH="paper-logs-fastpush"
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
[ -s "$CONF/dash_token" ] || head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$CONF/dash_token"
GH=$(cat "$CONF/github_token")

echo "== Installing system packages"
export DEBIAN_FRONTEND=noninteractive
[ -n "${SKIP_APT:-}" ] || apt-get update -qq
[ -n "${SKIP_APT:-}" ] || apt-get install -y -qq git python3 python3-venv python3-pip >/dev/null

id "$USER_" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$USER_"
chown -R "$USER_" "$CONF"; chmod 600 "$CONF"/*
git config --global --add safe.directory "*"
URL="${REPO_URL:-https://x-access-token:${GH}@github.com/${REPO}.git}"

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

echo "== Setting up the range bot log folder (pushed to the $RANGE_BRANCH branch every hour)"
if [ ! -d "$RANGE_LOGS/.git" ]; then
  git init -q "$RANGE_LOGS"
  git -C "$RANGE_LOGS" remote add origin "$URL"
  if git -C "$RANGE_LOGS" fetch -q origin "$RANGE_BRANCH" 2>/dev/null; then
    git -C "$RANGE_LOGS" checkout -q -B "$RANGE_BRANCH" "origin/$RANGE_BRANCH"
  else
    git -C "$RANGE_LOGS" checkout -q --orphan "$RANGE_BRANCH"
    echo "Hourly range paper-trading logs from the server. No real orders." > "$RANGE_LOGS/README.md"
    git -C "$RANGE_LOGS" add README.md
    git -C "$RANGE_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start range paper logs"
  fi
else
  git -C "$RANGE_LOGS" remote set-url origin "$URL"
fi
git -C "$RANGE_LOGS" config user.name "kalshi paper bot"
git -C "$RANGE_LOGS" config user.email "kbot@localhost"
git -C "$RANGE_LOGS" config push.default current
chown -R "$USER_" "$RANGE_LOGS"
sudo -u "$USER_" git -C "$RANGE_LOGS" push -q -u origin "$RANGE_BRANCH"

echo "== Setting up the fast-push bot log folder (pushed to the $FP_BRANCH branch every hour)"
if [ ! -d "$FP_LOGS/.git" ]; then
  git init -q "$FP_LOGS"
  git -C "$FP_LOGS" remote add origin "$URL"
  if git -C "$FP_LOGS" fetch -q origin "$FP_BRANCH" 2>/dev/null; then
    git -C "$FP_LOGS" checkout -q -B "$FP_BRANCH" "origin/$FP_BRANCH"
  else
    git -C "$FP_LOGS" checkout -q --orphan "$FP_BRANCH"
    echo "Fast-push 15-minute paper-trading logs from the server. No real orders." > "$FP_LOGS/README.md"
    git -C "$FP_LOGS" add README.md
    git -C "$FP_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start fast-push paper logs"
  fi
else
  git -C "$FP_LOGS" remote set-url origin "$URL"
fi
git -C "$FP_LOGS" config user.name "kalshi paper bot"
git -C "$FP_LOGS" config user.email "kbot@localhost"
git -C "$FP_LOGS" config push.default current
chown -R "$USER_" "$FP_LOGS"
sudo -u "$USER_" git -C "$FP_LOGS" push -q -u origin "$FP_BRANCH"

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
cat > /etc/systemd/system/kalshi-range.service <<EOF
[Unit]
Description=Kalshi hourly range PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$RANGE_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-range --out $RANGE_LOGS/range --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kalshi-fastpush.service <<EOF
[Unit]
Description=Kalshi fast-push level-break PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$FP_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-fastpush --out $FP_LOGS/fastpush --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q kalshi-paper kalshi-range kalshi-fastpush
systemctl restart kalshi-paper kalshi-range kalshi-fastpush

cat > /usr/local/bin/kbot <<'EOF'
#!/usr/bin/env bash
# kbot status | log | update | stop | start | chart | touch COIN TF BARS [TARGET]
case "${1:-status}" in
  status) for u in kalshi-paper kalshi-range kalshi-fastpush; do systemctl --no-pager status $u | head -3; done
          echo "-- 15-min breakdown bot"; cat /opt/kalshi-logs/breakdown/status.json 2>/dev/null
          echo; echo "-- hourly range bot"; cat /opt/kalshi-logs-range/range/status.json 2>/dev/null
          echo; echo "-- fast-push bot"; cat /opt/kalshi-logs-fastpush/fastpush/status.json 2>/dev/null ;;
  log)    for u in kalshi-paper kalshi-range kalshi-fastpush; do journalctl -u $u -n 20 --no-pager; done ;;
  update) curl -fsSL -H "Authorization: token $(cat /etc/kalshi-bot/github_token)" \
            "https://raw.githubusercontent.com/emmabuilders-ui/kalshi-btc-bot/${BRANCH:-main}/deploy/setup.sh" | bash ;;
  stop)   systemctl stop kalshi-paper kalshi-range kalshi-fastpush ;;
  start)  systemctl start kalshi-paper kalshi-range kalshi-fastpush ;;
  chart)  echo "Live chart: http://$(curl -fsS -4 https://api.ipify.org 2>/dev/null || hostname -I | cut -d' ' -f1):8080/$(cat /etc/kalshi-bot/dash_token)/" ;;
  touch)  shift; /opt/kalshi-btc-bot/.venv/bin/python -m kalshi_btc_bot touch "$@" ;;
  *) echo "usage: kbot status|log|update|stop|start|chart|touch COIN TF BARS [TARGET]" ;;
esac
EOF
chmod +x /usr/local/bin/kbot

sleep 20
echo
journalctl -u kalshi-paper -n 8 --no-pager || true
journalctl -u kalshi-range -n 8 --no-pager || true
journalctl -u kalshi-fastpush -n 8 --no-pager || true
echo
echo "Done. All three paper traders are running and restart by themselves after a reboot."
echo "Commands:  kbot status   kbot log   kbot update   kbot chart   kbot touch BTC 1h 6 +1%"
kbot chart
