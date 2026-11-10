#!/usr/bin/env bash
# One-time install of the PAPER traders on a fresh Ubuntu 22.04/24.04 server:
#   kalshi-paper  late-breakdown rule on the 15-minute markets (logs: paper-logs branch)
#   kalshi-range  hourly Bitcoin range rule (logs: paper-logs-range branch)
#   kalshi-fastpush  fast-push level breaks on the 15-minute markets (logs: paper-logs-fastpush branch)
#   kalshi-range5 5-account hourly range plan, separate from the others (logs: paper-logs-range5 branch)
#   kalshi-fastpush-range  fast-push level breaks on the hourly range bands (logs: paper-logs-fastpush-range branch)
#   kalshi-favorite  YES favorites 75-84c in the last 3 min on the hourly range bands (logs: paper-logs-favorite branch)
#   kalshi-trend-broad  trend-in YES 61-90c, 4-20 min before the hourly close (logs: paper-logs-trend-broad branch)
#   kalshi-trend-tight  trend-in YES 70-84c near the band centre, 4-20 min before the close (logs: paper-logs-trend-tight branch)
#   kalshi-maker        same signal as trend-tight, but a resting limit bid instead of taking the ask (logs: paper-logs-maker branch)
#   kalshi-quiet        quiet YES favorites 61-95c, 1-18 min before the hourly close (logs: paper-logs-quiet branch)
#   kalshi-squeeze      Bollinger band break while a squeeze is on, 15-minute markets, takes the ask (logs: paper-logs-squeeze branch)
#   kalshi-squeeze-maker  same signal, resting limit bid 1c under the ask (logs: paper-logs-squeeze-maker branch)
#   kalshi-model-gap    Direct ML model gap >= 3 pts, NO side 70-99c, 6-14 min left, cushion or breakout (logs: paper-logs-model-gap branch)
#   kalshi-dash   live chart of all of them at http://<server-ip>:8080/<secret>/  (kbot chart prints the link)
# No orders are ever sent. Run as root:  bash setup.sh
# It asks for: a GitHub token (read/write to this repo), the Kalshi API key ID, and the
# Kalshi private key (paste it). Re-running it is safe: it updates the code and restarts.
set -euo pipefail

main() {

REPO="emmabuilders-ui/kalshi-btc-bot"
BRANCH="${BRANCH:-main}"
LOG_BRANCH="paper-logs"
APP=/opt/kalshi-btc-bot
LOGS=/opt/kalshi-logs
RANGE_LOGS=/opt/kalshi-logs-range
RANGE_BRANCH="paper-logs-range"
FP_LOGS=/opt/kalshi-logs-fastpush
FP_BRANCH="paper-logs-fastpush"
R5_LOGS=/opt/kalshi-logs-range5
R5_BRANCH="paper-logs-range5"
FPR_LOGS=/opt/kalshi-logs-fastpush-range
FPR_BRANCH="paper-logs-fastpush-range"
FAV_LOGS=/opt/kalshi-logs-favorite
FAV_BRANCH="paper-logs-favorite"
TB_LOGS=/opt/kalshi-logs-trend-broad
TB_BRANCH="paper-logs-trend-broad"
TT_LOGS=/opt/kalshi-logs-trend-tight
TT_BRANCH="paper-logs-trend-tight"
MK_LOGS=/opt/kalshi-logs-maker
MK_BRANCH="paper-logs-maker"
QT_LOGS=/opt/kalshi-logs-quiet
QT_BRANCH="paper-logs-quiet"
SQ_LOGS=/opt/kalshi-logs-squeeze
SQ_BRANCH="paper-logs-squeeze"
SQM_LOGS=/opt/kalshi-logs-squeeze-maker
SQM_BRANCH="paper-logs-squeeze-maker"
MG_LOGS=/opt/kalshi-logs-model-gap
MG_BRANCH="paper-logs-model-gap"
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

# 13 Python processes on a 1 GB droplet run close to its memory; without swap the kernel kills one when it runs out,
# which shows up as an unexplained restart (and, for the chart, as "connection lost" in the browser).
if [ "$(swapon --show --noheadings | wc -l)" = 0 ] && [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap -q /swapfile && swapon /swapfile \
    && echo '/swapfile none swap sw 0 0' >> /etc/fstab && echo "Added a 1 GB swap file." || echo "Could not add a swap file; carrying on."
fi

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

echo "== Setting up the 5-account range bot log folder (pushed to the $R5_BRANCH branch every hour)"
if [ ! -d "$R5_LOGS/.git" ]; then
  git init -q "$R5_LOGS"
  git -C "$R5_LOGS" remote add origin "$URL"
  if git -C "$R5_LOGS" fetch -q origin "$R5_BRANCH" 2>/dev/null; then
    git -C "$R5_LOGS" checkout -q -B "$R5_BRANCH" "origin/$R5_BRANCH"
  else
    git -C "$R5_LOGS" checkout -q --orphan "$R5_BRANCH"
    echo "5-account hourly range paper-trading logs from the server. No real orders." > "$R5_LOGS/README.md"
    git -C "$R5_LOGS" add README.md
    git -C "$R5_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start 5-account range paper logs"
  fi
else
  git -C "$R5_LOGS" remote set-url origin "$URL"
fi
git -C "$R5_LOGS" config user.name "kalshi paper bot"
git -C "$R5_LOGS" config user.email "kbot@localhost"
git -C "$R5_LOGS" config push.default current
chown -R "$USER_" "$R5_LOGS"
sudo -u "$USER_" git -C "$R5_LOGS" push -q -u origin "$R5_BRANCH"

echo "== Setting up the fast-push range bot log folder (pushed to the $FPR_BRANCH branch every hour)"
if [ ! -d "$FPR_LOGS/.git" ]; then
  git init -q "$FPR_LOGS"
  git -C "$FPR_LOGS" remote add origin "$URL"
  if git -C "$FPR_LOGS" fetch -q origin "$FPR_BRANCH" 2>/dev/null; then
    git -C "$FPR_LOGS" checkout -q -B "$FPR_BRANCH" "origin/$FPR_BRANCH"
  else
    git -C "$FPR_LOGS" checkout -q --orphan "$FPR_BRANCH"
    echo "Fast-push hourly range band paper-trading logs from the server. No real orders." > "$FPR_LOGS/README.md"
    git -C "$FPR_LOGS" add README.md
    git -C "$FPR_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start fast-push range paper logs"
  fi
else
  git -C "$FPR_LOGS" remote set-url origin "$URL"
fi
git -C "$FPR_LOGS" config user.name "kalshi paper bot"
git -C "$FPR_LOGS" config user.email "kbot@localhost"
git -C "$FPR_LOGS" config push.default current
chown -R "$USER_" "$FPR_LOGS"
sudo -u "$USER_" git -C "$FPR_LOGS" push -q -u origin "$FPR_BRANCH"

echo "== Setting up the YES-favorite bot log folder (pushed to the $FAV_BRANCH branch every hour)"
if [ ! -d "$FAV_LOGS/.git" ]; then
  git init -q "$FAV_LOGS"
  git -C "$FAV_LOGS" remote add origin "$URL"
  if git -C "$FAV_LOGS" fetch -q origin "$FAV_BRANCH" 2>/dev/null; then
    git -C "$FAV_LOGS" checkout -q -B "$FAV_BRANCH" "origin/$FAV_BRANCH"
  else
    git -C "$FAV_LOGS" checkout -q --orphan "$FAV_BRANCH"
    echo "YES-favorite hourly range band paper-trading logs from the server. No real orders." > "$FAV_LOGS/README.md"
    git -C "$FAV_LOGS" add README.md
    git -C "$FAV_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start favorite paper logs"
  fi
else
  git -C "$FAV_LOGS" remote set-url origin "$URL"
fi
git -C "$FAV_LOGS" config user.name "kalshi paper bot"
git -C "$FAV_LOGS" config user.email "kbot@localhost"
git -C "$FAV_LOGS" config push.default current
chown -R "$USER_" "$FAV_LOGS"
sudo -u "$USER_" git -C "$FAV_LOGS" push -q -u origin "$FAV_BRANCH"
echo "== Setting up the trend-broad bot log folder (pushed to the $TB_BRANCH branch every hour)"
if [ ! -d "$TB_LOGS/.git" ]; then
  git init -q "$TB_LOGS"
  git -C "$TB_LOGS" remote add origin "$URL"
  if git -C "$TB_LOGS" fetch -q origin "$TB_BRANCH" 2>/dev/null; then
    git -C "$TB_LOGS" checkout -q -B "$TB_BRANCH" "origin/$TB_BRANCH"
  else
    git -C "$TB_LOGS" checkout -q --orphan "$TB_BRANCH"
    echo "Trend-in YES (broad rule) hourly range band paper-trading logs from the server. No real orders." > "$TB_LOGS/README.md"
    git -C "$TB_LOGS" add README.md
    git -C "$TB_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start trend-broad paper logs"
  fi
else
  git -C "$TB_LOGS" remote set-url origin "$URL"
fi
git -C "$TB_LOGS" config user.name "kalshi paper bot"
git -C "$TB_LOGS" config user.email "kbot@localhost"
git -C "$TB_LOGS" config push.default current
chown -R "$USER_" "$TB_LOGS"
sudo -u "$USER_" git -C "$TB_LOGS" push -q -u origin "$TB_BRANCH"
echo "== Setting up the trend-tight bot log folder (pushed to the $TT_BRANCH branch every hour)"
if [ ! -d "$TT_LOGS/.git" ]; then
  git init -q "$TT_LOGS"
  git -C "$TT_LOGS" remote add origin "$URL"
  if git -C "$TT_LOGS" fetch -q origin "$TT_BRANCH" 2>/dev/null; then
    git -C "$TT_LOGS" checkout -q -B "$TT_BRANCH" "origin/$TT_BRANCH"
  else
    git -C "$TT_LOGS" checkout -q --orphan "$TT_BRANCH"
    echo "Trend-in YES (tight rule) hourly range band paper-trading logs from the server. No real orders." > "$TT_LOGS/README.md"
    git -C "$TT_LOGS" add README.md
    git -C "$TT_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start trend-tight paper logs"
  fi
else
  git -C "$TT_LOGS" remote set-url origin "$URL"
fi
git -C "$TT_LOGS" config user.name "kalshi paper bot"
git -C "$TT_LOGS" config user.email "kbot@localhost"
git -C "$TT_LOGS" config push.default current
chown -R "$USER_" "$TT_LOGS"
sudo -u "$USER_" git -C "$TT_LOGS" push -q -u origin "$TT_BRANCH"
echo "== Setting up the maker bot log folder (pushed to the $MK_BRANCH branch every hour)"
if [ ! -d "$MK_LOGS/.git" ]; then
  git init -q "$MK_LOGS"
  git -C "$MK_LOGS" remote add origin "$URL"
  if git -C "$MK_LOGS" fetch -q origin "$MK_BRANCH" 2>/dev/null; then
    git -C "$MK_LOGS" checkout -q -B "$MK_BRANCH" "origin/$MK_BRANCH"
  else
    git -C "$MK_LOGS" checkout -q --orphan "$MK_BRANCH"
    echo "Limit-order (maker) trend-in YES hourly range band paper-trading logs from the server. No real orders." > "$MK_LOGS/README.md"
    git -C "$MK_LOGS" add README.md
    git -C "$MK_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start maker paper logs"
  fi
else
  git -C "$MK_LOGS" remote set-url origin "$URL"
fi
git -C "$MK_LOGS" config user.name "kalshi paper bot"
git -C "$MK_LOGS" config user.email "kbot@localhost"
git -C "$MK_LOGS" config push.default current
chown -R "$USER_" "$MK_LOGS"
sudo -u "$USER_" git -C "$MK_LOGS" push -q -u origin "$MK_BRANCH"
echo "== Setting up the quiet bot log folder (pushed to the $QT_BRANCH branch every hour)"
if [ ! -d "$QT_LOGS/.git" ]; then
  git init -q "$QT_LOGS"
  git -C "$QT_LOGS" remote add origin "$URL"
  if git -C "$QT_LOGS" fetch -q origin "$QT_BRANCH" 2>/dev/null; then
    git -C "$QT_LOGS" checkout -q -B "$QT_BRANCH" "origin/$QT_BRANCH"
  else
    git -C "$QT_LOGS" checkout -q --orphan "$QT_BRANCH"
    echo "Quiet-favorite YES hourly range band paper-trading logs from the server. No real orders." > "$QT_LOGS/README.md"
    git -C "$QT_LOGS" add README.md
    git -C "$QT_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start quiet paper logs"
  fi
else
  git -C "$QT_LOGS" remote set-url origin "$URL"
fi
git -C "$QT_LOGS" config user.name "kalshi paper bot"
git -C "$QT_LOGS" config user.email "kbot@localhost"
git -C "$QT_LOGS" config push.default current
chown -R "$USER_" "$QT_LOGS"
sudo -u "$USER_" git -C "$QT_LOGS" push -q -u origin "$QT_BRANCH"

echo "== Setting up the squeeze bot log folder (pushed to the $SQ_BRANCH branch every hour)"
if [ ! -d "$SQ_LOGS/.git" ]; then
  git init -q "$SQ_LOGS"
  git -C "$SQ_LOGS" remote add origin "$URL"
  if git -C "$SQ_LOGS" fetch -q origin "$SQ_BRANCH" 2>/dev/null; then
    git -C "$SQ_LOGS" checkout -q -B "$SQ_BRANCH" "origin/$SQ_BRANCH"
  else
    git -C "$SQ_LOGS" checkout -q --orphan "$SQ_BRANCH"
    echo "Squeeze-break (Bollinger band while a squeeze is on) 15-minute BTC market paper-trading logs from the server. No real orders." > "$SQ_LOGS/README.md"
    git -C "$SQ_LOGS" add README.md
    git -C "$SQ_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start squeeze paper logs"
  fi
else
  git -C "$SQ_LOGS" remote set-url origin "$URL"
fi
git -C "$SQ_LOGS" config user.name "kalshi paper bot"
git -C "$SQ_LOGS" config user.email "kbot@localhost"
git -C "$SQ_LOGS" config push.default current
chown -R "$USER_" "$SQ_LOGS"
sudo -u "$USER_" git -C "$SQ_LOGS" push -q -u origin "$SQ_BRANCH"
echo "== Setting up the squeeze maker bot log folder (pushed to the $SQM_BRANCH branch every hour)"
if [ ! -d "$SQM_LOGS/.git" ]; then
  git init -q "$SQM_LOGS"
  git -C "$SQM_LOGS" remote add origin "$URL"
  if git -C "$SQM_LOGS" fetch -q origin "$SQM_BRANCH" 2>/dev/null; then
    git -C "$SQM_LOGS" checkout -q -B "$SQM_BRANCH" "origin/$SQM_BRANCH"
  else
    git -C "$SQM_LOGS" checkout -q --orphan "$SQM_BRANCH"
    echo "Squeeze-break limit-order (maker) 15-minute BTC market paper-trading logs from the server. No real orders." > "$SQM_LOGS/README.md"
    git -C "$SQM_LOGS" add README.md
    git -C "$SQM_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start squeeze maker paper logs"
  fi
else
  git -C "$SQM_LOGS" remote set-url origin "$URL"
fi
git -C "$SQM_LOGS" config user.name "kalshi paper bot"
git -C "$SQM_LOGS" config user.email "kbot@localhost"
git -C "$SQM_LOGS" config push.default current
chown -R "$USER_" "$SQM_LOGS"
sudo -u "$USER_" git -C "$SQM_LOGS" push -q -u origin "$SQM_BRANCH"
echo "== Setting up the model-gap bot log folder (pushed to the $MG_BRANCH branch every hour)"
if [ ! -d "$MG_LOGS/.git" ]; then
  git init -q "$MG_LOGS"
  git -C "$MG_LOGS" remote add origin "$URL"
  if git -C "$MG_LOGS" fetch -q origin "$MG_BRANCH" 2>/dev/null; then
    git -C "$MG_LOGS" checkout -q -B "$MG_BRANCH" "origin/$MG_BRANCH"
  else
    git -C "$MG_LOGS" checkout -q --orphan "$MG_BRANCH"
    echo "Model-gap (Direct ML vs Kalshi price) 15-minute BTC market paper-trading logs from the server. No real orders." > "$MG_LOGS/README.md"
    git -C "$MG_LOGS" add README.md
    git -C "$MG_LOGS" -c user.name=kbot -c user.email=kbot@localhost commit -qm "start model-gap paper logs"
  fi
else
  git -C "$MG_LOGS" remote set-url origin "$URL"
fi
git -C "$MG_LOGS" config user.name "kalshi paper bot"
git -C "$MG_LOGS" config user.email "kbot@localhost"
git -C "$MG_LOGS" config push.default current
chown -R "$USER_" "$MG_LOGS"
sudo -u "$USER_" git -C "$MG_LOGS" push -q -u origin "$MG_BRANCH"

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
cat > /etc/systemd/system/kalshi-range5.service <<EOF
[Unit]
Description=Kalshi 5-account hourly range PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$R5_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-range5 --out $R5_LOGS/range5 --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kalshi-fastpush-range.service <<EOF
[Unit]
Description=Kalshi fast-push on hourly range bands PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$FPR_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-fastpush-range --out $FPR_LOGS/fastpush-range --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kalshi-favorite.service <<EOF
[Unit]
Description=Kalshi YES-favorite hourly range PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$FAV_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-favorite --out $FAV_LOGS/favorite --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kalshi-trend-broad.service <<EOF
[Unit]
Description=Kalshi trend-in YES (broad rule) hourly range PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$TB_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-trend --rule broad --out $TB_LOGS/trend-broad --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kalshi-trend-tight.service <<EOF
[Unit]
Description=Kalshi trend-in YES (tight rule) hourly range PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$TT_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-trend --rule tight --out $TT_LOGS/trend-tight --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/kalshi-maker.service <<EOF
[Unit]
Description=Kalshi limit-order (maker) trend-in YES hourly range PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$MK_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-maker --out $MK_LOGS/maker --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/kalshi-quiet.service <<EOF
[Unit]
Description=Kalshi quiet-favorite YES hourly range PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$QT_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-quiet --out $QT_LOGS/quiet --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kalshi-squeeze.service <<EOF
[Unit]
Description=Kalshi squeeze-break 15-minute market PAPER trader, takes the ask (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$SQ_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-squeeze --out $SQ_LOGS/squeeze --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/kalshi-squeeze-maker.service <<EOF
[Unit]
Description=Kalshi squeeze-break 15-minute market PAPER trader, limit bid (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$SQM_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-squeeze-maker --out $SQM_LOGS/squeeze-maker --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/kalshi-model-gap.service <<EOF
[Unit]
Description=Kalshi model-gap (Direct ML) 15-minute market PAPER trader (no real orders)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
WorkingDirectory=$MG_LOGS
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot paper-model-gap --out $MG_LOGS/model-gap --push-logs
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/kalshi-dash.service <<EOF
[Unit]
Description=Live chart of the Kalshi PAPER traders (read-only web page)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
EnvironmentFile=$CONF/env
ExecStart=$APP/.venv/bin/python -m kalshi_btc_bot dashboard --port 8080 --token-file $CONF/dash_token --breakdown $LOGS/breakdown --range $RANGE_LOGS/range --fastpush $FP_LOGS/fastpush --range5 $R5_LOGS/range5 --fastpush-range $FPR_LOGS/fastpush-range --favorite $FAV_LOGS/favorite --trend-broad $TB_LOGS/trend-broad --trend-tight $TT_LOGS/trend-tight --maker $MK_LOGS/maker --quiet $QT_LOGS/quiet --squeeze $SQ_LOGS/squeeze --squeeze-maker $SQM_LOGS/squeeze-maker --model-gap $MG_LOGS/model-gap
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q kalshi-paper kalshi-range kalshi-fastpush kalshi-range5 kalshi-fastpush-range kalshi-favorite kalshi-trend-broad kalshi-trend-tight kalshi-maker kalshi-quiet kalshi-squeeze kalshi-squeeze-maker kalshi-model-gap kalshi-dash
systemctl restart kalshi-paper kalshi-range kalshi-fastpush kalshi-range5 kalshi-fastpush-range kalshi-favorite kalshi-trend-broad kalshi-trend-tight kalshi-maker kalshi-quiet kalshi-squeeze kalshi-squeeze-maker kalshi-model-gap kalshi-dash

cat > /usr/local/bin/kbot <<'EOF'
#!/usr/bin/env bash
# kbot status | log | health | update | stop | start | chart | touch COIN TF BARS [TARGET]
case "${1:-status}" in
  status) for u in kalshi-paper kalshi-range kalshi-fastpush kalshi-range5 kalshi-fastpush-range kalshi-favorite kalshi-trend-broad kalshi-trend-tight kalshi-maker kalshi-quiet kalshi-squeeze kalshi-squeeze-maker kalshi-model-gap kalshi-dash; do systemctl --no-pager status $u | head -3; done
          echo "-- 15-min breakdown bot"; cat /opt/kalshi-logs/breakdown/status.json 2>/dev/null
          echo; echo "-- hourly range bot"; cat /opt/kalshi-logs-range/range/status.json 2>/dev/null
          echo; echo "-- fast-push bot"; cat /opt/kalshi-logs-fastpush/fastpush/status.json 2>/dev/null
          echo; echo "-- 5-account range bot"; cat /opt/kalshi-logs-range5/range5/status.json 2>/dev/null
          echo; echo "-- fast-push range bot"; cat /opt/kalshi-logs-fastpush-range/fastpush-range/status.json 2>/dev/null
          echo; echo "-- YES-favorite bot"; cat /opt/kalshi-logs-favorite/favorite/status.json 2>/dev/null
          echo; echo "-- trend broad bot"; cat /opt/kalshi-logs-trend-broad/trend-broad/status.json 2>/dev/null
          echo; echo "-- trend tight bot"; cat /opt/kalshi-logs-trend-tight/trend-tight/status.json 2>/dev/null
          echo; echo "-- maker (limit-order) bot"; cat /opt/kalshi-logs-maker/maker/status.json 2>/dev/null
          echo; echo "-- quiet favorite bot"; cat /opt/kalshi-logs-quiet/quiet/status.json 2>/dev/null
          echo; echo "-- squeeze bot"; cat /opt/kalshi-logs-squeeze/squeeze/status.json 2>/dev/null
          echo; echo "-- squeeze maker bot"; cat /opt/kalshi-logs-squeeze-maker/squeeze-maker/status.json 2>/dev/null
          echo; echo "-- model gap bot"; cat /opt/kalshi-logs-model-gap/model-gap/status.json 2>/dev/null ;;
  log)    for u in kalshi-paper kalshi-range kalshi-fastpush kalshi-range5 kalshi-fastpush-range kalshi-favorite kalshi-trend-broad kalshi-trend-tight kalshi-maker kalshi-quiet kalshi-squeeze kalshi-squeeze-maker kalshi-model-gap kalshi-dash; do journalctl -u $u -n 20 --no-pager; done ;;
  update) curl -fsSL -H "Authorization: token $(cat /etc/kalshi-bot/github_token)" \
            "https://raw.githubusercontent.com/emmabuilders-ui/kalshi-btc-bot/${BRANCH:-main}/deploy/setup.sh" | bash ;;
  stop)   systemctl stop kalshi-paper kalshi-range kalshi-fastpush kalshi-range5 kalshi-fastpush-range kalshi-favorite kalshi-trend-broad kalshi-trend-tight kalshi-maker kalshi-quiet kalshi-squeeze kalshi-squeeze-maker kalshi-model-gap kalshi-dash ;;
  start)  systemctl start kalshi-paper kalshi-range kalshi-fastpush kalshi-range5 kalshi-fastpush-range kalshi-favorite kalshi-trend-broad kalshi-trend-tight kalshi-maker kalshi-quiet kalshi-squeeze kalshi-squeeze-maker kalshi-model-gap kalshi-dash ;;
  health) echo "-- memory (MB)"; free -m; echo; echo "-- load (1, 5, 15 min) and uptime"; uptime
          echo; echo "-- restarts per service since boot"
          for u in kalshi-paper kalshi-range kalshi-fastpush kalshi-range5 kalshi-fastpush-range kalshi-favorite kalshi-trend-broad kalshi-trend-tight kalshi-maker kalshi-quiet kalshi-squeeze kalshi-squeeze-maker kalshi-model-gap kalshi-dash; do printf '%-24s %s\n' "$u" "$(systemctl show -p NRestarts --value $u)"; done
          echo; echo "-- processes the kernel killed for lack of memory (last 7 days)"
          journalctl -k --since "-7d" --no-pager 2>/dev/null | grep -i "out of memory\|oom-kill" | tail -10 || true
          echo; echo "-- chart service, last 15 log lines"; journalctl -u kalshi-dash -n 15 --no-pager ;;
  chart)  IP=$(curl -fsS -4 https://api.ipify.org 2>/dev/null || hostname -I | cut -d' ' -f1); TOK=$(cat /etc/kalshi-bot/dash_token)
          echo "Live chart: http://$IP:8080/$TOK/"
          echo "5-account range bot chart: http://$IP:8080/$TOK/range5/" ;;
  touch)  shift; /opt/kalshi-btc-bot/.venv/bin/python -m kalshi_btc_bot touch "$@" ;;
  *) echo "usage: kbot status|log|health|update|stop|start|chart|touch COIN TF BARS [TARGET]" ;;
esac
EOF
chmod +x /usr/local/bin/kbot

sleep 20
echo
journalctl -u kalshi-paper -n 8 --no-pager || true
journalctl -u kalshi-range -n 8 --no-pager || true
journalctl -u kalshi-fastpush -n 8 --no-pager || true
journalctl -u kalshi-range5 -n 8 --no-pager || true
journalctl -u kalshi-fastpush-range -n 8 --no-pager || true
journalctl -u kalshi-favorite -n 8 --no-pager || true
journalctl -u kalshi-trend-broad -n 8 --no-pager || true
journalctl -u kalshi-trend-tight -n 8 --no-pager || true
journalctl -u kalshi-maker -n 8 --no-pager || true
journalctl -u kalshi-quiet -n 8 --no-pager || true
journalctl -u kalshi-squeeze -n 8 --no-pager || true
journalctl -u kalshi-squeeze-maker -n 8 --no-pager || true
journalctl -u kalshi-model-gap -n 8 --no-pager || true
echo
echo "Done. All the paper traders and the live chart are running and restart by themselves after a reboot."
echo "Commands:  kbot status   kbot log   kbot health   kbot update   kbot chart   kbot touch BTC 1h 6 +1%"
kbot chart
}

main "$@"
