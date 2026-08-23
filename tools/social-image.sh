#!/usr/bin/env bash
#
# Generate a 1080x1350 LinkedIn share image (header image + title + first
# paragraph) for a post, published or draft, without ever publishing a page
# in this format on the real site. See _layouts/social-card.html and
# _config.social.yml for how the isolation works.
#
# Usage:
#   bash tools/social-image.sh <post-path-or-slug> [-o OUTPUT] [-p PORT]
#
# Examples:
#   bash tools/social-image.sh _drafts/how-boardbarian-fails.md
#   bash tools/social-image.sh how-boardbarian-fails -o /tmp/card.png

set -eu

PORT=4100
OUTPUT=""
POST=""
MODE="dark"

help() {
  echo "Generate a 1080x1350 LinkedIn share image for a post."
  echo
  echo "Usage:"
  echo
  echo "   bash $0 <post-path-or-slug> [options]"
  echo
  echo "Options:"
  echo "     -o, --output <path>   Output PNG path (default: social-cards/<slug>.png)"
  echo "     -m, --mode <mode>     Color scheme: dark or light (default: dark)"
  echo "     -p, --port <port>     Local port to serve the build on (default: 4100)"
  echo "     -h, --help            Print this information."
}

while (($#)); do
  opt="$1"
  case $opt in
  -o | --output)
    OUTPUT="$2"
    shift 2
    ;;
  -m | --mode)
    MODE="$2"
    shift 2
    ;;
  -p | --port)
    PORT="$2"
    shift 2
    ;;
  -h | --help)
    help
    exit 0
    ;;
  *)
    if [[ -n "$POST" ]]; then
      echo "Unexpected argument: $opt" >&2
      help
      exit 1
    fi
    POST="$opt"
    shift
    ;;
  esac
done

if [[ -z "$POST" ]]; then
  echo "Error: must specify a post path or slug" >&2
  help
  exit 1
fi

if [[ "$MODE" != "light" && "$MODE" != "dark" ]]; then
  echo "Error: --mode must be 'light' or 'dark' (got '$MODE')" >&2
  exit 1
fi

# Derive the post slug from a filename like _drafts/2026-08-15-my-post.md,
# or accept a bare slug like my-post directly.
SLUG="$(basename "$POST")"
SLUG="${SLUG%.*}"
SLUG="$(echo "$SLUG" | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}-//')"

OUTPUT="${OUTPUT:-social-cards/${SLUG}.png}"
mkdir -p "$(dirname "$OUTPUT")"

STUB=_social_cards/generated.html
SITE_DIR=_site_social

mkdir -p _social_cards

SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$SITE_DIR" "$STUB"
}
trap cleanup EXIT

cat >"$STUB" <<EOF
---
layout: social-card
target_slug: ${SLUG}
card_mode: ${MODE}
permalink: /__social-card__/
---
EOF

echo "> Building social card page for slug '${SLUG}' (${MODE} mode)..."
bundle exec jekyll build --config _config.yml,_config.social.yml --drafts -d "$SITE_DIR"

# The built site uses root-absolute asset paths (/assets/css/...), so it has to be
# served over HTTP rather than opened as a file:// URL.
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  exec 3<&- 3>&-
  echo "Error: port ${PORT} is already in use; pass -p to pick another." >&2
  exit 1
fi

python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$SITE_DIR" >/dev/null 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 50); do
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
    exec 3<&- 3>&-
    break
  fi
  sleep 0.1
done

echo "> Screenshotting at 1080px wide..."
.venv-playwright/bin/python3 tools/social-image/screenshot.py \
  --url "http://127.0.0.1:${PORT}/__social-card__/" \
  --out "$OUTPUT"

echo "> Wrote ${OUTPUT}"
