#!/bin/sh
# Generates a runtime config.js consumed by the SPA before its main bundle
# loads, so a single built/published image can point at a different backend
# per deployment (VITE_API_BASE_URL as a runtime env var) without rebuilding.
# Defaults to /api which nginx proxies internally (works local & remote).
set -eu

cat > /usr/share/nginx/html/config.js <<EOF
window.__API_BASE_URL__ = "${VITE_API_BASE_URL:-/api}";
EOF

exec "$@"
