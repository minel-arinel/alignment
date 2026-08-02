#!/usr/bin/env bash
# Download SuperElastix elastix binaries into tools/elastix/ (not committed; see .gitignore).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ELASTIX_INSTALL_DIR:-$ROOT/tools/elastix}"
VERSION="${ELASTIX_VERSION:-5.3.1}"

os="$(uname -s)"
arch="$(uname -m)"
case "${os}-${arch}" in
  Darwin-arm64|Darwin-aarch64)
    ZIP="elastix-${VERSION}-macos.zip"
    ;;
  Darwin-x86_64)
    echo "No official Intel macOS zip for ${VERSION}; use ELASTIX_EXECUTABLE or build from source." >&2
    exit 1
    ;;
  Linux-x86_64|Linux-amd64)
    ZIP="elastix-${VERSION}-ubuntu.zip"
    ;;
  *)
    echo "Unsupported platform: ${os} ${arch}" >&2
    exit 1
    ;;
esac

URL="https://github.com/SuperElastix/elastix/releases/download/${VERSION}/${ZIP}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading ${URL}"
curl -fsSL -o "${TMP}/${ZIP}" "${URL}"
rm -rf "${DEST}"
mkdir -p "$(dirname "$DEST")"
unzip -q "${TMP}/${ZIP}" -d "${TMP}/extract"
mv "${TMP}/extract" "${DEST}"

BIN="${DEST}/bin/elastix"
if [[ ! -x "${BIN}" ]]; then
  echo "Expected executable at ${BIN}" >&2
  exit 1
fi

"${BIN}" --version
echo "Installed elastix to ${DEST}"
echo "registration.sitkalignment will auto-detect: ${BIN}"
