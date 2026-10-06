#!/usr/bin/env bash
# One-time setup of a fresh Oracle Cloud Ubuntu VM. Run as the default "ubuntu" user.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

# Oracle's Ubuntu images reject everything but SSH in iptables, in addition to the VCN security list.
for rule in "tcp --dport 80" "tcp --dport 443" "udp --dport 443"; do
  # shellcheck disable=SC2086
  sudo iptables -C INPUT -m state --state NEW -p $rule -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 6 -m state --state NEW -p $rule -j ACCEPT
done
sudo netfilter-persistent save

echo "Done. Log out and back in so the docker group applies."
