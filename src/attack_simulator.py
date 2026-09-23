"""
================================================================================
                           CYBERGUARD RED TEAM
                Interactive Hacker Attack Simulator & Injector
================================================================================

Simulates offensive cyber attack vectors against the local environment:
1. TCP SYN Flood (State Table Exhaustion)
2. Stealth Port Reconnaissance Scan (Nmap-style)
3. Layer-7 DDoS HTTP Flood (Web Application Starvation)
4. SSH Brute Force Credential Stuffing
5. Polymorphic Zero-Day Exploit (Manifold Evasion)

Author: CyberGuard AI Engineering Team
Version: 1.0.0
================================================================================
"""

import sys
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# Force UTF-8 on Windows consoles to prevent charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
LIVE_ATTACK_FILE = DATA_DIR / "live_attack_stream.csv"


class HackerAttackSimulator:
    """Crafts offensive network telemetry mimicking real-world hacker tools."""

    @staticmethod
    def craft_syn_flood(n_packets: int = 150) -> pd.DataFrame:
        """Mimics 'hping3 -S --flood': Spoofed SYN packets with no ACK completion."""
        rows = []
        for i in range(n_packets):
            rows.append({
                "flow_duration": round(random.uniform(0.01, 1.2), 4),
                "protocol_type": "TCP",
                "service": "http",
                "flag": "S0",  # SYN sent, no ACK
                "src_bytes": random.randint(40, 70),  # Header only
                "dst_bytes": 0,  # Victim never responds
                "packet_rate": round(random.uniform(600.0, 1500.0), 2),
                "byte_rate": round(random.uniform(30000.0, 80000.0), 2),
                "failed_logins": 0,
                "logged_in": 0,
                "count": random.randint(350, 500),
                "srv_count": random.randint(200, 450),
                "serror_rate": round(random.uniform(0.92, 1.00), 4),
                "same_srv_rate": round(random.uniform(0.85, 0.99), 4),
                "diff_srv_rate": round(random.uniform(0.01, 0.08), 4),
                "dst_host_count": 255,
                "dst_host_srv_count": random.randint(180, 255),
                "dst_host_same_src_port_rate": round(random.uniform(0.80, 0.99), 4),
                "label": "DoS_SYN_Flood",
                "is_attack": 1,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def craft_port_scan(n_probes: int = 100) -> pd.DataFrame:
        """Mimics 'nmap -sS -p 1-1024': Rapid sequential port reconnaissance."""
        rows = []
        services = ["http", "ssh", "ssl", "ftp", "smtp", "dns", "other"]
        for i in range(n_probes):
            rows.append({
                "flow_duration": round(random.uniform(0.005, 0.08), 4),
                "protocol_type": "TCP",
                "service": random.choice(services),
                "flag": random.choice(["RSTO", "REJ", "SH"]),
                "src_bytes": random.randint(44, 80),
                "dst_bytes": random.randint(0, 40),
                "packet_rate": round(random.uniform(180.0, 450.0), 2),
                "byte_rate": round(random.uniform(9000.0, 25000.0), 2),
                "failed_logins": 0,
                "logged_in": 0,
                "count": random.randint(120, 250),
                "srv_count": random.randint(2, 15),
                "serror_rate": round(random.uniform(0.70, 0.98), 4),
                "same_srv_rate": round(random.uniform(0.02, 0.15), 4),
                "diff_srv_rate": round(random.uniform(0.75, 0.95), 4),
                "dst_host_count": 255,
                "dst_host_srv_count": random.randint(2, 20),
                "dst_host_same_src_port_rate": round(random.uniform(0.01, 0.10), 4),
                "label": "Port_Scan",
                "is_attack": 1,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def craft_http_flood(n_requests: int = 120) -> pd.DataFrame:
        """Mimics Layer-7 Application Flood (LOIC / GoldenEye): High concurrency web traffic."""
        rows = []
        for i in range(n_requests):
            rows.append({
                "flow_duration": round(random.uniform(0.5, 3.5), 4),
                "protocol_type": "TCP",
                "service": "http",
                "flag": "SF",  # Handshake completes normally to bypass L4 firewalls
                "src_bytes": random.randint(1100, 2200),
                "dst_bytes": random.randint(200, 600),
                "packet_rate": round(random.uniform(350.0, 800.0), 2),
                "byte_rate": round(random.uniform(60000.0, 140000.0), 2),
                "failed_logins": 0,
                "logged_in": 1,
                "count": random.randint(250, 480),
                "srv_count": random.randint(240, 450),
                "serror_rate": round(random.uniform(0.0, 0.05), 4),
                "same_srv_rate": round(random.uniform(0.92, 1.00), 4),
                "diff_srv_rate": round(random.uniform(0.00, 0.04), 4),
                "dst_host_count": 255,
                "dst_host_srv_count": 255,
                "dst_host_same_src_port_rate": round(random.uniform(0.60, 0.90), 4),
                "label": "DDoS_HTTP_Flood",
                "is_attack": 1,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def craft_brute_force(n_attempts: int = 80) -> pd.DataFrame:
        """Mimics SSH/FTP Hydra dictionary password cracking."""
        rows = []
        for i in range(n_attempts):
            rows.append({
                "flow_duration": round(random.uniform(2.0, 8.0), 4),
                "protocol_type": "TCP",
                "service": "ssh",
                "flag": "SF",
                "src_bytes": random.randint(450, 950),
                "dst_bytes": random.randint(350, 850),
                "packet_rate": round(random.uniform(25.0, 75.0), 2),
                "byte_rate": round(random.uniform(2500.0, 8500.0), 2),
                "failed_logins": random.randint(3, 9),  # Multiple failed logins
                "logged_in": 0,
                "count": random.randint(40, 95),
                "srv_count": random.randint(35, 90),
                "serror_rate": 0.0,
                "same_srv_rate": 1.0,
                "diff_srv_rate": 0.0,
                "dst_host_count": random.randint(8, 25),
                "dst_host_srv_count": random.randint(8, 25),
                "dst_host_same_src_port_rate": round(random.uniform(0.15, 0.40), 4),
                "label": "Brute_Force_SSH",
                "is_attack": 1,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def craft_zero_day(n_samples: int = 60) -> pd.DataFrame:
        """
        Crafts novel polymorphic anomaly:
        Features are intentionally distorted so Path A (XGBoost) does NOT recognize it,
        forcing Path B (Deep Autoencoder) to detect the anomaly via reconstruction loss.
        """
        rows = []
        for i in range(n_samples):
            rows.append({
                "flow_duration": round(random.uniform(25.0, 90.0), 4),  # Abnormally long session
                "protocol_type": "UDP",
                "service": "dns",
                "flag": "SF",
                "src_bytes": random.randint(4500, 12000),  # Giant DNS query (Data Exfiltration)
                "dst_bytes": random.randint(30, 80),
                "packet_rate": round(random.uniform(150.0, 320.0), 2),
                "byte_rate": round(random.uniform(85000.0, 190000.0), 2),
                "failed_logins": 0,
                "logged_in": 0,
                "count": random.randint(180, 310),
                "srv_count": random.randint(170, 300),
                "serror_rate": 0.0,
                "same_srv_rate": 0.98,
                "diff_srv_rate": 0.02,
                "dst_host_count": 255,
                "dst_host_srv_count": random.randint(15, 45),
                "dst_host_same_src_port_rate": round(random.uniform(0.70, 0.95), 4),
                "label": "ZeroDay_DNS_Exfil",
                "is_attack": 1,
            })
        return pd.DataFrame(rows)


def inject_attack(attack_type: str, n_samples: int = 100, mix_benign: bool = True) -> Path:
    """Injects crafted attack flows into live telemetry and exports to CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    attack_map = {
        "syn_flood": HackerAttackSimulator.craft_syn_flood,
        "port_scan": HackerAttackSimulator.craft_port_scan,
        "http_flood": HackerAttackSimulator.craft_http_flood,
        "brute_force": HackerAttackSimulator.craft_brute_force,
        "zero_day": HackerAttackSimulator.craft_zero_day,
    }

    craft_fn = attack_map.get(attack_type.lower(), HackerAttackSimulator.craft_syn_flood)
    attack_df = craft_fn(n_samples)

    if mix_benign:
        # Load sample benign baseline to simulate realistic mixed enterprise feed
        baseline_file = DATA_DIR / "train_known_traffic.csv"
        if baseline_file.exists():
            base_df = pd.read_csv(baseline_file)
            benign_df = base_df[base_df["label"] == "BENIGN"].sample(n=min(200, len(base_df)), random_state=42)
        else:
            benign_df = pd.DataFrame()

        # Combine: First 20 benign, then attack burst, then more benign
        combined_df = pd.concat([benign_df.head(25), attack_df, benign_df.tail(75)], ignore_index=True)
    else:
        combined_df = attack_df

    combined_df.to_csv(LIVE_ATTACK_FILE, index=False)
    print(f"🔥 Successfully injected {len(attack_df)} [{attack_type.upper()}] attack packets!")
    print(f"📁 Live Telemetry Stream written to: {LIVE_ATTACK_FILE}")
    print(f"👉 Select 'live_attack_stream.csv' in your CyberGuard Dashboard to visualize the defense!")
    return LIVE_ATTACK_FILE


def interactive_cli():
    print("\n" + "=" * 70)
    print("   🔴 CYBERGUARD RED TEAM // LIVE HACKER ATTACK SIMULATOR")
    print("=" * 70)
    print("Select an offensive attack vector to launch against the AI defense:")
    print("  [1] 🌊 TCP SYN Flood (State Table Saturation & conntrack exhaustion)")
    print("  [2] 🕵️ Stealth Port Scan (Reconnaissance hole-sweeping across ports)")
    print("  [3] 💥 DDoS HTTP Flood (Layer-7 Application Worker Starvation)")
    print("  [4] 🔑 SSH Brute Force (Dictionary Credential Stuffing on port 22)")
    print("  [5] ☣️ Polymorphic Zero-Day Exploit (Deep Autoencoder Manifold Evasion)")
    print("=" * 70)

    choice = input("Enter choice (1-5) [default: 1]: ").strip() or "1"
    attack_map = {
        "1": "syn_flood",
        "2": "port_scan",
        "3": "http_flood",
        "4": "brute_force",
        "5": "zero_day",
    }
    attack_name = attack_map.get(choice, "syn_flood")
    n = input("Number of attack packet flows to inject [default: 120]: ").strip() or "120"

    print(f"\n🚀 Launching {attack_name.upper()} attack...")
    time.sleep(0.8)
    out_file = inject_attack(attack_name, int(n), mix_benign=True)

    print("\n" + "-" * 70)
    print("🎯 NEXT STEP TO VISUALIZE IN DASHBOARD:")
    print("1. Open your dashboard: http://localhost:8501")
    print("2. In the sidebar under 'Select Traffic Dataset', choose:")
    print("   👉 'live_attack_stream.csv'")
    print("3. Watch CyberGuard intercept the attack, explain it via SHAP/LIME,")
    print("   and generate the defensive firewall rule to neutralize it!")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberGuard Live Attack Simulator")
    parser.add_argument("--type", type=str, choices=["syn_flood", "port_scan", "http_flood", "brute_force", "zero_day"], default=None)
    parser.add_argument("--count", type=int, default=120)
    args = parser.parse_args()

    if args.type:
        inject_attack(args.type, args.count)
    else:
        interactive_cli()
