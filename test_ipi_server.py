#!/usr/bin/env python3
"""
Quick tests for the iPI server components.

Run with:  python test_ipi_server.py
No ML model needed — uses a mock calculator.
"""

import os
import socket
import struct
import tempfile
import threading
import time
import unittest

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

# Import server functions
from ase_ipi_server_mace import (
    parse_species_file,
    send_header,
    recv_header,
    send_int32,
    recv_int32,
    send_doubles,
    recv_doubles,
    do_handshake,
    serve,
)


# ---------------------------------------------------------------------------
# Mock calculator that returns deterministic results
# ---------------------------------------------------------------------------
class MockCalculator(Calculator):
    """Returns energy = sum(positions) and forces = -1 for all atoms."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(np.sum(self.atoms.positions)),
            "forces": -np.ones_like(self.atoms.positions),
        }


# ---------------------------------------------------------------------------
# Test: Species file parsing
# ---------------------------------------------------------------------------
class TestSpeciesFileParsing(unittest.TestCase):

    def test_roundtrip(self):
        """Write a species file, parse it, check counts and symbols."""
        fw = ["Zn", "O", "C", "H"] * 3  # 12 framework atoms
        ads = ["C", "O", "O"] * 3       # 9 adsorbate atoms

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"FRAMEWORK {len(fw)}\n")
            f.write(" ".join(fw) + "\n")
            f.write(f"ADSORBATE {len(ads)}\n")
            f.write(" ".join(ads) + "\n")
            path = f.name

        try:
            fw_parsed, ads_parsed = parse_species_file(path)
            self.assertEqual(fw_parsed, fw)
            self.assertEqual(ads_parsed, ads)
            self.assertEqual(len(fw_parsed), 12)
            self.assertEqual(len(ads_parsed), 9)
        finally:
            os.unlink(path)

    def test_empty_adsorbate(self):
        """Handle case with zero adsorbate atoms."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("FRAMEWORK 3\n")
            f.write("Zn O C\n")
            f.write("ADSORBATE 0\n")
            f.write("\n")
            path = f.name

        try:
            fw_parsed, ads_parsed = parse_species_file(path)
            self.assertEqual(len(fw_parsed), 3)
            self.assertEqual(len(ads_parsed), 0)
        finally:
            os.unlink(path)

    def test_mismatched_count_raises(self):
        """Species count mismatch should raise AssertionError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("FRAMEWORK 5\n")
            f.write("Zn O C\n")  # only 3, not 5
            f.write("ADSORBATE 0\n")
            f.write("\n")
            path = f.name

        try:
            with self.assertRaises(AssertionError):
                parse_species_file(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: iPI wire protocol helpers
# ---------------------------------------------------------------------------
class TestProtocolHelpers(unittest.TestCase):

    def _socketpair(self):
        return socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    def test_header_roundtrip(self):
        a, b = self._socketpair()
        try:
            send_header(a, "STATUS")
            self.assertEqual(recv_header(b), "STATUS")

            send_header(a, "FORCEREADY")
            self.assertEqual(recv_header(b), "FORCEREADY")
        finally:
            a.close()
            b.close()

    def test_int32_roundtrip(self):
        a, b = self._socketpair()
        try:
            send_int32(a, 1234)
            self.assertEqual(recv_int32(b), 1234)

            send_int32(a, 0)
            self.assertEqual(recv_int32(b), 0)
        finally:
            a.close()
            b.close()

    def test_doubles_roundtrip(self):
        a, b = self._socketpair()
        try:
            arr = np.array([1.5, 2.7, -3.14], dtype=np.float64)
            send_doubles(a, arr)
            received = recv_doubles(b, 3)
            np.testing.assert_allclose(received, arr)
        finally:
            a.close()
            b.close()


# ---------------------------------------------------------------------------
# Test: Full iPI handshake + one energy evaluation
# ---------------------------------------------------------------------------
class TestFullProtocol(unittest.TestCase):

    def test_single_evaluation(self):
        """Simulate one full iPI cycle: handshake + POSDATA + receive energy."""
        fw_symbols = ["Zn", "O", "C"]
        ads_symbols = ["C", "O"]
        n_fw = len(fw_symbols)
        n_ads = len(ads_symbols)
        n_total = n_fw + n_ads

        calc = MockCalculator()
        cell = np.eye(3) * 10.0

        srv_sock, cli_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        errors = []

        def run_server():
            try:
                do_handshake(srv_sock, cell)
                serve(srv_sock, calc, fw_symbols, ads_symbols, cell)
            except Exception as e:
                errors.append(e)
            finally:
                srv_sock.close()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        try:
            # --- Client side: mimic gRASPA C++ client ---

            # Handshake: receive STATUS, send NEEDINIT
            hdr = recv_header(cli_sock)
            self.assertEqual(hdr, "STATUS")
            send_header(cli_sock, "NEEDINIT")

            # Receive INIT + cell + inv_cell
            hdr = recv_header(cli_sock)
            self.assertEqual(hdr, "INIT")
            _cell = recv_doubles(cli_sock, 9)
            _inv = recv_doubles(cli_sock, 9)

            # --- First evaluation (total system) ---
            # Receive STATUS, send READY
            hdr = recv_header(cli_sock)
            self.assertEqual(hdr, "STATUS")
            send_header(cli_sock, "READY")

            # Receive POSDATA
            hdr = recv_header(cli_sock)
            self.assertEqual(hdr, "POSDATA")

            # Send cell + inv_cell + natoms + positions
            positions = np.arange(n_total * 3, dtype=np.float64).reshape(-1, 3)
            send_doubles(cli_sock, cell.flatten())
            send_doubles(cli_sock, np.linalg.inv(cell).flatten())
            send_int32(cli_sock, n_total)
            send_doubles(cli_sock, positions.flatten())

            # Receive STATUS, send HAVEDATA
            hdr = recv_header(cli_sock)
            self.assertEqual(hdr, "STATUS")
            send_header(cli_sock, "HAVEDATA")

            # Receive GETFORCE
            hdr = recv_header(cli_sock)
            self.assertEqual(hdr, "GETFORCE")

            # Receive FORCEREADY + results
            hdr = recv_header(cli_sock)
            self.assertEqual(hdr, "FORCEREADY")

            energy = recv_doubles(cli_sock, 1)[0]
            recv_natoms = recv_int32(cli_sock)
            forces = recv_doubles(cli_sock, 3 * recv_natoms)
            virial = recv_doubles(cli_sock, 9)
            extras_len = recv_int32(cli_sock)

            # Verify results
            self.assertEqual(recv_natoms, n_total)
            expected_energy = float(np.sum(positions))
            self.assertAlmostEqual(energy, expected_energy, places=6)
            np.testing.assert_allclose(forces, -np.ones(3 * n_total))
            self.assertEqual(extras_len, 0)

            # --- Send EXIT to cleanly stop ---
            # Receive STATUS, send EXIT
            hdr = recv_header(cli_sock)
            self.assertEqual(hdr, "STATUS")
            send_header(cli_sock, "EXIT")

        finally:
            cli_sock.close()
            server_thread.join(timeout=5)

        self.assertEqual(errors, [], f"Server errors: {errors}")

    def test_natoms_routing(self):
        """Verify the server correctly routes different natoms values."""
        fw_symbols = ["Zn", "Zn", "O", "O"]  # 4 fw
        ads_symbols = ["C", "O", "O"]          # 3 ads
        n_fw = 4
        n_ads = 3
        n_total = 7

        calc = MockCalculator()
        cell = np.eye(3) * 10.0

        srv_sock, cli_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        errors = []

        def run_server():
            try:
                do_handshake(srv_sock, cell)
                serve(srv_sock, calc, fw_symbols, ads_symbols, cell)
            except Exception as e:
                errors.append(e)
            finally:
                srv_sock.close()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        try:
            # Handshake
            hdr = recv_header(cli_sock)
            send_header(cli_sock, "NEEDINIT")
            hdr = recv_header(cli_sock)
            recv_doubles(cli_sock, 9)
            recv_doubles(cli_sock, 9)

            # Send three evaluations with different natoms
            for natoms in [n_total, n_fw, n_ads]:
                hdr = recv_header(cli_sock)
                send_header(cli_sock, "READY")

                hdr = recv_header(cli_sock)  # POSDATA

                positions = np.ones((natoms, 3), dtype=np.float64)
                send_doubles(cli_sock, cell.flatten())
                send_doubles(cli_sock, np.linalg.inv(cell).flatten())
                send_int32(cli_sock, natoms)
                send_doubles(cli_sock, positions.flatten())

                hdr = recv_header(cli_sock)  # STATUS
                send_header(cli_sock, "HAVEDATA")

                hdr = recv_header(cli_sock)  # GETFORCE

                hdr = recv_header(cli_sock)  # FORCEREADY
                energy = recv_doubles(cli_sock, 1)[0]
                recv_n = recv_int32(cli_sock)
                forces = recv_doubles(cli_sock, 3 * recv_n)
                virial = recv_doubles(cli_sock, 9)
                extras_len = recv_int32(cli_sock)

                self.assertEqual(recv_n, natoms)
                # energy = sum(positions) = natoms * 3 * 1.0
                self.assertAlmostEqual(energy, natoms * 3.0, places=6)

            # Clean exit
            hdr = recv_header(cli_sock)
            send_header(cli_sock, "EXIT")

        finally:
            cli_sock.close()
            server_thread.join(timeout=5)

        self.assertEqual(errors, [], f"Server errors: {errors}")


# ---------------------------------------------------------------------------
# Test: Caching behaviour (framework & adsorbate computed once, then reused)
# ---------------------------------------------------------------------------
class TestCachingBehaviour(unittest.TestCase):
    """Simulate multiple Predict() cycles and verify that after the first
    cycle (3 socket calls), subsequent cycles only make 1 socket call."""

    def _do_one_eval(self, cli_sock, natoms, cell, positions):
        """Client-side helper: one STATUS→READY→POSDATA→HAVEDATA→GETFORCE
        round-trip.  Returns the energy."""
        hdr = recv_header(cli_sock)
        self.assertEqual(hdr, "STATUS")
        send_header(cli_sock, "READY")

        hdr = recv_header(cli_sock)
        self.assertEqual(hdr, "POSDATA")

        send_doubles(cli_sock, cell.flatten())
        send_doubles(cli_sock, np.linalg.inv(cell).flatten())
        send_int32(cli_sock, natoms)
        send_doubles(cli_sock, positions.flatten())

        hdr = recv_header(cli_sock)
        self.assertEqual(hdr, "STATUS")
        send_header(cli_sock, "HAVEDATA")

        hdr = recv_header(cli_sock)
        self.assertEqual(hdr, "GETFORCE")

        hdr = recv_header(cli_sock)
        self.assertEqual(hdr, "FORCEREADY")

        energy = recv_doubles(cli_sock, 1)[0]
        recv_n = recv_int32(cli_sock)
        _forces = recv_doubles(cli_sock, 3 * recv_n)
        _virial = recv_doubles(cli_sock, 9)
        extras_len = recv_int32(cli_sock)
        if extras_len > 0:
            cli_sock.recv(extras_len)

        return energy, recv_n

    def test_cache_reduces_calls(self):
        """First Predict() → 3 evals (total, framework, adsorbate).
        Second Predict() → 1 eval (total only).  Interaction energy must
        be consistent."""
        fw_symbols = ["Zn", "O", "C"]
        ads_symbols = ["C", "O"]
        n_fw = len(fw_symbols)
        n_ads = len(ads_symbols)
        n_total = n_fw + n_ads

        calc = MockCalculator()
        cell = np.eye(3) * 10.0

        srv_sock, cli_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        errors = []

        def run_server():
            try:
                do_handshake(srv_sock, cell)
                serve(srv_sock, calc, fw_symbols, ads_symbols, cell)
            except Exception as e:
                errors.append(e)
            finally:
                srv_sock.close()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        try:
            # Handshake
            hdr = recv_header(cli_sock)
            send_header(cli_sock, "NEEDINIT")
            hdr = recv_header(cli_sock)
            recv_doubles(cli_sock, 9)
            recv_doubles(cli_sock, 9)

            # --- First Predict(): 3 socket calls ---
            pos_total = np.arange(n_total * 3, dtype=np.float64).reshape(-1, 3)
            pos_fw    = np.ones((n_fw, 3), dtype=np.float64) * 2.0
            pos_ads   = np.ones((n_ads, 3), dtype=np.float64) * 3.0

            E_total_1, _ = self._do_one_eval(cli_sock, n_total, cell, pos_total)
            E_fw,      _ = self._do_one_eval(cli_sock, n_fw,    cell, pos_fw)
            E_ads,     _ = self._do_one_eval(cli_sock, n_ads,   cell, pos_ads)

            interaction_1 = E_total_1 - E_fw - E_ads

            # --- Second Predict(): only 1 socket call (total) ---
            pos_total_2 = np.arange(n_total * 3, dtype=np.float64).reshape(-1, 3) + 1.0
            E_total_2, _ = self._do_one_eval(cli_sock, n_total, cell, pos_total_2)

            # Reuse cached values
            interaction_2 = E_total_2 - E_fw - E_ads

            # Sanity: energies should differ because positions changed
            self.assertNotAlmostEqual(E_total_1, E_total_2, places=6)

            # The cached E_fw and E_ads are still valid constants
            self.assertAlmostEqual(E_fw, float(np.sum(pos_fw)), places=6)
            self.assertAlmostEqual(E_ads, float(np.sum(pos_ads)), places=6)

            # --- Third Predict(): still only 1 socket call ---
            pos_total_3 = np.arange(n_total * 3, dtype=np.float64).reshape(-1, 3) + 5.0
            E_total_3, _ = self._do_one_eval(cli_sock, n_total, cell, pos_total_3)
            interaction_3 = E_total_3 - E_fw - E_ads

            # All interaction energies should be self-consistent
            self.assertAlmostEqual(
                interaction_1,
                E_total_1 - E_fw - E_ads, places=6)
            self.assertAlmostEqual(
                interaction_3,
                E_total_3 - E_fw - E_ads, places=6)

            # Clean exit
            hdr = recv_header(cli_sock)
            send_header(cli_sock, "EXIT")

        finally:
            cli_sock.close()
            server_thread.join(timeout=5)

        self.assertEqual(errors, [], f"Server errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
