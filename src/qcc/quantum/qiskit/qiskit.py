"""
_summary_
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit, Aer, execute, ClassicalRegister
from qiskit.tools import job_monitor

from qcc.quantum import from_counts

if TYPE_CHECKING:
    from typing import Sequence


def potato(
    qc: QuantumCircuit,
    backend=None,
    shots: int | None = None,
    noisy_execution: bool = False,
    seed: int = 42069,
    meas: Sequence | None = None,
    monitor_job: bool = False,
):
    if backend is None:
        backend = Aer.get_backend("aer_simulator")
    if shots is None:
        shots = backend.configuration().max_shots

    if noisy_execution:
        if meas is None:
            qc.measure_all()
        else:
            creg = ClassicalRegister(len(meas))
            qc.add_register(creg)
            qc.measure(meas, creg)
    else:
        qc.save_statevector()

    job = execute(qc, backend=backend, shots=shots, seed_simulator=seed)
    if monitor_job:
        job_monitor(job)

    result = job.result()

    if noisy_execution:
        counts = result.get_counts(qc)
        psi_out = from_counts(counts, shots=shots, num_qubits=qc.num_qubits)
    else:
        psi_out = result.get_statevector(qc).data

    return psi_out
