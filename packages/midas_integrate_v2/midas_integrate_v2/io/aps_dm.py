"""Item 35 — APS Data Management (DM) connector.

Thin client over the APS DM REST API for experiment / file lookup.
**This module ships as an infrastructure scaffold** — production use
requires a DM auth token + sandbox / production endpoint. Tests run
against a mocked REST surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional


@dataclass
class ExperimentMetadata:
    experiment_id: str
    title: str
    files: List[str]
    pi: Optional[str] = None
    proposal_id: Optional[str] = None


class APSDMClient:
    """Minimal-surface client for APS DM.

    Parameters
    ----------
    dm_url :
        Base URL of the DM REST endpoint
        (e.g., ``https://dm-prod.aps.anl.gov``).
    auth_token :
        Optional bearer token. Without it, anonymous reads of public
        experiments only.
    session :
        Optional ``requests.Session`` for tests / mocking.
    """

    def __init__(
        self,
        *,
        dm_url: str,
        auth_token: Optional[str] = None,
        session=None,
    ):
        self.dm_url = dm_url.rstrip("/")
        self.auth_token = auth_token
        if session is not None:
            self._session = session
        else:
            try:
                import requests
                self._session = requests.Session()
            except ImportError as e:
                raise ImportError(
                    "APSDMClient requires `requests`; "
                    "pip install 'midas-integrate-v2[aps-dm]'"
                ) from e

    def _headers(self) -> Mapping[str, str]:
        h = {"Accept": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    def get_experiment(self, experiment_id: str) -> ExperimentMetadata:
        url = f"{self.dm_url}/experiments/{experiment_id}"
        resp = self._session.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return ExperimentMetadata(
            experiment_id=str(data.get("id", experiment_id)),
            title=data.get("title", ""),
            files=list(data.get("files", [])),
            pi=data.get("pi"),
            proposal_id=data.get("proposal_id"),
        )

    def list_files(
        self, experiment_id: str, *, role: str = "calibrant",
    ) -> List[Path]:
        url = f"{self.dm_url}/experiments/{experiment_id}/files"
        resp = self._session.get(
            url, headers=self._headers(),
            params={"role": role}, timeout=30,
        )
        resp.raise_for_status()
        return [Path(p) for p in resp.json().get("files", [])]

    def fetch_to_local(
        self, experiment_id: str, *, dest: Path,
    ) -> Path:
        """Trigger a Globus transfer to ``dest`` for the experiment data.

        Returns the local path. Note: this method intentionally raises
        if globus-sdk isn't available, since Globus is the required
        APS-side transfer mechanism.
        """
        try:
            import globus_sdk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "fetch_to_local requires globus-sdk; "
                "pip install 'midas-integrate-v2[aps-dm]'"
            ) from e
        # In production this triggers a Globus transfer via the DM
        # workflow. Implementation deferred until we have a sandbox
        # token to test against.
        raise NotImplementedError(
            "fetch_to_local: integrate with APS DM globus-sdk pending "
            "sandbox token from APS DM team"
        )


__all__ = ["APSDMClient", "ExperimentMetadata"]
