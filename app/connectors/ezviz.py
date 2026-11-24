"""EZVIZ Open Platform (Hik-Connect / EZVIZ cloud) connector.

Lets the server read a store camera through the EZVIZ cloud -- the same cloud
the owner's phone app uses -- so nothing has to be installed in the store.

Flow: appKey/appSecret -> accessToken (+ regional areaDomain)
      -> add the store's device (serial + 6-letter verification code on its sticker)
      -> request a live stream address (HLS/RTMP/FLV) -> read it with OpenCV/FFmpeg.

VERIFY ON A REAL ACCOUNT (scripts/ezviz_probe.py does this): the token endpoint
and /api/lapp/ form-POST style are documented by EZVIZ; the exact parameters of
the live-address endpoint, whether it is enabled for overseas accounts, its
pricing and stream-time limits are NOT confirmed yet.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

DEFAULT_BASE = "https://open.ezvizlife.com"   # global entry; token response tells the regional host
PROTOCOLS = {"ezopen": 1, "hls": 2, "rtmp": 3, "flv": 4}


class EzvizError(RuntimeError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"EZVIZ {code}: {msg}")
        self.code = code


@dataclass
class LiveAddress:
    url: str
    expires_at: float


class EzvizClient:
    def __init__(self, app_key: str, app_secret: str, base_url: str = DEFAULT_BASE,
                 client: httpx.Client | None = None):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url.rstrip("/")
        self.area_url: str | None = None
        self.client = client or httpx.Client(timeout=20)
        self._token: str | None = None
        self._token_exp = 0.0

    def _post(self, base: str, path: str, **data) -> dict:
        r = self.client.post(f"{base}{path}", data={k: v for k, v in data.items() if v is not None})
        r.raise_for_status()
        body = r.json()
        code = str(body.get("code", body.get("resultCode", "")))
        if code not in ("200", "0"):
            raise EzvizError(code, body.get("msg", body.get("message", "")))
        return body.get("data") or {}

    def token(self) -> str:
        if self._token and time.time() < self._token_exp - 3600:
            return self._token
        d = self._post(self.base_url, "/api/lapp/token/get", appKey=self.app_key, appSecret=self.app_secret)
        self._token = d["accessToken"]
        exp = float(d.get("expireTime", 0))
        self._token_exp = exp / 1000 if exp > 1e11 else (exp or time.time() + 6 * 86400)
        if d.get("areaDomain"):
            self.area_url = d["areaDomain"].rstrip("/")
        return self._token

    def call(self, path: str, **data) -> dict:
        tok = self.token()
        try:
            return self._post(self.area_url or self.base_url, path, accessToken=tok, **data)
        except EzvizError as e:
            if e.code in ("10002", "10001"):  # token expired / invalid -> refresh once
                self._token = None
                return self._post(self.area_url or self.base_url, path, accessToken=self.token(), **data)
            raise

    # --- devices ---
    def add_device(self, serial: str, verify_code: str) -> dict:
        return self.call("/api/lapp/device/add", deviceSerial=serial, validateCode=verify_code)

    def device_info(self, serial: str) -> dict:
        return self.call("/api/lapp/device/info", deviceSerial=serial)

    def cameras(self, serial: str) -> list[dict]:
        d = self.call("/api/lapp/device/camera/list", deviceSerial=serial)
        return d if isinstance(d, list) else d.get("list", [])

    # --- live stream ---
    def live_address(self, serial: str, channel: int = 1, protocol: str = "hls", quality: int = 2,
                     expire_seconds: int = 86400, verify_code: str | None = None) -> LiveAddress:
        """quality: 1 = HD (main stream), 2 = smooth (sub stream) -- sub stream is enough and cheaper."""
        d = self.call("/api/lapp/live/address/get", deviceSerial=serial, channelNo=channel,
                      protocol=PROTOCOLS[protocol], quality=quality, expireTime=expire_seconds,
                      code=verify_code)
        url = d.get("url") or d.get("hls") or d.get("rtmp")
        if not url:
            raise EzvizError("no-url", f"unexpected response: {d}")
        exp = d.get("expireTime")
        expires_at = (float(exp) / 1000 if exp and float(exp) > 1e11 else time.time() + expire_seconds)
        return LiveAddress(url, expires_at)


class EzvizSource:
    """Video source for the pipeline: fetches a live address and renews it before it expires.

    Wraps app.stream.RtspSource (works for any FFmpeg URL: HLS, RTMP, FLV)."""

    def __init__(self, client: EzvizClient, serial: str, channel: int = 1, verify_code: str | None = None,
                 target_fps: float = 2.0, protocol: str = "hls"):
        from ..stream import RtspSource

        self._RtspSource = RtspSource
        self.client, self.serial, self.channel = client, serial, channel
        self.verify_code, self.protocol, self.target_fps = verify_code, protocol, target_fps
        self.fps = target_fps
        self._addr: LiveAddress | None = None
        self._src = None

    def _refresh(self):
        self._addr = self.client.live_address(self.serial, self.channel, self.protocol,
                                              verify_code=self.verify_code)
        if self._src:
            self._src.stop()
        self._src = self._RtspSource(self._addr.url, self.target_fps)
        log.info("EZVIZ oqim manzili yangilandi: %s/%s", self.serial, self.channel)

    def frames(self):
        i = 0
        while True:
            if self._addr is None or time.time() > self._addr.expires_at - 600:
                self._refresh()
            for _, ts, frame in self._src.frames():
                yield i, ts, frame
                i += 1
                if time.time() > self._addr.expires_at - 600:
                    break
            else:
                return

    def stop(self):
        if self._src:
            self._src.stop()
