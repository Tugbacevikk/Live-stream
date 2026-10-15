"""API güvenliği: anahtar kontrolü ve callback adresi doğrulaması.

fine-tuning reposundaki ssrf_validator / verify_api_key'den uyarlandı. Fark: sistem
air-gapped bir iç ağda başka bir sisteme entegre edileceği için callback çoğu zaman
özel (iç ağ) bir adrestir. Bu yüzden:
  - WEBHOOK_ALLOWED_HOSTS doluysa: yalnızca bu adreslere izin (iç ağ adresi olabilir)
  - boşsa, production'da: özel / loopback / link-local adresler reddedilir (SSRF)
  - development'ta: her http(s) adresi serbest (yerel denemeler için)
"""
from __future__ import annotations

import hmac
import ipaddress
import socket
import urllib.parse

from fastapi import Header, HTTPException

from live_stream.interface.settings import Settings


def api_key_dependency(settings: Settings):
    expected = settings.api_key
    if settings.is_production and not expected:
        raise RuntimeError("APP_ENV=production iken API_KEY tanımlanmalı (.env)")

    def verify(x_api_key: str | None = Header(default=None)):
        if expected and not hmac.compare_digest(x_api_key or "", expected):
            raise HTTPException(401, "geçersiz ya da eksik X-API-Key")

    return verify


def callback_problem(url: str | None, settings: Settings) -> str | None:
    """Sorun yoksa None, varsa açıklama."""
    if not url:
        return None
    try:
        p = urllib.parse.urlparse(url.strip())
    except ValueError:
        return "callback_url ayrıştırılamadı"
    if p.scheme not in ("http", "https") or not p.hostname:
        return "callback_url http:// ya da https:// ile başlamalı"
    host = p.hostname.lower()
    if settings.webhook_allowed_hosts:
        if host not in settings.webhook_allowed_hosts:
            return f"callback adresi izin listesinde değil ({host}); WEBHOOK_ALLOWED_HOSTS"
        return None
    if not settings.is_production:
        return None
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return f"callback adresi çözülemedi: {host}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            return (f"callback iç ağ adresine gidiyor ({ip}); iç ağ alıcısıysa "
                    "WEBHOOK_ALLOWED_HOSTS'a ekleyin")
    return None
