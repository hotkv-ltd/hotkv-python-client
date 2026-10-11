# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Geospatial commands."""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Tuple

from .._typing import KeyT, ValueT
from ._shared import CallbackMap, ReadonlySet, as_float

__all__ = ["GeoCommands", "GeoSearchResult", "CALLBACKS", "READONLY"]


class GeoSearchResult(NamedTuple):
    """One `geosearch()` match. Fields not requested via `with*` are `None`."""

    member: Any
    distance: Optional[float]
    geohash: Optional[int]
    coordinates: Optional[Tuple[float, float]]


def _cb_geopos(reply: Any, **_options: Any) -> List[Optional[Tuple[float, float]]]:
    return [None if item is None else (as_float(item[0]), as_float(item[1])) for item in reply]


def _cb_geodist(reply: Any, **_options: Any) -> Optional[float]:
    return as_float(reply)


def _cb_geosearch(
    reply: Any,
    withcoord: bool = False,
    withdist: bool = False,
    withhash: bool = False,
    **_options: Any,
) -> Any:
    if not (withcoord or withdist or withhash):
        return reply
    results: List[GeoSearchResult] = []
    for item in reply:
        member = item[0]
        idx = 1
        distance = geohash = coordinates = None
        if withdist:
            distance = as_float(item[idx])
            idx += 1
        if withhash:
            geohash = int(item[idx])
            idx += 1
        if withcoord:
            lon, lat = item[idx]
            coordinates = (as_float(lon), as_float(lat))
            idx += 1
        results.append(GeoSearchResult(member, distance, geohash, coordinates))
    return results


CALLBACKS: CallbackMap = {
    "GEOPOS": _cb_geopos,
    "GEODIST": _cb_geodist,
    "GEOSEARCH": _cb_geosearch,
}

READONLY: ReadonlySet = frozenset({"GEOPOS", "GEODIST", "GEOHASH", "GEOSEARCH"})


class GeoCommands:
    """Typed wrappers for geospatial commands."""

    def geoadd(
        self,
        key: KeyT,
        members: Any,
        *,
        nx: bool = False,
        xx: bool = False,
        ch: bool = False,
    ) -> int:
        """Add `{member: (longitude, latitude)}` entries, returning the number added."""
        args: List[Any] = [key]
        if nx:
            args.append("NX")
        elif xx:
            args.append("XX")
        if ch:
            args.append("CH")
        for member, (lon, lat) in members.items():
            args += [lon, lat, member]
        return self.execute_command("GEOADD", *args)

    def geopos(self, key: KeyT, *members: ValueT) -> List[Optional[Tuple[float, float]]]:
        """Return `(longitude, latitude)` for each member, `None` if unset."""
        return self.execute_command("GEOPOS", key, *members)

    def geodist(
        self, key: KeyT, member1: ValueT, member2: ValueT, unit: str = "m"
    ) -> Optional[float]:
        """Distance between two members, in `unit` ("m"/"km"/"mi"/"ft")."""
        return self.execute_command("GEODIST", key, member1, member2, unit)

    def geohash(self, key: KeyT, *members: ValueT) -> List[Optional[bytes]]:
        """Return the standard geohash string for each member."""
        return self.execute_command("GEOHASH", key, *members)

    def geosearch(
        self,
        key: KeyT,
        *,
        member: Optional[ValueT] = None,
        longitude: Optional[float] = None,
        latitude: Optional[float] = None,
        radius: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        unit: str = "m",
        sort: Optional[str] = None,
        count: Optional[int] = None,
        any_order: bool = False,
        withcoord: bool = False,
        withdist: bool = False,
        withhash: bool = False,
    ) -> Any:
        """Search by radius (`radius` + `unit`) or box (`width`/`height` +
        `unit`), centered on `member` or `(longitude, latitude)`."""
        args = self._search_args(
            member=member,
            longitude=longitude,
            latitude=latitude,
            radius=radius,
            width=width,
            height=height,
            unit=unit,
            sort=sort,
            count=count,
            any_order=any_order,
        )
        if withcoord:
            args.append("WITHCOORD")
        if withdist:
            args.append("WITHDIST")
        if withhash:
            args.append("WITHHASH")
        return self.execute_command(
            "GEOSEARCH", key, *args, withcoord=withcoord, withdist=withdist, withhash=withhash
        )

    def geosearchstore(
        self,
        destination: KeyT,
        source: KeyT,
        *,
        member: Optional[ValueT] = None,
        longitude: Optional[float] = None,
        latitude: Optional[float] = None,
        radius: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        unit: str = "m",
        sort: Optional[str] = None,
        count: Optional[int] = None,
        any_order: bool = False,
        storedist: bool = False,
    ) -> int:
        """Like `geosearch()`, storing the matches into `destination`."""
        args = self._search_args(
            member=member,
            longitude=longitude,
            latitude=latitude,
            radius=radius,
            width=width,
            height=height,
            unit=unit,
            sort=sort,
            count=count,
            any_order=any_order,
        )
        if storedist:
            args.append("STOREDIST")
        return self.execute_command("GEOSEARCHSTORE", destination, source, *args)

    def _search_args(
        self,
        *,
        member: Optional[ValueT],
        longitude: Optional[float],
        latitude: Optional[float],
        radius: Optional[float],
        width: Optional[float],
        height: Optional[float],
        unit: str,
        sort: Optional[str],
        count: Optional[int],
        any_order: bool,
    ) -> List[Any]:
        args: List[Any] = []
        if member is not None:
            args += ["FROMMEMBER", member]
        else:
            args += ["FROMLONLAT", longitude, latitude]
        if radius is not None:
            args += ["BYRADIUS", radius, unit]
        else:
            args += ["BYBOX", width, height, unit]
        if sort:
            args.append(sort.upper())
        if count is not None:
            args += ["COUNT", count]
            if any_order:
                args.append("ANY")
        return args
