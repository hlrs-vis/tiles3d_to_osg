
#!/usr/bin/env python3
"""Download 3D Tiles, extract B3DM payloads, and write directory-level OSG pages.

Usage:
  python tiles3d_to_osg.py URL [--output DIR] [--min-lod L5]

The downloaded directory mirrors the server URL path layout. Converted GLB
geometry and tileset hierarchy pages are written below output/osg/ using the
same relative layout. Each PagedLOD loads its higher-detail tile pages.
Pillow is required to vertically mirror JPEG textures; install it with
`python -m pip install Pillow`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import functools
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


USER_AGENT = "tiles3d-to-osg/1.0"


def safe_relative_path(url: str, base_url: str, fallback: str) -> Path:
    """Map a URL's path relative to the initial URL directory into local paths."""
    parsed, base = urllib.parse.urlsplit(url), urllib.parse.urlsplit(base_url)
    path = urllib.parse.unquote(parsed.path)
    base_dir = urllib.parse.unquote(str(PurePosixPath(base.path).parent))
    try:
        rel = os.path.relpath(path, base_dir).replace("\\", "/")
    except ValueError:
        rel = PurePosixPath(path).name
    rel_path = PurePosixPath(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        # Keep paths from a different root/host under a stable host subdirectory.
        rel_path = PurePosixPath(urllib.parse.urlsplit(url).netloc) / PurePosixPath(path.lstrip("/"))
    if not rel_path.parts or str(rel_path) in (".", ""):
        rel_path = PurePosixPath(fallback)
    return Path(*rel_path.parts)


def fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def b3dm_rtc_center(data: bytes) -> list[float]:
    """Read the optional RTC_CENTER from a B3DM feature table."""
    if len(data) < 28 or data[:4] != b"b3dm":
        return [0.0, 0.0, 0.0]
    ft_json_len = struct.unpack_from("<I", data, 12)[0]
    if 28 + ft_json_len > len(data) or not ft_json_len:
        return [0.0, 0.0, 0.0]
    try:
        feature_table = json.loads(data[28:28 + ft_json_len].decode("utf-8").rstrip("\x00 \t\r\n"))
        value = feature_table.get("RTC_CENTER")
        if isinstance(value, list) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
        if isinstance(value, dict) and isinstance(value.get("byteOffset"), int):
            ft_bin_start = 28 + ft_json_len
            offset = ft_bin_start + value["byteOffset"]
            if offset + 12 <= len(data):
                return list(struct.unpack_from("<3f", data, offset))
    except (ValueError, UnicodeDecodeError, TypeError):
        pass
    return [0.0, 0.0, 0.0]


def decode_b3dm(data: bytes) -> tuple[bytes, list[float]] | None:
    """Return the embedded GLB and RTC_CENTER from a B3DM v1 payload."""
    if len(data) < 28 or data[:4] != b"b3dm":
        return None
    version, byte_length, ft_json_len, ft_bin_len, bt_json_len, bt_bin_len = struct.unpack_from("<6I", data, 4)
    if version != 1 or byte_length > len(data) or byte_length < 28:
        return None
    offset = 28 + ft_json_len + ft_bin_len + bt_json_len + bt_bin_len
    if offset > byte_length:
        return None
    # Tiles often pad the feature-table JSON to an 8-byte boundary. The header
    # lengths normally include that padding; this fallback tolerates malformed
    # length fields while still requiring an actual GLB signature.
    candidates = [offset]
    candidates.extend(i for i in range(max(28, offset - 8), min(byte_length, offset + 9)) if i not in candidates)
    for start in candidates:
        if data[start:start + 4] == b"glTF" and start + 12 <= byte_length:
            glb_len = struct.unpack_from("<I", data, start + 8)[0]
            end = start + glb_len
            if glb_len >= 12 and end <= byte_length:
                return data[start:end], b3dm_rtc_center(data)
    return None


def fmt(value: float) -> str:
    return f"{value:.9g}"


def osg_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def mirror_jpeg_vertically(data: bytes) -> bytes:
    """Vertically mirror a JPEG image and return the resulting JPEG bytes."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Mirroring JPEG textures requires Pillow; install it with 'python -m pip install Pillow'") from exc
    with Image.open(io.BytesIO(data)) as image:
        mirrored = ImageOps.flip(image)
        output = io.BytesIO()
        mirrored.save(output, format="JPEG", quality=100, subsampling=0)
        return output.getvalue()


IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
GLTF_UP_TO_Z_UP = {
    "Y": [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1],
    "Z": IDENTITY,
    "X": [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
}


def multiply_matrices(a: list[float], b: list[float]) -> list[float]:
    """Multiply column-major 4x4 matrices (the 3D Tiles convention)."""
    return [sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
            for col in range(4) for row in range(4)]


def translation_matrix(offset: list[float] | tuple[float, float, float]) -> list[float]:
    matrix = IDENTITY.copy()
    matrix[12:15] = list(offset)
    return matrix


def inverse_rigid_point(matrix: list[float], point: list[float]) -> list[float]:
    """Map a point through the inverse of a rotation/translation matrix."""
    delta = [point[i] - matrix[12 + i] for i in range(3)]
    return [sum(matrix[col * 4 + row] * delta[row] for row in range(3)) for col in range(3)]


def inverse_rigid_matrix(matrix: list[float]) -> list[float]:
    """Invert a rigid column-major transform (rotation plus translation)."""
    result = IDENTITY.copy()
    for row in range(3):
        for col in range(3):
            result[col * 4 + row] = matrix[row * 4 + col]
    for row in range(3):
        result[12 + row] = -sum(result[col * 4 + row] * matrix[12 + col] for col in range(3))
    return result


def ecef_to_enu_matrix(origin: list[float]) -> list[float]:
    """Build a column-major ECEF-to-local-ENU matrix at an ECEF origin."""
    x, y, z = origin
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    # Convert geocentric latitude to WGS84 geodetic latitude.
    a, e2 = 6378137.0, 6.69437999014e-3
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(5):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)
    slon, clon = math.sin(lon), math.cos(lon)
    slat, clat = math.sin(lat), math.cos(lat)
    # ECEF to ENU rotation (rows are East, North, Up).
    rows = [
        [-slon, clon, 0.0],
        [-slat * clon, -slat * slon, clat],
        [clat * clon, clat * slon, slat],
    ]
    matrix = IDENTITY.copy()
    for row in range(3):
        for col in range(3):
            matrix[col * 4 + row] = rows[row][col]
        matrix[12 + row] = -sum(rows[row][col] * origin[col] for col in range(3))
    return matrix


@functools.lru_cache(maxsize=1)
def etrs89_dhhn2016_transformer():
    """Get a PROJ operation from geocentric WGS84 to ETRS89/DHHN2016."""
    try:
        from pyproj.transformer import Transformer
        import pyproj
    except ImportError as exc:
        raise RuntimeError("ETRS89/DHHN2016 requires pyproj/PROJ; install it with 'python -m pip install pyproj'") from exc
    # Allow PROJ to download the DHHN2016 geoid grid when it is not installed locally.
    pyproj.network.set_network_enabled(True)
    try:
        transformer = Transformer.from_crs(
            "EPSG:4978",
            "EPSG:25832+7837",
            always_xy=True,
            allow_ballpark=True,
        )
    except Exception as exc:
        raise RuntimeError(f"PROJ could not set up EPSG:25832+7837 transformation: {exc}") from exc
    return transformer


def ecef_to_etrs89_dhhn2016(point: list[float]) -> list[float]:
    try:
        return list(etrs89_dhhn2016_transformer().transform(*point, errcheck=True))
    except Exception as exc:
        raise RuntimeError(f"PROJ failed to transform ECEF coordinates to EPSG:25832+7837: {exc}") from exc


def ecef_to_etrs89_dhhn2016_matrix(origin: list[float]) -> list[float]:
    """Build the local derivative of the PROJ ECEF-to-ETRS89/DHHN2016 operation."""
    base = ecef_to_etrs89_dhhn2016(origin)
    delta = 1.0
    matrix = IDENTITY.copy()
    for col in range(3):
        sample = origin.copy()
        sample[col] += delta
        projected = ecef_to_etrs89_dhhn2016(sample)
        for row in range(3):
            matrix[col * 4 + row] = projected[row] - base[row]
    for row in range(3):
        matrix[12 + row] = base[row] - sum(matrix[col * 4 + row] * origin[col] for col in range(3))
    return matrix


def matrix_max_scale(matrix: list[float]) -> float:
    return max(math.sqrt(sum(matrix[col * 4 + row] ** 2 for row in range(3))) for col in range(3))


def tile_sphere(tile: dict[str, Any], parent_transform: list[float] | None = None
                ) -> tuple[list[float], float, list[float], list[float], float]:
    """Return world sphere, composed transform, and local sphere for a tile."""
    local = tile.get("transform", IDENTITY)
    # 3D Tiles matrices are column-major; compose parent * local.
    parent = parent_transform or IDENTITY
    world = multiply_matrices(parent, local)
    bv = tile.get("boundingVolume", {})
    box = bv.get("box")
    if box and len(box) >= 12:
        center = box[:3]
        axes = [box[3:6], box[6:9], box[9:12]]
        radius = sum(math.sqrt(sum(v * v for v in axis)) for axis in axes)
    elif "sphere" in bv and len(bv["sphere"]) >= 4:
        center = bv["sphere"][:3]
        radius = float(bv["sphere"][3])
    elif "region" in bv and len(bv["region"]) >= 6:
        west, south, east, north, low, high = bv["region"][:6]
        center = [6378137.0 * (west + east) / 2, 6378137.0 * (south + north) / 2, (low + high) / 2]
        radius = 6378137.0 * max(east - west, north - south) / 2 + (high - low) / 2
    else:
        center, radius = [0.0, 0.0, 0.0], 1.0
    x, y, z = center
    world_center = [world[0] * x + world[4] * y + world[8] * z + world[12],
                    world[1] * x + world[5] * y + world[9] * z + world[13],
                    world[2] * x + world[6] * y + world[10] * z + world[14]]
    scale = max(math.sqrt(sum(world[c * 4 + r] ** 2 for r in range(3))) for c in range(3))
    return world_center, radius * scale, world, center, radius


GLTF_COMPONENTS = {
    5120: ("b", 1, -128, 127), 5121: ("B", 1, 0, 255),
    5122: ("h", 2, -32768, 32767), 5123: ("H", 2, 0, 65535),
    5125: ("I", 4, 0, 4294967295), 5126: ("f", 4, None, None),
}
GLTF_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def gltf_accessor(doc: dict[str, Any], binary: bytes, accessor_index: int) -> list[tuple[float, ...]]:
    accessor = doc["accessors"][accessor_index]
    width = GLTF_TYPE_WIDTH[accessor["type"]]
    fmt_code, component_size, min_value, max_value = GLTF_COMPONENTS[accessor["componentType"]]
    count = accessor["count"]
    output = [tuple(0.0 for _ in range(width)) for _ in range(count)]
    if "bufferView" in accessor:
        view = doc["bufferViews"][accessor["bufferView"]]
        stride = view.get("byteStride", component_size * width)
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        fmt_string = "<" + fmt_code * width
        values = []
        for i in range(count):
            row = struct.unpack_from(fmt_string, binary, offset + i * stride)
            if accessor.get("normalized") and min_value is not None:
                if fmt_code.islower():
                    row = tuple(max(-1.0, v / max_value) for v in row)
                else:
                    row = tuple(v / max_value for v in row)
            values.append(tuple(float(v) for v in row))
        output = values
    sparse = accessor.get("sparse")
    if sparse:
        sparse_count = sparse["count"]
        indices_info, values_info = sparse["indices"], sparse["values"]
        index_fmt, index_size, _, _ = GLTF_COMPONENTS[indices_info["componentType"]]
        index_view = doc["bufferViews"][indices_info["bufferView"]]
        value_view = doc["bufferViews"][values_info["bufferView"]]
        index_offset = index_view.get("byteOffset", 0) + indices_info.get("byteOffset", 0)
        value_offset = value_view.get("byteOffset", 0) + values_info.get("byteOffset", 0)
        fmt_string = "<" + fmt_code * width
        for i in range(sparse_count):
            target = struct.unpack_from("<" + index_fmt, binary, index_offset + i * index_size)[0]
            row = struct.unpack_from(fmt_string, binary, value_offset + i * component_size * width)
            output[target] = tuple(float(v) for v in row)
    return output


def gltf_trs(node: dict[str, Any]) -> list[float]:
    if "matrix" in node:
        return [float(v) for v in node["matrix"]]
    tx, ty, tz = node.get("translation", (0.0, 0.0, 0.0))
    sx, sy, sz = node.get("scale", (1.0, 1.0, 1.0))
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rotation = [
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
        2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y,
    ]
    # Convert row-major rotation entries into glTF's column-major matrix and apply scale by column.
    result = IDENTITY.copy()
    for row in range(3):
        for col in range(3):
            result[col * 4 + row] = rotation[row * 3 + col] * (sx, sy, sz)[col]
    result[12:15] = [float(tx), float(ty), float(tz)]
    return result


def transform_point(matrix: list[float], point: tuple[float, ...]) -> tuple[float, float, float]:
    x, y, z = point[:3]
    return (matrix[0]*x + matrix[4]*y + matrix[8]*z + matrix[12],
            matrix[1]*x + matrix[5]*y + matrix[9]*z + matrix[13],
            matrix[2]*x + matrix[6]*y + matrix[10]*z + matrix[14])


def transform_normal(matrix: list[float], normal: tuple[float, ...]) -> tuple[float, float, float]:
    # Inverse-transpose of the linear 3x3, preserving normals under node scale.
    a,b,c,d,e,f,g,h,i = (matrix[0],matrix[4],matrix[8],matrix[1],matrix[5],matrix[9],matrix[2],matrix[6],matrix[10])
    det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    if abs(det) < 1e-20:
        return normal[:3]
    inv = ((e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det,
           (f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det,
           (d*h-e*g)/det,(b*g-a*h)/det,(a*e-b*d)/det)
    x,y,z = normal[:3]
    values = (inv[0]*x+inv[3]*y+inv[6]*z, inv[1]*x+inv[4]*y+inv[7]*z, inv[2]*x+inv[5]*y+inv[8]*z)
    length = math.sqrt(sum(v*v for v in values))
    return tuple(v/length for v in values) if length else values


def text_values(values: list[tuple[float, ...]], width: int = 6) -> list[str]:
    flat = [fmt(v) for row in values for v in row]
    return [" ".join(flat[i:i+width]) for i in range(0, len(flat), width)]


class Converter:
    def __init__(self, output: Path, root_url: str, timeout: float, coordinate_frame: str,
                 min_lod: int | None = None, convert_ive: bool = False):
        self.output = output
        self.root_url = root_url
        self.timeout = timeout
        self.coordinate_frame = coordinate_frame
        self.min_lod = min_lod
        self.convert_ive = convert_ive
        self.enu_transform: list[float] | None = None
        self.downloaded: set[str] = set()
        self.generated: set[Path] = set()
        self.tile_pages: set[Path] = set()
        self.errors = 0
        self.page_counter = 0
        self.progress_total = 0
        self.progress_done = 0
        self.progress_tty = sys.stdout.isatty()
        self.progress_line_length = 0
        self.scanned_tilesets = 0
        self.converted_glbs: dict[tuple[str, tuple[float, ...]], Path] = {}
        self.glb_transform_by_path: dict[Path, tuple[float, ...]] = {}

    def within_lod_limit(self, url: str) -> bool:
        """Include URL levels at or above the requested minimum (for example L5 includes L17..L5)."""
        if self.min_lod is None:
            return True
        match = re.search(r"(?:^|/)L(\d+)(?:/|$)", urllib.parse.urlsplit(url).path, re.IGNORECASE)
        return match is None or int(match.group(1)) >= self.min_lod

    def log(self, message: str) -> None:
        if self.progress_tty and self.progress_line_length:
            sys.stdout.write("\r" + " " * self.progress_line_length + "\r")
            self.progress_line_length = 0
        print(message, flush=True)

    def _directory_label(self, value: str | Path) -> str:
        path = value if isinstance(value, Path) else self.local_path(value, "tileset.json")
        try:
            label = path.parent.relative_to(self.output).as_posix()
        except ValueError:
            label = path.parent.as_posix()
        return label if label not in ("", ".") else "."

    def draw_progress(self, stage: str, done: int, total: int, directory: str | Path) -> None:
        total = max(1, total)
        done = min(done, total)
        fraction = done / total
        width = 24
        filled = int(width * fraction)
        bar = "#" * filled + "." * (width - filled)
        label = self._directory_label(directory)
        if len(label) > 48:
            label = "..." + label[-45:]
        line = f"{stage} [{bar}] {done}/{total} ({fraction:5.1%}) | {label}"
        if self.progress_tty:
            sys.stdout.write("\r" + line[:150].ljust(150))
            self.progress_line_length = min(150, len(line))
            sys.stdout.flush()
        else:
            print(line, flush=True)

    def finish_progress(self) -> None:
        if self.progress_tty and self.progress_line_length:
            sys.stdout.write(chr(10))
            sys.stdout.flush()
            self.progress_line_length = 0

    def draw_scan_status(self, tileset_url: str) -> None:
        self.scanned_tilesets += 1
        label = self._directory_label(tileset_url)
        line = f"Scanning tileset JSONs: {self.scanned_tilesets} visited | {label}"
        if self.progress_tty:
            sys.stdout.write("\r" + line[:150].ljust(150))
            self.progress_line_length = min(150, len(line))
            sys.stdout.flush()
        else:
            print(line, flush=True)

    def count_tiles_in_tileset(self, tileset_url: str, active: set[str] | None = None) -> int:
        active = active if active is not None else set()
        key = urllib.parse.urldefrag(tileset_url)[0]
        if key in active:
            return 0
        active.add(key)
        self.draw_scan_status(tileset_url)
        try:
            downloaded = self.get(tileset_url, "tileset.json")
            if downloaded is None:
                return 0
            try:
                tileset = json.loads(downloaded[0].decode("utf-8-sig"))
                root = tileset["root"]
            except (ValueError, KeyError, UnicodeDecodeError):
                return 0
            def count_tile(tile: dict[str, Any]) -> int:
                count = 1
                content = tile.get("content") or tile.get("contents") or []
                if isinstance(content, dict):
                    content = [content]
                if isinstance(content, list):
                    for item in content:
                        uri = item.get("uri") or item.get("url")
                        if not uri:
                            continue
                        nested_url = urllib.parse.urljoin(tileset_url, uri)
                        if not self.within_lod_limit(nested_url):
                            continue
                        if urllib.parse.urlsplit(nested_url).path.lower().endswith(".json"):
                            count += self.count_tiles_in_tileset(nested_url, active)
                for child in tile.get("children", []):
                    count += count_tile(child)
                return count
            return count_tile(root)
        finally:
            active.remove(key)

    def prepare_progress(self, tileset_url: str) -> None:
        self.log("Scanning tileset hierarchy to determine progress...")
        self.progress_total = max(1, self.count_tiles_in_tileset(tileset_url))
        self.progress_done = 0
        self.log(f"Found {self.progress_total} tile entries.")

    def local_path(self, url: str, fallback: str) -> Path:
        return self.output / safe_relative_path(url, self.root_url, fallback)

    def get(self, url: str, fallback: str) -> tuple[bytes, Path] | None:
        path = self.local_path(url, fallback)
        key = urllib.parse.urldefrag(url)[0]
        # Reuse files from earlier runs as well as files fetched during this run.
        # A zero-byte file is not considered a successful download.
        if path.is_file() and path.stat().st_size > 0:
            self.downloaded.add(key)
            try:
                return path.read_bytes(), path
            except OSError as exc:
                print(f"ERROR reading cached file {path}: {exc}", file=sys.stderr)
                self.errors += 1
                return None
        if key not in self.downloaded:
            try:
                data = fetch(url, self.timeout)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                self.downloaded.add(key)
                self.log(f"Downloaded {url}")
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                print(f"ERROR downloading {url}: {exc}", file=sys.stderr)
                self.errors += 1
                return None
        else:
            try:
                data = path.read_bytes()
            except OSError:
                return None
        return data, path

    def write_tile_page(self, path: Path, tile: dict[str, Any], child_pages: list[Path]) -> None:
        """Write one tile page: its geometry is inline and its direct refinements are paged."""
        def inline_osg(osg_file: Path, indent: str) -> list[str]:
            source_lines = osg_file.read_text(encoding="utf-8").splitlines()
            if len(source_lines) < 2:
                return []
            # The standalone geometry page is a root MatrixTransform. Inline its
            # node body and adjust texture paths to the new containing page.
            body = source_lines
            def adjust_texture(line: str) -> str:
                match = re.match(r'^(\s*file\s+")([^"]+)(".*)$', line)
                if not match:
                    return line
                source_texture = osg_file.parent / match.group(2).replace("/", os.sep)
                target_texture = os.path.relpath(source_texture, path.parent).replace("\\", "/")
                return match.group(1) + osg_escape(target_texture) + match.group(3)
            return [indent + adjust_texture(line) for line in body]

        own_paths = [p for p in tile["contents"] if p.is_file() and p.stat().st_size > 0]
        if own_paths and not child_pages:
            # A leaf is the finest available detail: load its geometry directly
            # instead of giving it a PagedLOD range that can expire at close range.
            if len(own_paths) == 1:
                lines = inline_osg(own_paths[0], "")
            else:
                lines = ["Group {", "  nodeMask 0xffffffff", f"  num_children {len(own_paths)}"]
                for own_path in own_paths:
                    lines.extend(inline_osg(own_path, "  "))
                lines.append("}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(chr(10).join(lines + [""]), encoding="utf-8")
            self.generated.add(path)
            self.tile_pages.add(path)
            return

        center, radius, error = tile["center"], tile["radius"], tile["error"]
        threshold = max(1.0, radius * 3.0)
        ranges = ([(threshold, 3.40282e38)] if own_paths else []) + [
            (0.0, threshold) for _ in child_pages]
        filenames = ([""] if own_paths else []) + [
            os.path.relpath(child_page, path.parent).replace("\\", "/") for child_page in child_pages]
        out = ["PagedLOD {", "  nodeMask 0xffffffff", "  cullingActive TRUE",
               "  Center " + " ".join(fmt(v) for v in center), f"  Radius {fmt(radius)}",
               "  RangeMode DISTANCE_FROM_EYE_POINT", f"  RangeList {len(ranges)} {{"]
        out.extend(f"    {fmt(near)} {fmt(far)}" for near, far in ranges)
        out += ["  }", "  DatabasePath .", "  NumChildrenThatCannotBeExpired 0",
                "  DisableExternalChildrenPaging 0", f"  FileNameList {len(filenames)} {{"]
        out.extend("    " + ('""' if filename == "" else osg_escape(filename))
                   for filename in filenames)
        out += ["  }", f"  num_children {1 if own_paths else 0}"]
        if own_paths:
            if len(own_paths) == 1:
                out.extend(inline_osg(own_paths[0], "  "))
            else:
                out += ["  Group {", "    nodeMask 0xffffffff", f"    num_children {len(own_paths)}"]
                for own_path in own_paths:
                    out.extend(inline_osg(own_path, "    "))
                out.append("  }")
        out += ["  Comment {", f"    Geometric Error : {fmt(error)}", "  }", "}", ""]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(chr(10).join(out), encoding="utf-8")
        self.generated.add(path)
        self.tile_pages.add(path)

    def write_json_pages(self, root: dict[str, Any]) -> Path:
        def page_path(tile: dict[str, Any], path_hint: Path | None = None) -> Path:
            if path_hint is None:
                contents = [p for p in tile["contents"] if isinstance(p, Path)]
                if contents:
                    source = contents[0]
                    rel = source.relative_to(self.output)
                    path_hint = (self.output / "osg" / rel).with_name(rel.stem + "_tile.osg")
                else:
                    parent = tile.get("parent_page_path", self.output / "osg" / "root")
                    path_hint = parent.with_name(parent.stem + f"_child_{tile.get('page_index', 0)}.osg")
            return path_hint

        all_tiles: list[dict[str, Any]] = []
        def collect(tile: dict[str, Any]) -> None:
            all_tiles.append(tile)
            for child in tile["children"]:
                collect(child)
        collect(root)
        self.progress_total = max(1, len(all_tiles))
        self.progress_done = 0
        def emit(tile: dict[str, Any], page: Path) -> Path:
            child_pages: list[Path] = []
            for index, child in enumerate(tile["children"]):
                child["parent_page_path"] = page
                child["page_index"] = index
                child_page = page_path(child)
                child_pages.append(emit(child, child_page))
            self.write_tile_page(page, tile, child_pages)
            self.progress_done += 1
            self.draw_progress("Writing tile pages", self.progress_done, self.progress_total, page)
            return page

        root_path = self.output / "osg" / "root_tile.osg"
        return emit(root, root_path)

    def write_tile_pages(self, root: dict[str, Any], root_path: Path) -> None:
        """Write one .osg file per tile/LOD directory, each with exactly one PagedLOD."""
        def count_nodes(tile: dict[str, Any]) -> int:
            return 1 + sum(count_nodes(child) for child in tile["children"])

        self.progress_total = max(1, count_nodes(root))
        self.progress_done = 0

        def emit(tile: dict[str, Any], path: Path) -> Path:
            child_paths = []
            for index, child in enumerate(tile["children"]):
                child_path = child.get("page_hint")
                if child_path is None:
                    if child["contents"]:
                        child_path = child["contents"][0][0].with_suffix(".osg")
                    else:
                        self.page_counter += 1
                        child_path = self.output / "pages" / f"tile_{self.page_counter:08d}.osg"
                    child["page_path"] = child_path
                child_paths.append(emit(child, child_path))
            self.write_tile_page(path, tile, child_paths)
            self.progress_done += 1
            self.draw_progress("Writing pages", self.progress_done, self.progress_total, path)
            tile["page_path"] = path
            return path

        root["page_path"] = root_path
        emit(root, root_path)

    def find_ecef_origin(self, tile: dict[str, Any]) -> list[float] | None:
        transform = tile.get("transform", IDENTITY)
        origin = [transform[12], transform[13], transform[14]]
        if math.sqrt(sum(v * v for v in origin)) > 1_000_000.0:
            return origin
        for child in tile.get("children", []):
            found = self.find_ecef_origin(child)
            if found is not None:
                return found
        return None

    def osg_path(self, source_path: Path) -> Path:
        try:
            rel = source_path.relative_to(self.output)
        except ValueError:
            rel = Path(source_path.name)
        return (self.output / "osg" / rel).with_suffix(".osg")

    def hierarchy_page_path(self, source_json: Path) -> Path:
        """Use one hierarchy page per source directory, mirroring its path under osg/."""
        try:
            rel_dir = source_json.parent.relative_to(self.output)
        except ValueError:
            rel_dir = Path()
        return self.output / "osg" / rel_dir / "lod.osg"

    def convert_glb(self, glb_path: Path, world_transform: list[float],
                    tile_center_ecef: list[float] | None = None) -> Path:
        output_transform = world_transform
        vertex_transform: list[float] | None = None
        vertex_ecef_transform: list[float] | None = None
        projected_center: list[float] | None = None
        if self.coordinate_frame == "etrs89_dhhn2016":
            if tile_center_ecef is None:
                raise ValueError("UTM 32N conversion requires the ECEF center of the tile")
            projected_center = ecef_to_etrs89_dhhn2016(tile_center_ecef)
            projection = ecef_to_etrs89_dhhn2016_matrix(tile_center_ecef)
            vertex_transform = multiply_matrices(projection, world_transform)
            # Keep a local derivative for normals; positions use PROJ per vertex.
            for axis in range(3):
                vertex_transform[12 + axis] -= projected_center[axis]
            vertex_ecef_transform = world_transform
            output_transform = translation_matrix(projected_center)
        transform_key = tuple(round(float(v), 9) for v in output_transform + (vertex_transform or [])
                              + (vertex_ecef_transform or []) + (projected_center or []))
        cache_key = (str(glb_path.resolve()), transform_key)
        if cache_key in self.converted_glbs:
            return self.converted_glbs[cache_key]
        osg_path = self.osg_path(glb_path)
        existing_transform = self.glb_transform_by_path.get(osg_path)
        if existing_transform is not None and existing_transform != transform_key:
            suffix = hashlib.sha1(repr(transform_key).encode("ascii")).hexdigest()[:8]
            osg_path = osg_path.with_name(osg_path.stem + "_" + suffix + ".osg")
        self.write_glb_osg(glb_path, osg_path, output_transform, vertex_transform,
                           vertex_ecef_transform, projected_center)
        self.converted_glbs[cache_key] = osg_path
        self.glb_transform_by_path[osg_path] = transform_key
        return osg_path

    def write_glb_osg(self, glb_path: Path, osg_path: Path, world_transform: list[float],
                      vertex_transform: list[float] | None = None,
                      vertex_ecef_transform: list[float] | None = None,
                      projected_center: list[float] | None = None) -> None:
        data = glb_path.read_bytes()
        if len(data) < 20 or data[:4] != b"glTF":
            raise ValueError(f"Invalid GLB header: {glb_path}")
        version, total_length = struct.unpack_from("<II", data, 4)
        if version != 2 or total_length > len(data):
            raise ValueError(f"Unsupported GLB version or length: {glb_path}")
        offset, document, binary = 12, None, b""
        while offset + 8 <= total_length:
            chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset:offset + chunk_length]
            offset += chunk_length
            if chunk_type == 0x4E4F534A:
                document = json.loads(chunk.decode("utf-8").rstrip("\x00 \t\r\n"))
            elif chunk_type == 0x004E4942:
                binary = chunk
        if document is None:
            raise ValueError(f"GLB has no JSON chunk: {glb_path}")
        if document.get("extensionsRequired") and "KHR_draco_mesh_compression" in document["extensionsRequired"]:
            raise ValueError(f"Draco-compressed GLB is not supported by the built-in converter: {glb_path}")

        rel = glb_path.relative_to(self.output)
        texture_dir = self.output / "osg" / rel.parent / "textures" / rel.stem
        image_paths: list[Path | None] = []
        for image_index, image in enumerate(document.get("images", [])):
            mime = image.get("mimeType", "")
            ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(mime)
            uri = image.get("uri")
            image_bytes = None
            if uri and uri.startswith("data:"):
                header, payload = uri.split(",", 1)
                if ";base64" in header:
                    image_bytes = base64.b64decode(payload)
                else:
                    image_bytes = urllib.parse.unquote_to_bytes(payload)
                if not ext:
                    ext = ".jpg" if "jpeg" in header else ".png"
            elif uri:
                image_source = glb_path.parent / urllib.parse.unquote(uri)
                try:
                    image_bytes = image_source.read_bytes()
                except OSError:
                    self.log(f"WARNING missing external GLB image: {image_source}")
                if not ext:
                    ext = image_source.suffix or ".bin"
            elif "bufferView" in image:
                view = document["bufferViews"][image["bufferView"]]
                start = view.get("byteOffset", 0)
                image_bytes = binary[start:start + view["byteLength"]]
                if not ext:
                    ext = ".jpg" if "jpeg" in mime else ".png"
            if image_bytes is None:
                image_paths.append(None)
                continue
            texture_dir.mkdir(parents=True, exist_ok=True)
            image_path = texture_dir / f"{rel.stem}_{image_index}{ext or '.bin'}"
            is_jpeg = bool(ext and ext.lower() in (".jpg", ".jpeg"))
            if is_jpeg:
                marker_path = image_path.with_name(image_path.name + ".mirrored.sha256")
                source_hash = hashlib.sha256(image_bytes).hexdigest()
                try:
                    current_marker = marker_path.read_text(encoding="ascii").strip()
                except OSError:
                    current_marker = ""
                if not image_path.is_file() or image_path.stat().st_size == 0 or current_marker != source_hash:
                    image_path.write_bytes(mirror_jpeg_vertically(image_bytes))
                    marker_path.write_text(source_hash + "\n", encoding="ascii")
            elif not image_path.is_file() or image_path.stat().st_size == 0:
                image_path.write_bytes(image_bytes)
            image_paths.append(image_path)

        def image_for_material(material_index: int | None) -> tuple[Path | None, dict[str, Any]]:
            if material_index is None or material_index >= len(document.get("materials", [])):
                return None, {}
            material = document["materials"][material_index]
            pbr = material.get("pbrMetallicRoughness", {})
            texture_info = pbr.get("baseColorTexture", {})
            texture_index = texture_info.get("index")
            if texture_index is None or texture_index >= len(document.get("textures", [])):
                return None, material
            texture = document["textures"][texture_index]
            source = texture.get("source")
            if source is None:
                source = texture.get("extensions", {}).get("KHR_texture_basisu", {}).get("source")
            return (image_paths[source] if source is not None and source < len(image_paths) else None), material

        lines: list[str] = []
        def emit_primitive(mesh_name: str, primitive_index: int, primitive: dict[str, Any], node_matrix: list[float]) -> None:
            nonlocal lines
            attributes = primitive.get("attributes", {})
            position_index = attributes.get("POSITION")
            if position_index is None:
                return
            positions = []
            for point in gltf_accessor(document, binary, position_index):
                local_point = transform_point(node_matrix, point)
                if vertex_ecef_transform is not None and projected_center is not None:
                    projected = ecef_to_etrs89_dhhn2016(
                        transform_point(vertex_ecef_transform, local_point))
                    positions.append(tuple(projected[i] - projected_center[i] for i in range(3)))
                elif vertex_transform is not None:
                    positions.append(transform_point(vertex_transform, local_point))
                else:
                    positions.append(local_point)
            normals = []
            if "NORMAL" in attributes:
                normals = [transform_normal(vertex_transform, transform_normal(node_matrix, n))
                           if vertex_transform is not None else transform_normal(node_matrix, n)
                           for n in gltf_accessor(document, binary, attributes["NORMAL"])]
            texcoords = gltf_accessor(document, binary, attributes["TEXCOORD_0"]) if "TEXCOORD_0" in attributes else []
            colors = gltf_accessor(document, binary, attributes["COLOR_0"]) if "COLOR_0" in attributes else []
            if colors and len(colors[0]) == 3:
                colors = [tuple(c) + (1.0,) for c in colors]
            if "indices" in primitive:
                index_rows = gltf_accessor(document, binary, primitive["indices"])
                indices = [int(row[0]) for row in index_rows]
            else:
                indices = list(range(len(positions)))
            mode = primitive.get("mode", 4)
            primitive_types = {0: "POINTS", 1: "LINES", 2: "LINE_LOOP", 3: "LINE_STRIP",
                               4: "TRIANGLES", 5: "TRIANGLE_STRIP", 6: "TRIANGLE_FAN"}
            if mode not in primitive_types:
                return
            texture_path, material = image_for_material(primitive.get("material"))
            state_lines: list[str] = []
            if material:
                pbr = material.get("pbrMetallicRoughness", {})
                rgba = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
                alpha = float(rgba[3]) if len(rgba) > 3 else 1.0
                double_sided = material.get("doubleSided", False)
                state_lines = ["  StateSet {", "    rendering_hint DEFAULT_BIN", "    renderBinMode INHERIT",
                               f"    GL_CULL_FACE {'OFF' if double_sided else 'ON'}",
                               f"    GL_LIGHTING {'ON' if normals else 'OFF'}",
                               "    osg::Material {", "      diffuse FRONT_AND_BACK " + " ".join(fmt(float(v)) for v in rgba[:4]),
                               "      ambient FRONT_AND_BACK 0.2 0.2 0.2 1", "      specular FRONT_AND_BACK 0.04 0.04 0.04 1",
                               "      shininess FRONT_AND_BACK 16", "    }"]
                if alpha < 1.0 or material.get("alphaMode") == "BLEND":
                    state_lines += ["    GL_BLEND ON", "    rendering_hint TRANSPARENT_BIN"]
                if texture_path:
                    texture_rel = os.path.relpath(texture_path, osg_path.parent).replace("\\", "/")
                    sampler = document.get("samplers", [])
                    texture_info = pbr.get("baseColorTexture", {})
                    texture_index = texture_info.get("index", 0)
                    sampler_index = document.get("textures", [{}])[texture_index].get("sampler", 0) if document.get("textures") else 0
                    sampler_data = sampler[sampler_index] if sampler_index < len(sampler) else {}
                    wrap = {10497: "REPEAT", 33071: "CLAMP_TO_EDGE", 33648: "MIRRORED_REPEAT"}
                    filters = {9728: "NEAREST", 9729: "LINEAR", 9984: "NEAREST_MIPMAP_NEAREST",
                               9985: "LINEAR_MIPMAP_NEAREST", 9986: "NEAREST_MIPMAP_LINEAR", 9987: "LINEAR_MIPMAP_LINEAR"}
                    state_lines += ["    textureUnit 0 {", "      GL_TEXTURE_2D ON", "      osg::Texture2D {",
                                    "        file " + '"' + osg_escape(texture_rel) + '"',
                                    "        wrap_s " + wrap.get(sampler_data.get("wrapS", 10497), "REPEAT"),
                                    "        wrap_t " + wrap.get(sampler_data.get("wrapT", 10497), "REPEAT"),
                                    "        min_filter " + filters.get(sampler_data.get("minFilter", 9987), "LINEAR_MIPMAP_LINEAR"),
                                    "        mag_filter " + filters.get(sampler_data.get("magFilter", 9729), "LINEAR"),
                                    "      }", "    }"]
                state_lines.append("  }")
            indices_per_set = len(indices)
            primitive_lines = ["    PrimitiveSets 1", "    {", f"      DrawElementsUInt {primitive_types[mode]} {indices_per_set}", "      {"]
            primitive_lines.extend("        " + " ".join(str(v) for v in indices[i:i+18]) for i in range(0, len(indices), 18))
            primitive_lines += ["      }", "    }"]
            lines.extend(["Geode {", "  name " + '"' + osg_escape(f"{mesh_name}_primitive_{primitive_index}") + '"',
                          "  nodeMask 0xffffffff", "  cullingActive TRUE", *state_lines, "  num_drawables 1",
                          "  osg::Geometry {", "    DataVariance STATIC", "    useDisplayList FALSE", "    useVertexBufferObjects TRUE",
                          f"    VertexArray Vec3Array {len(positions)}", "    {",
                          *["      " + row for row in text_values(positions, 6)], "    }"])
            if normals:
                lines += [f"    NormalArray Vec3Array {len(normals)}", "    {", *["      " + row for row in text_values(normals, 6)], "    }", "    NormalBinding PER_VERTEX"]
            if texcoords:
                lines += [f"    TexCoordArray 0 Vec2Array {len(texcoords)}", "    {", *["      " + row for row in text_values(texcoords, 6)], "    }"]
            if colors:
                lines += [f"    ColorArray Vec4Array {len(colors)}", "    {", *["      " + row for row in text_values(colors, 4)], "    }", "    ColorBinding PER_VERTEX"]
            lines += primitive_lines + ["  }", "}"]

        meshes = document.get("meshes", [])
        nodes = document.get("nodes", [])
        def visit(node_index: int, parent: list[float]) -> None:
            node = nodes[node_index]
            current = multiply_matrices(parent, gltf_trs(node))
            if "mesh" in node and node["mesh"] < len(meshes):
                mesh = meshes[node["mesh"]]
                mesh_name = node.get("name") or mesh.get("name") or f"mesh_{node['mesh']}"
                for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
                    emit_primitive(mesh_name, primitive_index, primitive, current)
            for child_index in node.get("children", []):
                visit(child_index, current)
        scenes = document.get("scenes", [])
        if nodes:
            if scenes:
                scene_index = document.get("scene", 0)
                roots = scenes[scene_index].get("nodes", []) if scene_index < len(scenes) else []
            else:
                referenced = {child for node in nodes for child in node.get("children", [])}
                roots = [i for i in range(len(nodes)) if i not in referenced]
            for node_index in roots:
                visit(node_index, IDENTITY)
        else:
            for mesh_index, mesh in enumerate(meshes):
                mesh_name = mesh.get("name") or f"mesh_{mesh_index}"
                for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
                    emit_primitive(mesh_name, primitive_index, primitive, IDENTITY)
        rows = [[world_transform[row * 4 + col] for col in range(4)] for row in range(4)]
        geode_count = sum(1 for line in lines if line == "Geode {")
        output = ["MatrixTransform {", "  nodeMask 0xffffffff", "  cullingActive TRUE", "  Matrix {",
                  *["    " + " ".join(fmt(v) for v in row) for row in rows], "  }",
                  f"  num_children {geode_count}"]
        output.extend("  " + line if line else "" for line in lines)
        output += ["}", ""]
        osg_path.parent.mkdir(parents=True, exist_ok=True)
        osg_path.write_text(chr(10).join(output), encoding="utf-8")
        self.generated.add(osg_path)

    def content_file(self, content_url: str) -> tuple[Path, list[float]] | None:
        downloaded = self.get(content_url, "content.bin")
        if downloaded is None:
            return None
        data, original_path = downloaded
        if data[:4] == b"b3dm":
            rtc_center = b3dm_rtc_center(data)
            glb_path = original_path.with_suffix(".glb")
            if glb_path.is_file() and glb_path.stat().st_size > 0:
                return glb_path, rtc_center
            extracted = decode_b3dm(data)
            if extracted is None:
                print(f"ERROR invalid/unsupported B3DM: {content_url}", file=sys.stderr)
                self.errors += 1
                return None
            glb, rtc_center = extracted
            glb_path.write_bytes(glb)
            self.log(f"Extracted {glb_path}")
            return glb_path, rtc_center
        if data[:4] == b"glTF":
            if original_path.suffix.lower() != ".glb":
                glb_path = original_path.with_suffix(".glb")
                if glb_path.is_file() and glb_path.stat().st_size > 0:
                    return glb_path, [0.0, 0.0, 0.0]
                glb_path.write_bytes(data)
                return glb_path, [0.0, 0.0, 0.0]
            return original_path, [0.0, 0.0, 0.0]
        print(f"WARNING unsupported tile content (kept as downloaded): {content_url}", file=sys.stderr)
        return None

    def read_tileset_node(self, tileset_url: str, parent_transform: list[float] | None = None) -> dict[str, Any] | None:
        downloaded = self.get(tileset_url, "tileset.json")
        if downloaded is None:
            return None
        data, json_path = downloaded
        try:
            tileset = json.loads(data.decode("utf-8-sig"))
            root = tileset["root"]
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            print(f"ERROR invalid tileset JSON {tileset_url}: {exc}", file=sys.stderr)
            self.errors += 1
            return None
        # 3D Tiles 1.0 defaults embedded glTF content to Y-up. A tileset may
        # explicitly declare Z or X; preserve that declaration per tileset.
        up_axis = str(tileset.get("asset", {}).get("gltfUpAxis", "Y")).upper()
        if up_axis not in GLTF_UP_TO_Z_UP:
            print(f"WARNING unsupported gltfUpAxis {up_axis!r}; using Z", file=sys.stderr)
            up_axis = "Z"
        node = self.process_tile(root, tileset_url, parent_transform, up_axis)
        if node is not None:
            node["json_path"] = json_path
        return node

    def process_tileset(self, tileset_url: str, top: bool = False,
                        parent_transform: list[float] | None = None) -> Path | None:
        self.prepare_progress(tileset_url)
        root = self.read_tileset_node(tileset_url, parent_transform)
        self.finish_progress()
        if root is None:
            return None
        self.progress_done = 0
        self.log("Writing OSG pages...")
        root_page = self.write_json_pages(root)
        root_path = self.output / "osg" / "root.osg"
        if root_page != root_path:
            root_path.parent.mkdir(parents=True, exist_ok=True)
            root_path.write_text(root_page.read_text(encoding="utf-8"), encoding="utf-8")
            self.generated.add(root_path)
            self.tile_pages.add(root_path)
        self.finish_progress()
        if self.convert_ive:
            ive_root_created = self.convert_pages_to_ive(root_path)
            ive_root = root_path.with_suffix(".ive")
            if ive_root_created:
                return ive_root
        return root_path

    def convert_pages_to_ive(self, root_path: Path) -> bool:
        """Convert generated OSG files with osgconv and rewrite page references."""
        osgconv = shutil.which("osgconv")
        if osgconv is None:
            raise RuntimeError("--ive requires osgconv to be installed and available on PATH")
        pages = {page.resolve() for page in self.generated
                 if page.suffix.lower() == ".osg" and page.is_file()}
        pages.add(root_path.resolve())
        osg_root = self.output / "osg"
        if osg_root.is_dir():
            pages.update(path.resolve() for path in osg_root.rglob("*.osg") if path.is_file())
        ordered_pages: list[Path] = []
        visited: set[Path] = set()
        reference_pattern = re.compile(r'^\s*(?:"([^"]+\.osg)"|([^{}\s]+\.osg))\s*$', re.MULTILINE)

        def page_references(text: str) -> list[str]:
            return [match.group(1) or match.group(2)
                    for match in reference_pattern.finditer(text)]

        def visit_page(page: Path) -> None:
            if page in visited:
                return
            visited.add(page)
            try:
                text = page.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                text = ""
            for reference in page_references(text):
                child = (page.parent / reference).resolve()
                if child in pages:
                    visit_page(child)
            ordered_pages.append(page)

        for page in pages:
            visit_page(page)
        self.log("Converting OSG hierarchy pages to IVE...")
        self.progress_total = max(1, len(ordered_pages))
        self.progress_done = 0
        root_created = False
        converted_pages: set[Path] = set()
        for page in ordered_pages:
            target = page.with_suffix(".ive")
            temporary_path: Path | None = None
            try:
                text = page.read_text(encoding="utf-8")
                def replace_reference(match: re.Match[str]) -> str:
                    reference = match.group(1) or match.group(2)
                    child = (page.parent / reference).resolve()
                    if child not in converted_pages:
                        return match.group(0)
                    converted_reference = reference[:-4] + ".ive"
                    if match.group(1) is not None:
                        return f'"{converted_reference}"'
                    return "    " + converted_reference
                text = reference_pattern.sub(replace_reference, text)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", suffix=".osg",
                    prefix=".ive_convert_", dir=page.parent, delete=False,
                ) as temporary:
                    temporary.write(text)
                    temporary_path = Path(temporary.name)
                result = subprocess.run([osgconv, "--compressed", str(temporary_path), str(target)],
                                        capture_output=True, text=True, check=False)
                if result.returncode != 0 or not target.is_file():
                    detail = (result.stderr or result.stdout).strip()
                    if not detail:
                        detail = "osgconv returned success but did not write the output file"
                    print(f"ERROR osgconv failed for {page}: {detail}", file=sys.stderr)
                    self.errors += 1
                else:
                    converted_pages.add(page)
                    self.generated.add(target)
                    if page == root_path:
                        root_created = True
            except (OSError, UnicodeError) as exc:
                print(f"ERROR converting {page} to IVE: {exc}", file=sys.stderr)
                self.errors += 1
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            self.progress_done += 1
            self.draw_progress("Converting to IVE", self.progress_done, self.progress_total, target)
        for page in converted_pages:
            try:
                page.unlink()
                self.generated.discard(page)
                self.tile_pages.discard(page)
            except OSError as exc:
                print(f"ERROR removing converted OSG file {page}: {exc}", file=sys.stderr)
                self.errors += 1
        return root_created

    def process_tile(self, tile: dict[str, Any], tileset_url: str,
                     parent_transform: list[float] | None = None,
                     gltf_up_axis: str = "Y") -> dict[str, Any] | None:
        center, radius, world_transform, _local_center, _local_radius = tile_sphere(tile, parent_transform)
        tile_center_ecef = center.copy()
        if self.coordinate_frame == "etrs89_dhhn2016":
            projection = ecef_to_etrs89_dhhn2016_matrix(center)
            center = ecef_to_etrs89_dhhn2016(center)
            radius *= matrix_max_scale(projection)
        progress_url = tileset_url
        result: dict[str, Any] = {
            "center": center,
            "radius": radius,
            "error": float(tile.get("geometricError", 0.0) or 0.0),
            "contents": [],
            "children": [],
            "page_hint": None,
        }
        content = tile.get("content") or tile.get("contents") or []
        if isinstance(content, dict):
            content = [content]
        if isinstance(content, list) and content:
            first_uri = content[0].get("uri") or content[0].get("url")
            if first_uri:
                progress_url = urllib.parse.urljoin(tileset_url, first_uri)
        self.draw_progress("Processing tiles", self.progress_done, self.progress_total, progress_url)
        if isinstance(content, list):
            for item in content:
                uri = item.get("uri") or item.get("url")
                if not uri:
                    continue
                content_url = urllib.parse.urljoin(tileset_url, uri)
                if not self.within_lod_limit(content_url):
                    continue
                if urllib.parse.urlsplit(content_url).path.lower().endswith(".json"):
                    nested = self.read_tileset_node(content_url, world_transform)
                    if nested is not None:
                        result["children"].append(nested)
                    continue
                extracted = self.content_file(content_url)
                if extracted:
                    glb_path, rtc_center = extracted
                    # 3D Tiles order: glTF node hierarchy (baked into vertices),
                    # glTF-up to tileset Z-up, RTC_CENTER, then the accumulated
                    # parent/child tile transforms into the global coordinates.
                    local_content_transform = multiply_matrices(
                        translation_matrix(rtc_center), GLTF_UP_TO_Z_UP[gltf_up_axis])
                    full_transform = multiply_matrices(world_transform, local_content_transform)
                    try:
                        osg_path = self.convert_glb(glb_path, full_transform, tile_center_ecef)
                        result["contents"].append(osg_path)
                        if result["page_hint"] is None:
                            result["page_hint"] = osg_path
                    except (ValueError, OSError, KeyError, IndexError, RuntimeError, struct.error) as exc:
                        print(f"ERROR converting GLB {glb_path}: {exc}", file=sys.stderr)
                        self.errors += 1
        for child in tile.get("children", []):
            child_node = self.process_tile(child, tileset_url, world_transform, gltf_up_axis)
            if child_node is not None:
                result["children"].append(child_node)
        self.progress_done += 1
        self.draw_progress("Processing tiles", self.progress_done, self.progress_total, progress_url)
        if not result["contents"] and not result["children"]:
            return None
        return result


def parse_lod_level(value: str) -> int:
    match = re.fullmatch(r"L?(\d+)", value.strip(), re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError("LOD must be a level such as L5 or 5")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="URL of the root tileset.json")
    parser.add_argument("--output", "-o", type=Path, default=Path("tiles3d_download"), help="output directory")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--min-lod", type=parse_lod_level, metavar="LEVEL",
                        help="lowest LOD directory to process (for example L5 processes L17 through L5)")
    parser.add_argument("--coordinate-frame", choices=("ecef", "etrs89_dhhn2016"), default="ecef",
                        help="OSG coordinates: geocentric ECEF (default) or ETRS89 / UTM 32N + DHHN2016 (EPSG:25832+7837)")
    parser.add_argument("--ive", action="store_true",
                        help="also convert hierarchy pages to .ive with osgconv (must be on PATH)")
    args = parser.parse_args()
    if not args.url.lower().split("?", 1)[0].endswith(".json"):
        parser.error("URL must point to a tileset JSON file")
    converter = Converter(args.output.resolve(), args.url, args.timeout, args.coordinate_frame,
                          args.min_lod, args.ive)
    try:
        result = converter.process_tileset(args.url, top=True)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result is None:
        print("No OSG output was generated.", file=sys.stderr)
        return 1
    print(f"Root entry point: {result}")
    root_lod = result
    if root_lod.is_file():
        print(f"Lowest-LOD startup page: {root_lod}")
    print(f"Downloaded {len(converter.downloaded)} files; generated {len(converter.generated)} OSG files; errors: {converter.errors}")
    return 1 if converter.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
