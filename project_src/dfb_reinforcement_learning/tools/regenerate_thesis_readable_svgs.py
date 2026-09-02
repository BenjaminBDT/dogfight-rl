from __future__ import annotations

import copy
from pathlib import Path
from xml.etree import ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def _svg_tag(local_name: str) -> str:
    return f"{{{SVG_NS}}}{local_name}"


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _polyline_bounds(points: str) -> tuple[float, float, float, float] | None:
    coords: list[tuple[float, float]] = []
    for token in points.split():
        if "," not in token:
            continue
        x_raw, y_raw = token.split(",", 1)
        coords.append((float(x_raw), float(y_raw)))
    if not coords:
        return None
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _element_bounds(element: ET.Element) -> tuple[float, float, float, float] | None:
    tag = element.tag
    if tag == _svg_tag("rect"):
        x = _parse_float(element.get("x"))
        y = _parse_float(element.get("y"))
        width = _parse_float(element.get("width"))
        height = _parse_float(element.get("height"))
        return x, y, x + width, y + height
    if tag == _svg_tag("line"):
        x1 = _parse_float(element.get("x1"))
        y1 = _parse_float(element.get("y1"))
        x2 = _parse_float(element.get("x2"))
        y2 = _parse_float(element.get("y2"))
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if tag == _svg_tag("text"):
        x = _parse_float(element.get("x"))
        y = _parse_float(element.get("y"))
        size = _parse_float(element.get("font-size"), 14.0)
        return x, y - size, x + 600.0, y + size * 0.35
    if tag == _svg_tag("polyline"):
        points = element.get("points")
        if points is None:
            return None
        return _polyline_bounds(points)
    return None


def _clone_group(elements: list[ET.Element], *, translate_x: float, translate_y: float, scale_x: float, scale_y: float) -> ET.Element:
    group = ET.Element(_svg_tag("g"))
    group.set("transform", f"translate({translate_x:.2f},{translate_y:.2f}) scale({scale_x:.4f},{scale_y:.4f})")
    for element in elements:
        group.append(copy.deepcopy(element))
    return group


def _make_text(x: float, y: float, text: str, *, font_family: str, size: int, anchor: str = "middle", weight: str = "normal") -> ET.Element:
    element = ET.Element(
        _svg_tag("text"),
        {
            "x": f"{x:.1f}",
            "y": f"{y:.1f}",
            "font-family": font_family,
            "font-size": str(size),
            "text-anchor": anchor,
            "font-weight": weight,
        },
    )
    element.text = text
    return element


def _group_bounds(elements: list[ET.Element]) -> tuple[float, float, float, float]:
    bounds = [element_bounds for element in elements if (element_bounds := _element_bounds(element)) is not None]
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)
    return min_x, min_y, max_x, max_y


def _load_svg(path: Path) -> tuple[ET.ElementTree, ET.Element, list[ET.Element]]:
    tree = ET.parse(path)
    root = tree.getroot()
    children = list(root)
    return tree, root, children


def _text_content(element: ET.Element) -> str:
    return "".join(element.itertext()).strip()


def _split_by_panel_titles(elements: list[ET.Element], panel_title_prefixes: list[str]) -> list[list[ET.Element]]:
    groups: list[list[ET.Element]] = []
    current_group: list[ET.Element] = []
    title_index = 0
    for element in elements:
        text = _text_content(element) if element.tag == _svg_tag("text") else ""
        if title_index < len(panel_title_prefixes) and text.startswith(panel_title_prefixes[title_index]):
            if current_group:
                groups.append(current_group)
                current_group = []
            title_index += 1
        current_group.append(element)
    if current_group:
        groups.append(current_group)
    return groups


def regenerate_fig4(input_path: Path, output_path: Path) -> None:
    _, _, children = _load_svg(input_path)
    font_family = children[1].get("font-family", "sans-serif")
    title = "典型 reward diagnostics 时序图"
    subtitle = children[2].text or ""

    figure_body = children[3:]
    panel_groups = _split_by_panel_titles(figure_body, ["A.", "B.", "C.", "D."])
    panel_targets = [
        (70.0, 130.0),
        (70.0, 610.0),
        (70.0, 1090.0),
        (70.0, 1570.0),
    ]
    new_width = 1450
    new_height = 2080
    scale_x = 1.00
    scale_y = 1.42

    new_root = ET.Element(
        _svg_tag("svg"),
        {
            "width": str(new_width),
            "height": str(new_height),
            "viewBox": f"0 0 {new_width} {new_height}",
        },
    )
    new_root.append(ET.Element(_svg_tag("rect"), {"width": "100%", "height": "100%", "fill": "white"}))
    new_root.append(_make_text(new_width / 2, 46, title, font_family=font_family, size=30, weight="bold"))
    new_root.append(_make_text(new_width / 2, 82, subtitle, font_family=font_family, size=18))

    for elements, (target_x, target_y) in zip(panel_groups, panel_targets, strict=True):
        min_x, min_y, _, _ = _group_bounds(elements)
        translate_x = target_x - min_x * scale_x
        translate_y = target_y - min_y * scale_y
        new_root.append(
            _clone_group(
                elements,
                translate_x=translate_x,
                translate_y=translate_y,
                scale_x=scale_x,
                scale_y=scale_y,
            )
        )

    ET.ElementTree(new_root).write(output_path, encoding="utf-8", xml_declaration=False)


def regenerate_fig5(input_path: Path, output_path: Path) -> None:
    _, _, children = _load_svg(input_path)
    font_family = children[1].get("font-family", "sans-serif")
    title = children[1].text or "图 5-1"
    subtitle = children[2].text or ""

    figure_body = children[3:]
    panel_groups = _split_by_panel_titles(figure_body, ["A.", "B.", "C."])
    panel_targets = [
        (80.0, 150.0),
        (80.0, 860.0),
        (80.0, 1570.0),
    ]
    new_width = 1180
    new_height = 2300
    scale_x = 1.30
    scale_y = 1.30

    new_root = ET.Element(
        _svg_tag("svg"),
        {
            "width": str(new_width),
            "height": str(new_height),
            "viewBox": f"0 0 {new_width} {new_height}",
        },
    )
    new_root.append(ET.Element(_svg_tag("rect"), {"width": "100%", "height": "100%", "fill": "white"}))
    new_root.append(_make_text(new_width / 2, 50, title, font_family=font_family, size=32, weight="bold"))
    new_root.append(_make_text(new_width / 2, 88, subtitle, font_family=font_family, size=18))

    for elements, (target_x, target_y) in zip(panel_groups, panel_targets, strict=True):
        min_x, min_y, _, _ = _group_bounds(elements)
        translate_x = target_x - min_x * scale_x
        translate_y = target_y - min_y * scale_y
        new_root.append(
            _clone_group(
                elements,
                translate_x=translate_x,
                translate_y=translate_y,
                scale_x=scale_x,
                scale_y=scale_y,
            )
        )

    ET.ElementTree(new_root).write(output_path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    asset_root = repo_root / "docs" / "thesis" / "assets"
    regenerate_fig4(
        asset_root / "part3_fig4_2_reward_diagnostics.svg",
        asset_root / "part3_fig4_1_reward_diagnostics_readable.svg",
    )
    regenerate_fig5(
        asset_root / "part3_fig5_1_specialist_triptych.svg",
        asset_root / "part3_fig5_1_specialist_triptych_readable.svg",
    )


if __name__ == "__main__":
    main()
