import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np


"""
Density-guided preprocessing — V2: bottom-cut only.

Identical to density_processing.py except the top midcut is removed.
Rationale: above-sandhangan (wulu, cecak, layar, wignyan) are not
ink-connected to the nglegena body in printed Aksara Jawa, so the top
midcut provides no isolation benefit while potentially severing the
ascenders of tall base characters (ja, ba, la, ya).

The bottom midcut is kept because pasangan and suku can genuinely
connect to the base body from below.
"""


def process_image_density_guided(
    input_path,
    output_dir,
    threshold=160,
    kernel_size=3,
    min_area=15,
    main_zone_ratio=0.55,
    valley_ratio=0.22,
    anchor_overlap_ratio=0.35,
    min_anchor_area_ratio=0.65,
    main_merge_gap=6,
    attachment_horizontal_gap_ratio=0.60,
    attachment_vertical_gap_ratio=0.75,
    overlap_padding=5,
):
    try:
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")
    except Exception as exc:
        print(f"Error reading file: {exc}")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    height, width = image.shape[:2]
    timestamp = int(time.time())
    base_filename = (
        f"densityv2_{os.path.splitext(os.path.basename(input_path))[0]}_{timestamp}"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        gray, threshold, 255, cv2.THRESH_BINARY_INV
    )

    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(1, kernel_size)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    zone = estimate_main_zone(
        closed_image,
        main_zone_ratio=main_zone_ratio,
        valley_ratio=valley_ratio,
    )

    # ── V2: bottom midcut only ──
    # The top midcut is intentionally omitted. Above-sandhangan are not
    # ink-connected to the base character in printed text, so severing the
    # top produces no benefit and may cut tall ascenders.
    midcut_bottom = (zone["main_bottom"] + zone["cut_bottom"]) // 2
    zone["midcut_bottom"] = int(midcut_bottom)

    # ── Two-pass CCA ──
    # Pass 1: CCA on the uncut image to discover small components
    # (sandhangan like pangkon) that span the bottom midcut line.
    num_labels_pre, labels_pre, stats_pre, _ = cv2.connectedComponentsWithStats(
        closed_image, connectivity=8
    )

    protection_mask = np.zeros_like(closed_image)
    max_area = 0.95 * (width * height)

    areas_all = [
        int(stats_pre[i, cv2.CC_STAT_AREA])
        for i in range(1, num_labels_pre)
        if min_area <= int(stats_pre[i, cv2.CC_STAT_AREA]) <= max_area
    ]
    median_area_pre = float(np.median(areas_all)) if areas_all else 1.0

    for label_id in range(1, num_labels_pre):
        area = int(stats_pre[label_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp_y = int(stats_pre[label_id, cv2.CC_STAT_TOP])
        comp_h = int(stats_pre[label_id, cv2.CC_STAT_HEIGHT])
        comp_y2 = comp_y + comp_h
        crosses_bottom = comp_y < midcut_bottom < comp_y2
        if not crosses_bottom:
            continue
        if area < median_area_pre * min_anchor_area_ratio:
            protection_mask[labels_pre == label_id] = 255

    # Pass 2: apply bottom midcut with protection
    cca_image = closed_image.copy()
    unprotected = protection_mask[midcut_bottom, :] == 0
    cca_image[midcut_bottom, unprotected] = 0

    (
        num_labels,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(cca_image, connectivity=8)

    components = build_components(
        stats=stats,
        centroids=centroids,
        min_area=min_area,
        max_area=0.95 * (width * height),
        main_top=zone["main_top"],
        main_bottom=zone["main_bottom"],
        cut_top=zone["cut_top"],
        cut_bottom=zone["cut_bottom"],
    )

    for comp in components:
        mask = labels == comp["id"]
        pre_labels_in_mask = labels_pre[mask]
        foreground = pre_labels_in_mask[pre_labels_in_mask > 0]
        if len(foreground) > 0:
            values, counts = np.unique(foreground, return_counts=True)
            comp["pre_parent"] = int(values[np.argmax(counts)])
        else:
            comp["pre_parent"] = 0

    cut_top = zone["cut_top"]
    cut_bottom = zone["cut_bottom"]
    for comp in components:
        pp = comp.get("pre_parent", 0)
        if pp <= 0 or pp >= num_labels_pre:
            continue
        pre_y = int(stats_pre[pp, cv2.CC_STAT_TOP])
        pre_h = int(stats_pre[pp, cv2.CC_STAT_HEIGHT])
        pre_y2 = pre_y + pre_h
        pre_area = int(stats_pre[pp, cv2.CC_STAT_AREA])
        extends_above = pre_y < cut_top
        extends_below = pre_y2 > cut_bottom
        if extends_above and extends_below:
            comp["force_secondary"] = True

    if not components:
        return _save_empty_result(
            image=image,
            binary_image=closed_image,
            output_dir=output_dir,
            base_filename=base_filename,
            params={
                "threshold": threshold,
                "kernel_size": kernel_size,
                "min_area": min_area,
                "main_zone_ratio": main_zone_ratio,
                "valley_ratio": valley_ratio,
                "anchor_overlap_ratio": anchor_overlap_ratio,
                "min_anchor_area_ratio": min_anchor_area_ratio,
                "main_merge_gap": main_merge_gap,
            },
            zone=zone,
            width=width,
        )

    components = _absorb_tiny_fragments(components, zone)

    labeled_img_color = np.zeros((height, width, 3), dtype=np.uint8)
    for component in components:
        label_id = component["id"]
        labeled_img_color[labels == label_id] = [
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 100),
        ]

    components_with_boxes = labeled_img_color.copy()
    for component in components:
        x, y, w, h = component["x"], component["y"], component["w"], component["h"]
        cv2.rectangle(components_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)

    anchor_components, secondary_components = classify_anchor_components(
        components=components,
        anchor_overlap_ratio=anchor_overlap_ratio,
        min_anchor_area_ratio=min_anchor_area_ratio,
    )

    groups = density_guided_grouping(
        anchor_components=anchor_components,
        secondary_components=secondary_components,
        main_merge_gap=main_merge_gap,
        attachment_horizontal_gap_ratio=attachment_horizontal_gap_ratio,
        attachment_vertical_gap_ratio=attachment_vertical_gap_ratio,
    )

    zone_guidance = render_zone_guidance(image, zone)
    density_profile = render_density_profile(zone, width)
    classification_img = render_component_classification(
        image=image,
        anchor_components=anchor_components,
        secondary_components=secondary_components,
        zone=zone,
    )
    image_with_final_boxes = render_final_groups(image, groups)

    def save_and_get_path(suffix, img_data):
        try:
            full_path = os.path.join(output_dir, f"{base_filename}_{suffix}.jpg")
            cv2.imwrite(full_path, img_data)
            return os.path.join(
                "static", "results", f"{base_filename}_{suffix}.jpg"
            ).replace("\\", "/")
        except Exception as exc:
            print(f"Error saving {suffix}: {exc}")
            return None

    animation_steps = {
        "original": save_and_get_path("step_original", image),
        "binary": save_and_get_path("step_binary", closed_image),
        "density_profile": save_and_get_path("step_density_profile", density_profile),
        "zone_guidance": save_and_get_path("step_zone_guidance", zone_guidance),
        "components_with_boxes": save_and_get_path(
            "step_components_with_boxes", components_with_boxes
        ),
        "classification": save_and_get_path(
            "step_classification", classification_img
        ),
        "final_boxes": save_and_get_path("step_final_boxes", image_with_final_boxes),
    }

    cropped_results = []
    for index, group in enumerate(groups):
        anchor_x1, anchor_y1, anchor_x2, anchor_y2 = group_bounds(group["anchors"])

        bx1 = max(0, anchor_x1 - overlap_padding)
        by1 = max(0, anchor_y1 - overlap_padding)
        bx2 = min(width, anchor_x2 + overlap_padding)
        by2 = min(height, anchor_y2 + overlap_padding)
        base_crop = image[by1:by2, bx1:bx2]
        base_path = save_and_get_path(f"char_{index}_base", base_crop) if base_crop.size > 0 else None

        secondaries = [
            c for c in group["components"]
            if c not in group["anchors"]
        ]

        above_marks = [c for c in secondaries if c["cy"] < anchor_y1]
        below_marks = [c for c in secondaries if c["cy"] >= anchor_y2]
        inline_marks = [
            c for c in secondaries
            if anchor_y1 <= c["cy"] < anchor_y2
        ]

        for m in inline_marks:
            above_overlap = any(
                overlap_length(m["x"], m["x2"], a["x"], a["x2"]) > 0
                and interval_gap(m["y"], m["y2"], a["y"], a["y2"]) <= 3
                for a in above_marks
            )
            below_overlap = any(
                overlap_length(m["x"], m["x2"], b["x"], b["x2"]) > 0
                and interval_gap(m["y"], m["y2"], b["y"], b["y2"]) <= 3
                for b in below_marks
            )
            if below_overlap and not above_overlap:
                below_marks.append(m)
            elif above_overlap and not below_overlap:
                above_marks.append(m)
            else:
                dist_top = abs(m["cy"] - anchor_y1)
                dist_bot = abs(m["cy"] - anchor_y2)
                if dist_bot <= dist_top:
                    below_marks.append(m)
                else:
                    above_marks.append(m)

        beside_marks = []
        if below_marks and any(c["cx"] > anchor_x2 or c["cx"] < anchor_x1 for c in below_marks):
            beside_marks = below_marks
            below_marks = []

        wrapped_marks = []
        all_mark_pools = [
            ("above", above_marks),
            ("below", below_marks),
            ("beside", beside_marks),
        ]

        parent_to_pools = defaultdict(set)
        for pool_name, pool in all_mark_pools:
            for c in pool:
                pp = c.get("pre_parent", -1)
                if pp > 0:
                    parent_to_pools[pp].add(pool_name)
        shared_parents = {pp for pp, pools in parent_to_pools.items() if len(pools) >= 2}

        if shared_parents:
            new_above = []
            new_below = []
            new_beside = []
            for c in above_marks:
                if c.get("pre_parent", -1) in shared_parents:
                    wrapped_marks.append(c)
                else:
                    new_above.append(c)
            for c in below_marks:
                if c.get("pre_parent", -1) in shared_parents:
                    wrapped_marks.append(c)
                else:
                    new_below.append(c)
            for c in beside_marks:
                if c.get("pre_parent", -1) in shared_parents:
                    wrapped_marks.append(c)
                else:
                    new_beside.append(c)
            above_marks = new_above
            below_marks = new_below
            beside_marks = new_beside

        above_path, above_box = _composite_marks(image, above_marks, overlap_padding,
                                      height, width, save_and_get_path,
                                      f"char_{index}_above")
        below_path, below_box = _composite_marks(image, below_marks, overlap_padding,
                                      height, width, save_and_get_path,
                                      f"char_{index}_below")
        beside_path, beside_box = _composite_marks(image, beside_marks, overlap_padding,
                                       height, width, save_and_get_path,
                                       f"char_{index}_beside", ref_y1=by1)
        wrapped_path, wrapped_box = _composite_marks(image, wrapped_marks, overlap_padding,
                                        height, width, save_and_get_path,
                                        f"char_{index}_wrapped")

        gx1, gy1, gx2, gy2 = group_bounds(group["components"])
        gy1 = max(0, gy1 - overlap_padding)
        gx1 = max(0, gx1 - overlap_padding)
        gy2 = min(height, gy2 + overlap_padding)
        gx2 = min(width, gx2 + overlap_padding)
        full_crop = image[gy1:gy2, gx1:gx2]
        full_path = save_and_get_path(f"char_{index}", full_crop) if full_crop.size > 0 else None

        base_box = (int(bx1), int(by1), int(bx2), int(by2)) if base_path else None

        cropped_results.append({
            "full": full_path,
            "base": base_path,
            "above": above_path,
            "below": below_path,
            "beside": beside_path,
            "wrapped": wrapped_path,
            "box": (int(gx1), int(gy1), int(gx2), int(gy2)),
            "boxes": {
                "base": base_box,
                "above": above_box,
                "below": below_box,
                "beside": beside_box,
                "wrapped": wrapped_box,
            }
        })

    if any(value is None for value in animation_steps.values()):
        return {}

    return {
        "steps": animation_steps,
        "cropped_results": cropped_results,
        "params": {
            "threshold": threshold,
            "kernel_size": kernel_size,
            "min_area": min_area,
            "main_zone_ratio": main_zone_ratio,
            "valley_ratio": valley_ratio,
            "anchor_overlap_ratio": anchor_overlap_ratio,
            "min_anchor_area_ratio": min_anchor_area_ratio,
            "main_merge_gap": main_merge_gap,
            "attachment_horizontal_gap_ratio": attachment_horizontal_gap_ratio,
            "attachment_vertical_gap_ratio": attachment_vertical_gap_ratio,
        },
        "debug": {
            "main_top": zone["main_top"],
            "main_bottom": zone["main_bottom"],
            "cut_top": zone["cut_top"],
            "cut_bottom": zone["cut_bottom"],
            "midcut_bottom": zone["midcut_bottom"],
            "peak_row": zone["peak_row"],
            "peak_value": float(zone["peak_value"]),
            "core_threshold": float(zone["core_threshold"]),
            "valley_threshold": float(zone["valley_threshold"]),
            "total_components": len(components),
            "anchor_components": len(anchor_components),
            "secondary_components": len(secondary_components),
            "groups_formed": len(groups),
        },
    }


def estimate_main_zone(binary_image, main_zone_ratio=0.55, valley_ratio=0.22):
    """Estimate the main body rows and broader cut lines from row density."""
    height, width = binary_image.shape
    row_density = np.count_nonzero(binary_image, axis=1).astype(np.float32)
    normalized_density = row_density / max(width, 1)

    smooth_window = max(5, int(round(height * 0.05)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window > height:
        smooth_window = height if height % 2 == 1 else max(1, height - 1)
    if smooth_window <= 1:
        smoothed_density = normalized_density.copy()
    else:
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        smoothed_density = np.convolve(normalized_density, kernel, mode="same")

    peak_row = int(np.argmax(smoothed_density))
    peak_value = float(smoothed_density[peak_row])
    background_level = float(np.percentile(smoothed_density, 20))
    core_threshold = background_level + (peak_value - background_level) * main_zone_ratio
    valley_threshold = background_level + (peak_value - background_level) * valley_ratio

    main_top = peak_row
    while main_top > 0 and smoothed_density[main_top - 1] >= core_threshold:
        main_top -= 1

    main_bottom = peak_row
    while main_bottom < height - 1 and smoothed_density[main_bottom + 1] >= core_threshold:
        main_bottom += 1

    cut_top = _find_cut_line(
        profile=smoothed_density,
        start_index=main_top,
        direction=-1,
        threshold=valley_threshold,
    )
    cut_bottom = _find_cut_line(
        profile=smoothed_density,
        start_index=main_bottom,
        direction=1,
        threshold=valley_threshold,
    )

    cut_top = max(0, cut_top - 1)
    cut_bottom = min(height - 1, cut_bottom + 1)

    return {
        "row_density": normalized_density,
        "smoothed_density": smoothed_density,
        "peak_row": peak_row,
        "peak_value": peak_value,
        "background_level": background_level,
        "core_threshold": core_threshold,
        "valley_threshold": valley_threshold,
        "main_top": int(main_top),
        "main_bottom": int(main_bottom),
        "cut_top": int(min(cut_top, main_top)),
        "cut_bottom": int(max(cut_bottom, main_bottom)),
    }


def _find_cut_line(profile, start_index, direction, threshold):
    """Find the nearest low-density valley outside the main zone."""
    length = len(profile)
    max_search = max(4, int(round(length * 0.35)))
    current_index = start_index
    steps = 0

    while 0 <= current_index + direction < length and steps < max_search:
        current_index += direction
        steps += 1

        if profile[current_index] <= threshold:
            valley_index = current_index
            while (
                0 <= valley_index + direction < length
                and profile[valley_index + direction] <= profile[valley_index]
            ):
                valley_index += direction
            return valley_index

    if direction < 0:
        segment = profile[max(0, start_index - max_search) : start_index + 1]
        local_index = int(np.argmin(segment))
        return max(0, start_index - max_search) + local_index

    segment = profile[start_index : min(length, start_index + max_search + 1)]
    local_index = int(np.argmin(segment))
    return start_index + local_index


def build_components(
    stats,
    centroids,
    min_area,
    max_area,
    main_top,
    main_bottom,
    cut_top,
    cut_bottom,
):
    """Build component dictionaries with zone-overlap metadata."""
    components = []
    for label_id in range(1, len(stats)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        x2 = x + w
        y2 = y + h
        cx = float(centroids[label_id][0])
        cy = float(centroids[label_id][1])

        main_overlap = overlap_length(y, y2, main_top, main_bottom + 1)
        cut_overlap = overlap_length(y, y2, cut_top, cut_bottom + 1)

        components.append(
            {
                "id": label_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "area": area,
                "main_overlap_ratio": main_overlap / max(h, 1),
                "cut_overlap_ratio": cut_overlap / max(h, 1),
                "center_in_main": main_top <= cy <= main_bottom,
                "center_in_cut": cut_top <= cy <= cut_bottom,
            }
        )

    components.sort(key=lambda component: component["x"])
    return components


def _absorb_tiny_fragments(components, zone, fragment_ratio=0.12):
    """Absorb very small components into their nearest neighbor."""
    if len(components) < 2:
        return components

    areas = [c["area"] for c in components]
    median_area = float(np.median(areas))
    area_floor = median_area * fragment_ratio

    widths = [c["w"] for c in components if c["area"] >= area_floor]
    median_width = float(np.median(widths)) if widths else 50.0
    max_absorption_dist = median_width * 1.5

    fragments = [c for c in components if c["area"] < area_floor]
    keepers = [c for c in components if c["area"] >= area_floor]

    if not fragments or not keepers:
        return components

    for frag in fragments:
        best_idx = None
        best_dist = float("inf")
        for i, k in enumerate(keepers):
            h_gap = interval_gap(frag["x"], frag["x2"], k["x"], k["x2"])
            v_gap = interval_gap(frag["y"], frag["y2"], k["y"], k["y2"])
            dist = (h_gap ** 2 + v_gap ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None or best_dist > max_absorption_dist:
            continue

        k = keepers[best_idx]
        new_x = min(k["x"], frag["x"])
        new_y = min(k["y"], frag["y"])
        new_x2 = max(k["x2"], frag["x2"])
        new_y2 = max(k["y2"], frag["y2"])
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y
        new_area = k["area"] + frag["area"]
        new_cx = (k["cx"] * k["area"] + frag["cx"] * frag["area"]) / new_area
        new_cy = (k["cy"] * k["area"] + frag["cy"] * frag["area"]) / new_area

        main_top = zone["main_top"]
        main_bottom = zone["main_bottom"]
        cut_top = zone["cut_top"]
        cut_bottom = zone["cut_bottom"]
        main_overlap = overlap_length(new_y, new_y2, main_top, main_bottom + 1)
        cut_overlap = overlap_length(new_y, new_y2, cut_top, cut_bottom + 1)

        keepers[best_idx] = {
            "id": k["id"],
            "x": new_x, "y": new_y, "w": new_w, "h": new_h,
            "x2": new_x2, "y2": new_y2,
            "cx": float(new_cx), "cy": float(new_cy),
            "area": new_area,
            "main_overlap_ratio": main_overlap / max(new_h, 1),
            "cut_overlap_ratio": cut_overlap / max(new_h, 1),
            "center_in_main": main_top <= new_cy <= main_bottom,
            "center_in_cut": cut_top <= new_cy <= cut_bottom,
            "pre_parent": k.get("pre_parent", frag.get("pre_parent", 0)),
            "force_secondary": k.get("force_secondary", False) or frag.get("force_secondary", False),
        }

    keepers.sort(key=lambda c: c["x"])
    return keepers


def _merge_overlapping_secondary(components, zone, max_gap=3):
    """Iteratively merge secondary components that overlap or are very close."""
    if len(components) < 2:
        return components

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(components):
            j = i + 1
            while j < len(components):
                a = components[i]
                b = components[j]
                h_ol = overlap_length(a["x"], a["x2"], b["x"], b["x2"])
                v_gap = interval_gap(a["y"], a["y2"], b["y"], b["y2"])
                if h_ol > 0 and v_gap <= max_gap:
                    mx = min(a["x"], b["x"])
                    my = min(a["y"], b["y"])
                    mx2 = max(a["x2"], b["x2"])
                    my2 = max(a["y2"], b["y2"])
                    mw = mx2 - mx
                    mh = my2 - my
                    total_area = a["area"] + b["area"]
                    mcx = (a["cx"] * a["area"] + b["cx"] * b["area"]) / total_area
                    mcy = (a["cy"] * a["area"] + b["cy"] * b["area"]) / total_area

                    main_overlap = overlap_length(
                        my, my2, zone["main_top"], zone["main_bottom"] + 1
                    )
                    cut_overlap = overlap_length(
                        my, my2, zone["cut_top"], zone["cut_bottom"] + 1
                    )

                    components[i] = {
                        "id": a["id"],
                        "x": mx, "y": my, "w": mw, "h": mh,
                        "x2": mx2, "y2": my2,
                        "cx": float(mcx), "cy": float(mcy),
                        "area": total_area,
                        "main_overlap_ratio": main_overlap / max(mh, 1),
                        "cut_overlap_ratio": cut_overlap / max(mh, 1),
                        "center_in_main": zone["main_top"] <= mcy <= zone["main_bottom"],
                        "center_in_cut": zone["cut_top"] <= mcy <= zone["cut_bottom"],
                    }
                    components.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    components.sort(key=lambda c: c["x"])
    return components


def _build_overlap_clusters(components):
    """Group components whose bounding boxes overlap on the x-axis."""
    if not components:
        return []

    max_gap = 5
    median_w = float(np.median([c["w"] for c in components]))
    max_combined_width = median_w * 2.1

    sorted_comps = sorted(components, key=lambda c: c["x"])
    clusters = [[sorted_comps[0]]]

    for comp in sorted_comps[1:]:
        merged = False
        for cluster in clusters:
            cluster_x1 = min(c["x"] for c in cluster)
            cluster_x2 = max(c["x2"] for c in cluster)
            h_overlap = overlap_length(comp["x"], comp["x2"], cluster_x1, cluster_x2)
            h_gap = interval_gap(comp["x"], comp["x2"], cluster_x1, cluster_x2)

            min_width = min(comp["w"], cluster_x2 - cluster_x1)

            if min_width > 0 and h_overlap / min_width >= 0.40:
                cluster.append(comp)
                merged = True
                break

            combined_w = max(comp["x2"], cluster_x2) - min(comp["x"], cluster_x1)

            if combined_w <= max_combined_width and (h_gap <= max_gap or h_overlap > 0):
                cluster.append(comp)
                merged = True
                break

        if not merged:
            clusters.append([comp])

    return clusters


def _cluster_metrics(cluster):
    """Compute combined bounding box metrics for a cluster of components."""
    total_area = sum(c["area"] for c in cluster)
    x1 = min(c["x"] for c in cluster)
    x2 = max(c["x2"] for c in cluster)
    combined_width = x2 - x1
    return total_area, combined_width


def classify_anchor_components(
    components, anchor_overlap_ratio=0.35, min_anchor_area_ratio=0.35
):
    """Split components into base anchors and secondary marks."""
    if not components:
        return [], []

    heights = [component["h"] for component in components]
    median_height = float(np.median(heights))

    forced_secondary = [c for c in components if c.get("force_secondary")]
    normal_components = [c for c in components if not c.get("force_secondary")]

    zone_components = [
        c for c in normal_components if c["main_overlap_ratio"] >= 0.30
    ]
    if not zone_components:
        zone_components = normal_components

    ref_area = float(np.median([c["area"] for c in zone_components]))
    ref_width = float(np.median([c["w"] for c in zone_components]))

    min_area_threshold = ref_area * min_anchor_area_ratio
    min_width_threshold = ref_width * min_anchor_area_ratio

    cluster_area_floor = ref_area * 0.20
    cluster_width_floor = ref_width * 0.20

    inside_zone = [
        c for c in normal_components
        if c["center_in_cut"] or c["cut_overlap_ratio"] >= 0.50
    ]
    outside_zone = [
        c for c in normal_components
        if not (c["center_in_cut"] or c["cut_overlap_ratio"] >= 0.50)
    ]

    cluster_eligible = [
        c for c in inside_zone
        if c["area"] >= cluster_area_floor or c["w"] >= cluster_width_floor
    ]
    inside_zone_secondary = [
        c for c in inside_zone
        if not (c["area"] >= cluster_area_floor or c["w"] >= cluster_width_floor)
    ]

    clusters = _build_overlap_clusters(cluster_eligible)
    for c in outside_zone:
        clusters.append([c])
    for c in inside_zone_secondary:
        clusters.append([c])

    anchor_components = []
    secondary_components = []

    for cluster in clusters:
        cluster_area, cluster_width = _cluster_metrics(cluster)

        size_ok = (
            cluster_area >= min_area_threshold
            or cluster_width >= min_width_threshold
        )

        zone_ok = any(
            c["main_overlap_ratio"] >= anchor_overlap_ratio
            or c["center_in_main"]
            or (
                c["h"] >= median_height * 0.85
                and c["cut_overlap_ratio"] >= 0.55
            )
            for c in cluster
        )

        if size_ok and zone_ok:
            if len(cluster) == 1:
                anchor_components.append(cluster[0])
            else:
                mx = min(c["x"] for c in cluster)
                my = min(c["y"] for c in cluster)
                mx2 = max(c["x2"] for c in cluster)
                my2 = max(c["y2"] for c in cluster)
                mw = mx2 - mx
                mh = my2 - my
                total_area = sum(c["area"] for c in cluster)
                mcx = sum(c["cx"] * c["area"] for c in cluster) / total_area
                mcy = sum(c["cy"] * c["area"] for c in cluster) / total_area

                best_main = max(c["main_overlap_ratio"] for c in cluster)
                best_cut = max(c["cut_overlap_ratio"] for c in cluster)

                anchor_components.append({
                    "id": cluster[0]["id"],
                    "x": mx, "y": my, "w": mw, "h": mh,
                    "x2": mx2, "y2": my2,
                    "cx": float(mcx), "cy": float(mcy),
                    "area": total_area,
                    "main_overlap_ratio": best_main,
                    "cut_overlap_ratio": best_cut,
                    "center_in_main": any(c["center_in_main"] for c in cluster),
                    "center_in_cut": any(c["center_in_cut"] for c in cluster),
                })
        else:
            secondary_components.extend(cluster)

    if not anchor_components:
        fallback = sorted(
            components,
            key=lambda component: (
                component["area"],
                component["main_overlap_ratio"],
                component["cut_overlap_ratio"],
            ),
            reverse=True,
        )
        anchor_components = fallback[: max(1, min(3, len(fallback)))]
        secondary_ids = {component["id"] for component in anchor_components}
        secondary_components = [
            component
            for component in components
            if component["id"] not in secondary_ids
        ]

    secondary_components.extend(forced_secondary)

    anchor_components.sort(key=lambda component: component["x"])
    secondary_components.sort(key=lambda component: component["x"])

    return anchor_components, secondary_components


def density_guided_grouping(
    anchor_components,
    secondary_components,
    main_merge_gap=6,
    attachment_horizontal_gap_ratio=0.60,
    attachment_vertical_gap_ratio=0.75,
):
    """Overlap-based grouping: merge anchors, then attach secondaries."""
    if not anchor_components:
        return []

    median_anchor_width = float(np.median([c["w"] for c in anchor_components]))
    median_anchor_height = float(np.median([c["h"] for c in anchor_components]))
    max_group_width = max(median_anchor_width * 1.8, median_anchor_width + main_merge_gap)

    groups = [{"anchors": [anchor_components[0]], "components": [anchor_components[0]]}]

    for comp in anchor_components[1:]:
        cur = groups[-1]
        cur_x1, _, cur_x2, _ = group_bounds(cur["anchors"])
        gap = comp["x"] - cur_x2
        combined_width = comp["x2"] - cur_x1

        if gap <= main_merge_gap and combined_width <= max_group_width:
            cur["anchors"].append(comp)
            cur["components"].append(comp)
        else:
            groups.append({"anchors": [comp], "components": [comp]})

    max_promote_gap = max(5, int(median_anchor_width * 0.15))
    max_promote_combined_w = median_anchor_width * 2.1
    promoted = set()
    for i, sec in enumerate(secondary_components):
        if not sec["center_in_main"]:
            continue
        best_g = -1
        best_gap = float("inf")
        for g_idx, group in enumerate(groups):
            g_x1, _, g_x2, _ = group_bounds(group["anchors"])
            hgap = interval_gap(sec["x"], sec["x2"], g_x1, g_x2)
            combined_w = max(sec["x2"], g_x2) - min(sec["x"], g_x1)
            if hgap <= max_promote_gap and combined_w <= max_promote_combined_w and hgap < best_gap:
                best_gap = hgap
                best_g = g_idx
        if best_g >= 0:
            groups[best_g]["anchors"].append(sec)
            groups[best_g]["components"].append(sec)
            promoted.add(i)

    remaining_secondaries = [s for i, s in enumerate(secondary_components) if i not in promoted]

    max_vertical_gap = max(2.0, median_anchor_height * attachment_vertical_gap_ratio)

    ungrouped = []
    for sec in remaining_secondaries:
        sec_x1, sec_x2 = sec["x"], sec["x2"]
        sec_y1, sec_y2 = sec["y"], sec["y2"]

        best_idx = -1
        best_score = -float("inf")

        for g_idx, group in enumerate(groups):
            g_x1, g_y1, g_x2, g_y2 = group_bounds(group["anchors"])

            v_gap = interval_gap(sec_y1, sec_y2, g_y1, g_y2)
            if v_gap > max_vertical_gap:
                continue

            h_overlap = overlap_length(sec_x1, sec_x2, g_x1, g_x2)
            if h_overlap > 0:
                g_cx = (g_x1 + g_x2) / 2.0
                center_dist = abs(sec["cx"] - g_cx)
                score = h_overlap - center_dist
                if score > best_score:
                    best_score = score
                    best_idx = g_idx

        if best_idx >= 0:
            groups[best_idx]["components"].append(sec)
        else:
            ungrouped.append(sec)

    max_gap_pass2 = 3
    changed = True
    while changed:
        changed = False
        still_ungrouped = []
        for sec in ungrouped:
            sec_x1, sec_x2 = sec["x"], sec["x2"]
            sec_y1, sec_y2 = sec["y"], sec["y2"]
            attached = False

            for g_idx, group in enumerate(groups):
                for comp in group["components"]:
                    h_ol = overlap_length(sec_x1, sec_x2, comp["x"], comp["x2"])
                    v_gap = interval_gap(sec_y1, sec_y2, comp["y"], comp["y2"])
                    if h_ol > 0 and v_gap <= max_gap_pass2:
                        groups[g_idx]["components"].append(sec)
                        attached = True
                        changed = True
                        break
                if attached:
                    break

            if not attached:
                still_ungrouped.append(sec)
        ungrouped = still_ungrouped

    if ungrouped:
        used = [False] * len(ungrouped)
        for i in range(len(ungrouped)):
            if used[i]:
                continue
            cluster = [ungrouped[i]]
            used[i] = True
            expanded = True
            while expanded:
                expanded = False
                for j in range(len(ungrouped)):
                    if used[j]:
                        continue
                    for member in cluster:
                        h_ol = overlap_length(
                            ungrouped[j]["x"], ungrouped[j]["x2"],
                            member["x"], member["x2"],
                        )
                        v_gap = interval_gap(
                            ungrouped[j]["y"], ungrouped[j]["y2"],
                            member["y"], member["y2"],
                        )
                        if h_ol > 0 and v_gap <= max_gap_pass2:
                            cluster.append(ungrouped[j])
                            used[j] = True
                            expanded = True
                            break
            groups.append({"anchors": cluster, "components": list(cluster)})

    for group in groups:
        group["anchors"].sort(key=lambda c: c["x"])
        group["components"].sort(key=lambda c: c["x"])

    groups.sort(key=lambda g: min(c["x"] for c in g["components"]))
    return groups


def render_zone_guidance(image, zone):
    """Overlay the detected core zone and cut lines on the source image."""
    overlay = image.copy()
    height, width = overlay.shape[:2]

    shaded = overlay.copy()
    cv2.rectangle(
        shaded,
        (0, zone["cut_top"]),
        (width - 1, zone["cut_bottom"]),
        (200, 220, 255),
        -1,
    )
    cv2.rectangle(
        shaded,
        (0, zone["main_top"]),
        (width - 1, zone["main_bottom"]),
        (190, 255, 190),
        -1,
    )
    cv2.addWeighted(shaded, 0.25, overlay, 0.75, 0, overlay)

    cv2.line(overlay, (0, zone["cut_top"]), (width - 1, zone["cut_top"]), (0, 0, 255), 1)
    cv2.line(overlay, (0, zone["cut_bottom"]), (width - 1, zone["cut_bottom"]), (0, 0, 255), 1)
    cv2.line(overlay, (0, zone["main_top"]), (width - 1, zone["main_top"]), (0, 180, 0), 1)
    cv2.line(overlay, (0, zone["main_bottom"]), (width - 1, zone["main_bottom"]), (0, 180, 0), 1)
    cv2.line(overlay, (0, zone["peak_row"]), (width - 1, zone["peak_row"]), (255, 0, 0), 1)

    # midcut_top intentionally absent in V2 — no dashed line drawn above
    if "midcut_bottom" in zone:
        for seg_x in range(0, width - 1, 8):
            cv2.line(
                overlay,
                (seg_x, zone["midcut_bottom"]),
                (min(seg_x + 4, width - 1), zone["midcut_bottom"]),
                (0, 140, 255),
                1,
            )

    return overlay


def render_density_profile(zone, image_width):
    """Render a compact density profile image using rows as the vertical axis."""
    smoothed = zone["smoothed_density"]
    raw = zone["row_density"]
    height = len(smoothed)
    canvas_width = 360
    canvas = np.full((height, canvas_width, 3), 255, dtype=np.uint8)

    max_value = float(max(np.max(smoothed), np.max(raw), 1e-6))
    plot_width = canvas_width - 70
    x_offset = 45

    for row in range(height - 1):
        raw_x1 = x_offset + int((raw[row] / max_value) * plot_width)
        raw_x2 = x_offset + int((raw[row + 1] / max_value) * plot_width)
        cv2.line(canvas, (raw_x1, row), (raw_x2, row + 1), (200, 200, 200), 1)

        smooth_x1 = x_offset + int((smoothed[row] / max_value) * plot_width)
        smooth_x2 = x_offset + int((smoothed[row + 1] / max_value) * plot_width)
        cv2.line(canvas, (smooth_x1, row), (smooth_x2, row + 1), (0, 120, 255), 1)

    for marker, color in (
        ("cut_top", (0, 0, 255)),
        ("main_top", (0, 180, 0)),
        ("peak_row", (255, 0, 0)),
        ("main_bottom", (0, 180, 0)),
        ("cut_bottom", (0, 0, 255)),
    ):
        row = int(zone[marker])
        cv2.line(canvas, (0, row), (canvas_width - 1, row), color, 1)

    cv2.putText(
        canvas,
        f"width={image_width}px",
        (10, min(height - 6, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 80, 80),
        1,
    )

    return canvas


def render_component_classification(
    image,
    anchor_components,
    secondary_components,
    zone,
):
    """Visualize anchors, secondary marks, and the detected main band."""
    output = render_zone_guidance(image, zone)

    for component in anchor_components:
        cv2.rectangle(
            output,
            (component["x"], component["y"]),
            (component["x2"], component["y2"]),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            output,
            "A",
            (component["x"], max(10, component["y"] - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )

    for component in secondary_components:
        cv2.rectangle(
            output,
            (component["x"], component["y"]),
            (component["x2"], component["y2"]),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            output,
            "S",
            (component["x"], max(10, component["y"] - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
        )

    return output


def render_final_groups(image, groups):
    """Draw final grouped bounding boxes on the image."""
    output = image.copy()

    for group in groups:
        x1, y1, x2, y2 = group_bounds(group["components"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output


def group_bounds(components):
    """Return x1, y1, x2, y2 bounds for a list of components."""
    x1 = min(component["x"] for component in components)
    y1 = min(component["y"] for component in components)
    x2 = max(component["x2"] for component in components)
    y2 = max(component["y2"] for component in components)
    return x1, y1, x2, y2


def _composite_marks(image, marks, padding, img_h, img_w, save_fn, suffix, ref_y1=None):
    """Create a masked composite of multiple marks on a background-colored canvas."""
    if not marks:
        return None, None

    cx1 = max(0, min(m["x"] for m in marks) - padding)
    cy1 = max(0, min(m["y"] for m in marks) - padding)
    if ref_y1 is not None:
        cy1 = min(cy1, ref_y1)

    cx2 = min(img_w, max(m["x2"] for m in marks) + padding)
    cy2 = min(img_h, max(m["y2"] for m in marks) + padding)
    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw <= 0 or ch <= 0:
        return None, None

    if len(image.shape) == 3:
        bg = tuple(int(v) for v in np.median(image.reshape(-1, 3), axis=0))
        canvas = np.full((ch, cw, 3), bg, dtype=np.uint8)
    else:
        bg = int(np.median(image))
        canvas = np.full((ch, cw), bg, dtype=np.uint8)

    for m in marks:
        mx1 = max(0, m["x"] - padding)
        my1 = max(0, m["y"] - padding)
        mx2 = min(img_w, m["x2"] + padding)
        my2 = min(img_h, m["y2"] + padding)

        src = image[my1:my2, mx1:mx2]
        dx = mx1 - cx1
        dy = my1 - cy1
        dh, dw = src.shape[:2]
        canvas[dy:dy + dh, dx:dx + dw] = src

    return save_fn(suffix, canvas), (int(cx1), int(cy1), int(cx2), int(cy2))


def overlap_length(a1, a2, b1, b2):
    """Length of the intersection between two 1D intervals."""
    return max(0, min(a2, b2) - max(a1, b1))


def interval_gap(a1, a2, b1, b2):
    """Distance between two 1D intervals, zero when they overlap."""
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0


def _save_empty_result(image, binary_image, output_dir, base_filename, params, zone, width):
    """Return a valid response structure when no usable components exist."""

    def save_and_get_path(suffix, img_data):
        try:
            full_path = os.path.join(output_dir, f"{base_filename}_{suffix}.jpg")
            cv2.imwrite(full_path, img_data)
            return os.path.join(
                "static", "results", f"{base_filename}_{suffix}.jpg"
            ).replace("\\", "/")
        except Exception as exc:
            print(f"Error saving {suffix}: {exc}")
            return None

    empty_profile = render_density_profile(zone, width)
    zone_guidance = render_zone_guidance(image, zone)

    return {
        "steps": {
            "original": save_and_get_path("step_original", image),
            "binary": save_and_get_path("step_binary", binary_image),
            "density_profile": save_and_get_path("step_density_profile", empty_profile),
            "zone_guidance": save_and_get_path("step_zone_guidance", zone_guidance),
            "components_with_boxes": save_and_get_path(
                "step_components_with_boxes", np.zeros_like(image)
            ),
            "classification": save_and_get_path("step_classification", zone_guidance),
            "final_boxes": save_and_get_path("step_final_boxes", image),
        },
        "cropped_results": [],
        "params": params,
        "debug": {
            "main_top": zone["main_top"],
            "main_bottom": zone["main_bottom"],
            "cut_top": zone["cut_top"],
            "cut_bottom": zone["cut_bottom"],
            "peak_row": zone["peak_row"],
            "total_components": 0,
            "anchor_components": 0,
            "secondary_components": 0,
            "groups_formed": 0,
        },
    }
