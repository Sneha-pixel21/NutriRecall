"""Extractive source units and positive food-list projections.

The model selects IDs/indices only. Sentence text, list relationships and scope
come from the curated passages. Valid selection does not prove relevance.
"""

import re


def _food_list(text):
    clause = text.strip().rstrip(".;")
    patterns = (
        (r"Protein-containing foods include (.+)", "protein_foods"),
        (r"(.+) provide protein", "protein_foods"),
        (r"(.+) can add calories", "calorie_foods"),
    )
    for pattern, kind in patterns:
        match = re.fullmatch(pattern, clause, re.I)
        if match:
            items = re.split(r",\s*(?:and\s+)?|\s+and\s+", match[1])
            # Only plain, positive noun lists in these source constructions.
            if len(items) > 1 and all(re.fullmatch(r"[A-Za-z][A-Za-z -]*", item.strip()) for item in items):
                return kind, [item.strip() for item in items]
    return "sentence", []


def build_units(packet):
    result = {}
    for source_id, source in packet.items():
        body = source["text"].partition("Applies to:")[0].strip()
        units = []
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            clauses = [clause.strip() for clause in sentence.split(";")]
            # Split only when every clause is an independently recognized food
            # relationship. Never split a general statement at its negation.
            pieces = clauses if len(clauses) > 1 and all(_food_list(c)[1] for c in clauses) else [sentence]
            for piece in pieces:
                kind, items = _food_list(piece)
                units.append({"index": len(units), "text": piece, "kind": kind, "items": items})
        if source["source_review_warning"]:
            units.append({"index": len(units), "text": source["source_review_warning"],
                          "kind": "source_review", "items": []})
        result[source_id] = {**source, "units": units}
    return result


def model_packet(sources):
    """Send each source fact once, without a second full-passage copy."""
    return {source_id: {
        "units": [{"index": unit["index"], "kind": unit["kind"],
                   **({"items": {str(index): item for index, item in enumerate(unit["items"])}}
                      if unit["items"] else {"text": unit["text"]})}
                  for unit in source["units"]],
        "population": source["population"], "limitations": source["limitations"],
    } for source_id, source in sources.items()}


def _list_text(unit, indices):
    names = [unit["items"][index] for index in indices]
    text = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
    label = "Protein food examples" if unit["kind"] == "protein_foods" else "Calorie additions"
    return f"{label}: {text}."


def validate_selections(plan, sources):
    accepted, issues, unknown = [], [], []
    prefixes = {source_id.rsplit("_", 1)[0] for source_id in sources}
    for position, selection in enumerate(plan["claims"], 1):
        source_id = selection["source_id"]
        if source_id not in sources:
            # Unknown model strings are not allowed to echo credentials or data.
            safe = source_id if re.fullmatch(r"[a-z]+_\d{1,3}", source_id) and source_id.rsplit("_", 1)[0] in prefixes else "unavailable_source"
            unknown.append(safe)
            issues.append(f"Unavailable source in selection {position}: {safe}")
            continue
        units = sources[source_id]["units"]
        index = selection["unit_index"]
        if not 0 <= index < len(units):
            issues.append(f"Invalid sentence index in selection {position} for {source_id}")
            continue
        unit = units[index]
        indices = list(dict.fromkeys(selection["item_indices"]))
        if unit["items"]:
            indices = indices or list(range(len(unit["items"])))
            if any(not 0 <= item < len(unit["items"]) for item in indices):
                issues.append(f"Invalid food item index in selection {position} for {source_id}")
                continue
            text = _list_text(unit, indices)
        elif indices:
            issues.append(f"Food items supplied for a non-list sentence in selection {position}")
            continue
        else:
            text = unit["text"]
        accepted.append({"source_id": source_id, "unit_index": index, "item_indices": indices,
                         "text": text, "quote": unit["text"]})
    return accepted, issues, sorted(set(unknown))


def render_selections(selections, sources, intent):
    lines = []
    for source_id in dict.fromkeys(selection["source_id"] for selection in selections):
        source = sources[source_id]
        units = source["units"]
        selected = [choice for choice in selections if choice["source_id"] == source_id]
        if intent == "medical_boundary":
            if source["limitations"]:
                lines.append(f"- {source['limitations']} [{source_id}]")
            continue
        food_source = any(unit["items"] for unit in units)
        advice = any(units[choice["unit_index"]]["kind"] != "source_review" for choice in selected)
        emitted = set()
        if advice and not food_source:
            # Keep the complete small context for health guidance, including
            # uncertainty, negation and general-vs-personal qualifications.
            text = source["text"].partition("Applies to:")[0].strip()
            lines.append(f"- {text} [{source_id}]")
        elif advice:
            for choice in selected:
                unit = units[choice["unit_index"]]
                if unit["kind"] == "source_review":
                    continue
                if choice["text"] not in emitted:
                    lines.append(f"- {choice['text']} [{source_id}]")
                    emitted.add(choice["text"])
            # Preserve explicit uncertainty/negative qualifications such as
            # supplements not automatically being necessary, even for subsets.
            for unit in units:
                if unit["kind"] == "sentence" and re.search(r"\b(?:not|no|mixed|uncertain)\b", unit["text"], re.I) and unit["text"] not in emitted:
                    lines.append(f"- {unit['text']} [{source_id}]")
                    emitted.add(unit["text"])
        for choice in selected:
            if units[choice["unit_index"]]["kind"] == "source_review" and choice["text"] not in emitted:
                lines.append(f"- Source review: {choice['text']} [{source_id}]")
                emitted.add(choice["text"])
        if advice:
            scope = f"Applies to: {source['population']} Limitations: {source['limitations']}"
            lines.append(f"*{scope}* [{source_id}]")
    return lines
