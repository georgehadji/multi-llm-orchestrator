"""Re-run graphify on the orchestrator codebase after Phase 5 refactoring.
Orchestrator: scans all files under orchestrator/ and rebuilds graph."""
import json
from pathlib import Path
from collections import Counter

from graphify.detect import detect
from graphify.extract import extract as extract_fn
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html

def main():
    OUTPUT = Path("graphify-out")
    ROOT = Path("orchestrator").resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / ".graphify_root").write_text(str(ROOT))

    print("=== Phase 1: Detect files ===", flush=True)
    detection = detect(ROOT)
    print(f"  Detection complete", flush=True)

    print("\n=== Phase 2: Extract AST ===", flush=True)
    all_files = sorted(ROOT.rglob("*"))
    print(f"  Scanning {len(all_files)} files...", flush=True)
    nodes_edges = extract_fn(all_files)
    n_nodes = len(nodes_edges.get("nodes", []))
    n_edges = len(nodes_edges.get("edges", []))
    print(f"  Nodes: {n_nodes}, Edges: {n_edges}", flush=True)

    print("\n=== Phase 3: Build graph ===", flush=True)
    graph = build_from_json(nodes_edges, directed=True, root=ROOT)
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges", flush=True)

    print("\n=== Phase 4: Cluster ===", flush=True)
    community_map = cluster(graph)  # dict[int, list[str]]
    n_comms = len(set(community_map.keys()))
    print(f"  Communities: {n_comms}", flush=True)
    cohesion = score_all(graph, community_map)

    print("\n=== Phase 5: Analyze ===", flush=True)
    gods = god_nodes(graph, top_n=20)
    print(f"  God nodes: {len(gods)}", flush=True)
    surprising = surprising_connections(graph, community_map, top_n=10)
    print(f"  Surprising connections: {len(surprising)}", flush=True)

    comm_sizes = {cid: len(members) for cid, members in community_map.items()}
    labels = {cid: f"Community_{cid}" for cid in community_map}
    questions = suggest_questions(graph, community_map, labels, top_n=5)
    print(f"  Suggested questions: {len(questions)}", flush=True)

    print("\n=== Phase 6: Export ===", flush=True)

    # JSON graph
    graph_json = OUTPUT / "graph.json"
    to_json(graph, community_map, str(graph_json), force=True)
    print(f"  JSON: {graph_json}", flush=True)

    # HTML visualization
    try:
        html_path = OUTPUT / "graph.html"
        to_html(graph, community_map, str(html_path), community_labels=labels, member_counts=comm_sizes)
        print(f"  HTML: {html_path}", flush=True)
    except Exception as e:
        print(f"  HTML skipped: {e}", flush=True)

    # Markdown report
    report_path = OUTPUT / "GRAPH_REPORT.md"
    generate(
        graph, community_map, cohesion, labels, gods, surprising,
        detection,
        {"total_cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0},
        root=str(ROOT), suggested_questions=questions,
    )
    print(f"  Report: {report_path}", flush=True)

    # Save community labels
    label_path = OUTPUT / ".graphify_labels.json"
    json.dump(labels, label_path.open("w"), indent=2)
    print(f"  Labels: {label_path}", flush=True)

    # Save manifest
    manifest = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "communities": n_comms,
        "root": str(ROOT),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest: saved", flush=True)

    print("\n=== Done ===", flush=True)

if __name__ == '__main__':
    main()