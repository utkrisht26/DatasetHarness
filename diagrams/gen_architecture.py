"""
Generator for the Agentic Dataset Harness architecture diagram (.drawio).

Layered "two-engine + shared spine" architecture:
    Job Spec (goal, NO sources) -> Engine A (Discovery + Acquisition) -> Raw Corpus
        -> Engine B (Synthesis + Curation) -> Output
Engine A leads with SOURCE DISCOVERY: the user states a goal; the system finds where the data lives
across three channels (existing datasets > web crawl > pure synthesis), with a human approval gate.
A foundation band holds the cross-cutting concerns (the SHARED SPINE).

Run:  python3 gen_architecture.py
Output: architecture.drawio  (open in draw.io / diagrams.net or the VS Code extension)
"""

import xml.etree.ElementTree as ET

mxfile = ET.Element(
    "mxfile",
    {
        "host": "app.diagrams.net",
        "agent": "AgenticDatasetHarness/architecture-generator",
        "version": "24.0.0",
        "type": "device",
    },
)
diagram = ET.SubElement(mxfile, "diagram", {"id": "adh-arch", "name": "Agentic Dataset Harness"})
model = ET.SubElement(
    diagram,
    "mxGraphModel",
    {
        "dx": "2400",
        "dy": "1500",
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": "2700",
        "pageHeight": "1500",
        "math": "0",
        "shadow": "0",
    },
)
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

S = {
    "title": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=26;fontStyle=1;fontColor=#0F172A;",
    "sub": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=13;fontColor=#475569;",
    "engineA": "rounded=1;whiteSpace=wrap;html=1;fillColor=#F0F9FF;strokeColor=#0284C7;strokeWidth=2;fontColor=#075985;fontStyle=1;fontSize=15;align=center;verticalAlign=top;spacingTop=8;arcSize=6;",
    "engineB": "rounded=1;whiteSpace=wrap;html=1;fillColor=#F0FDF4;strokeColor=#16A34A;strokeWidth=2;fontColor=#166534;fontStyle=1;fontSize=15;align=center;verticalAlign=top;spacingTop=8;arcSize=6;",
    "spine": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF7ED;strokeColor=#EA580C;strokeWidth=2;fontColor=#9A3412;fontStyle=1;fontSize=15;align=center;verticalAlign=top;spacingTop=8;arcSize=6;",
    "input": "rounded=1;whiteSpace=wrap;html=1;fillColor=#EEF2FF;strokeColor=#4F46E5;strokeWidth=2;fontColor=#3730A3;fontStyle=1;fontSize=13;align=center;arcSize=14;",
    "a_node": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#0EA5E9;strokeWidth=1.5;fontColor=#0C4A6E;fontSize=12;align=center;arcSize=12;",
    "gate": "rhombus;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=2;fontColor=#92400E;fontStyle=1;fontSize=12;",
    "chan": "rounded=1;whiteSpace=wrap;html=1;fillColor=#E0F2FE;strokeColor=#0369A1;strokeWidth=2;fontColor=#0C4A6E;fontStyle=1;fontSize=12;align=center;arcSize=10;",
    "b_node": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#22C55E;strokeWidth=1.5;fontColor=#14532D;fontSize=12;align=center;arcSize=12;",
    "s_node": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#F97316;strokeWidth=1.5;fontColor=#7C2D12;fontSize=12;align=center;arcSize=12;",
    "corpus": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=14;fillColor=#FEF9C3;strokeColor=#CA8A04;strokeWidth=2;fontColor=#713F12;fontStyle=1;fontSize=12;",
    "out": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF1F2;strokeColor=#E11D48;strokeWidth=2;fontColor=#9F1239;fontStyle=1;fontSize=12;align=center;arcSize=12;",
}
E = {
    "flow": "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#475569;strokeWidth=2;fontSize=11;fontColor=#1E293B;endArrow=block;",
    "data": "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#0284C7;strokeWidth=2;dashed=1;fontSize=11;fontColor=#0369A1;endArrow=block;",
    "instr": "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#EA580C;strokeWidth=2;dashed=1;fontSize=11;fontColor=#9A3412;endArrow=open;",
    "loop": "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#7C3AED;strokeWidth=2;dashed=1;fontSize=11;fontColor=#5B21B6;endArrow=block;",
}


def node(i, val, x, y, w, h, style):
    c = ET.SubElement(
        root,
        "mxCell",
        {"id": str(i), "value": val, "style": S[style], "vertex": "1", "parent": "1"},
    )
    ET.SubElement(
        c,
        "mxGeometry",
        {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
    )


def edge(i, val, src, tgt, style):
    c = ET.SubElement(
        root,
        "mxCell",
        {
            "id": str(i),
            "value": val,
            "style": E[style],
            "edge": "1",
            "parent": "1",
            "source": str(src),
            "target": str(tgt),
        },
    )
    ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})


# ---------- titles ----------
node(10, "Agentic Dataset Generation Harness", 700, 20, 1300, 40, "title")
node(
    11,
    "Discovery-first: the user gives a goal, NOT sources. The system finds where the data lives (datasets > web > synthesis), then curates it into a verified dataset.",
    500,
    64,
    1700,
    24,
    "sub",
)

# ---------- input ----------
node(
    100,
    "Job Spec\n(goal + target record schema +\nverifiability tier + limits;\nNO sources required)",
    60,
    320,
    220,
    150,
    "input",
)

# ---------- ENGINE A: DISCOVERY + ACQUISITION ----------
node(
    200,
    "ENGINE A — DISCOVERY + ACQUISITION  (deepagents / LangGraph)",
    320,
    130,
    940,
    470,
    "engineA",
)
node(
    201,
    "Source Discovery Agent\ngoal -> aspects -> diverse queries  •  evaluate + rank candidate sources",
    350,
    185,
    880,
    50,
    "a_node",
)
node(
    202,
    "Channel Router\nper aspect, pick the cheapest path:  reuse  >  crawl  >  synthesize",
    350,
    245,
    880,
    42,
    "a_node",
)
node(203, "Human Approval Gate\n(default ON; flag to skip)", 350, 300, 880, 48, "gate")
node(
    204,
    "DATASETS channel\nHF Hub / Kaggle / data.gov\nfetch + reuse (no scrape)\nhighest-yield path",
    350,
    360,
    285,
    150,
    "chan",
)
node(
    205,
    "WEB channel\nCrawl4AI MCP (SSE)\n+ web-scout subagents\nadaptive saturation stop\ncontext-offload",
    655,
    360,
    285,
    150,
    "chan",
)
node(
    206,
    "SYNTHESIS channel\nmodel-knowledge generation\nNO external source\n(-> Engine B directly)",
    960,
    360,
    270,
    150,
    "chan",
)
node(
    207,
    "Virtual FS + durable checkpointing (LangGraph)  +  limit circuit breaker (floor)",
    350,
    525,
    880,
    40,
    "a_node",
)

# ---------- CORPUS ----------
node(300, "Raw Corpus\n+ per-record\nprovenance", 1310, 370, 150, 150, "corpus")

# ---------- ENGINE B: SYNTHESIS + CURATION ----------
node(400, "ENGINE B — SYNTHESIS + CURATION  (the missing half)", 1510, 130, 900, 470, "engineB")
node(
    401,
    "Taxonomy / Seed Engine\npersonas • topics • difficulty (Self-Instruct / Evol-Instruct / Persona Hub)",
    1540,
    185,
    840,
    52,
    "b_node",
)
node(402, "Record Generation  (Distilabel Pipeline -> Task -> LLM)", 1540, 247, 840, 52, "b_node")
node(
    403,
    "Verification Stack — tiered, cheapest-first\nhash -> heuristics -> n-gram decontam -> MinHash -> deterministic verifier -> quality clf -> semantic dedup -> LLM-jury",
    1540,
    309,
    840,
    72,
    "b_node",
)
node(
    404,
    "Dedup + Diversity + Decontamination  (datatrove MinHash • SemDeDup • n-gram + paraphrase)",
    1540,
    391,
    840,
    52,
    "b_node",
)
node(
    405,
    "Label / Structure / Format  — validate against target record schema",
    1540,
    453,
    840,
    48,
    "b_node",
)

# ---------- OUTPUT ----------
node(500, "Dataset\nJSONL / Parquet", 2460, 300, 230, 58, "out")
node(501, "Manifest + Datasheet\n+ Croissant metadata", 2460, 368, 230, 72, "out")
node(502, "Versioned release\n(DVC / HF Datasets)", 2460, 450, 230, 52, "out")

# ---------- SHARED SPINE ----------
node(600, "SHARED SPINE  —  cross-cutting, instruments both engines", 320, 660, 2370, 235, "spine")
node(601, "Orchestration\nLangGraph + checkpointer / store", 350, 715, 400, 72, "s_node")
node(
    602,
    "State, Limits + Circuit Breaker\nmetrics reducers, graceful shutdown",
    775,
    715,
    400,
    72,
    "s_node",
)
node(603, "Storage\nS3 / MinIO object store", 1200, 715, 360, 72, "s_node")
node(604, "Provenance + Lineage\nW3C PROV / MLMD per record", 1585, 715, 360, 72, "s_node")
node(605, "Cost + Model Routing\ncheap bulk gen -> frontier judge", 1970, 715, 470, 72, "s_node")
node(
    606,
    "Observability\n$/accepted-record • accept rate • coverage • drift",
    350,
    805,
    620,
    60,
    "s_node",
)
node(607, "Governance + HITL\nlicense/PII • Argilla review surface", 1000, 805, 600, 60, "s_node")
node(608, "Reproducibility\nseeds • prompt hashes • config snapshots", 1630, 805, 810, 60, "s_node")

# ---------- edges ----------
edge(1000, "1. goal + record schema (no sources)", 100, 201, "flow")
edge(1001, "2. aspects + queries", 201, 202, "flow")
edge(1002, "3. ranked candidate sources", 202, 203, "flow")
edge(1003, "approve: reuse", 203, 204, "flow")
edge(1004, "approve: crawl", 203, 205, "flow")
edge(1005, "approve: synthesize", 203, 206, "flow")
edge(1006, "fetched + provenance", 204, 300, "data")
edge(1007, "scraped + provenance", 205, 300, "data")
edge(1008, "synthesize directly (no corpus)", 206, 402, "loop")
edge(1009, "derive taxonomy", 300, 401, "data")
edge(1010, "ground generation", 300, 402, "data")
edge(1011, "seeds", 401, 402, "flow")
edge(1012, "candidates", 402, 403, "flow")
edge(1013, "survivors", 403, 404, "flow")
edge(1014, "reject / re-seed", 403, 401, "loop")
edge(1015, "deduped + decontaminated", 404, 405, "flow")
edge(1016, "emit records", 405, 500, "flow")
edge(1017, "coverage gap -> re-discover", 400, 201, "loop")
edge(1018, "spec drives orchestration", 100, 601, "instr")
edge(1019, "instrumented by", 200, 600, "instr")
edge(1020, "instrumented by", 400, 600, "instr")

ET.ElementTree(mxfile).write("architecture.drawio", encoding="utf-8", xml_declaration=True)
print("wrote architecture.drawio")
