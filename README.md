# Multimodal Knowledge Graph and Agentic AI Framework for Interpretable Melanoma Treatment Response Prediction

This package implements PATHANCHOR, the framework the manuscript describes. Its
central claim is structural: guideline adherence is a property of the estimator
rather than a prompting strategy, so the recommendation is the arg max of a
path-marginalised response score restricted to the treatment set the guideline
makes admissible. Around that estimator sit four supporting mechanisms — a typed
multimodal melanoma knowledge graph, a budgeted proposal of evidence paths, a
five-agent reasoning team whose interfaces carry typed objects rather than prose,
and a split-conformal layer that sets the abstention region from the width of a
prediction set. The package reproduces the construction chain (ontology, graph
census, grounding, proposal, validation, critique, projection, conformal
calibration), the trust axes the manuscript reports beside discrimination
(calibration error, guideline discordance, net benefit, audit agreement) and the
comparison and ablation runs.

## Installation

With pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

With conda:

```bash
conda env create -f environment.yml
conda activate pathanchor
```

With Docker:

```bash
docker build -t pathanchor .
docker run --rm pathanchor
```

## Data

Official access links, licences and the content check each one passed are in
`dataset_urls.txt`; third-party terms are restated in `NOTICE`.

The construction chain reads six public resources: HGNC, DrugBank, KEGG, STRING,
Hetionet and the guideline statement table. The last is not redistributed — NCCN
Melanoma: Cutaneous v2.2026 is copyright the National Comprehensive Cancer
Network — so `src/pathanchor/guideline/statements.py` ships the record contract
and a declared statement set, and a caller holding a licensed copy can produce
the same table from its own parse.

The four-site retrospective cohort is **not redistributed**. It is held at its
contributing institutions under a data-use agreement, irreversibly anonymised
with every re-linkage key destroyed. What this package ships instead is the
record contract (`src/pathanchor/cohort/schema.py`) and a schema-compatible
cohort fixture (`src/pathanchor/cohort/fixture.py`) that holds the manuscript's
reported margins exactly: the joint molecular-by-treatment table, the site, sex
and age margins, the 60/20/20 partition at 2,572 / 858 / 856, the 1,731
responders, and on the test split the subgroup census of Table III. The
allocation is an integer program over the crossed margins, not a rounding, so
every cell is exact and `margin_deviations` reports zero for each margin.

The four external transcriptomic series are public and are named, with their
accessions, in `src/pathanchor/evaluation/external.py`. They are not
redistributed either: the external module reads a caller-supplied profile table.

Every cohort-level value the manuscript reports is therefore marked `NOT_RUN` in
`verification_report.json` with its reason. The values this package measures are
its own, and they are labelled as such wherever they are printed.

To rebuild the graph snapshot and its census:

```bash
bash scripts/launch_graph.sh
```

Expected on-disk size is roughly 90 MB. `integrity_manifest.json` carries the
SHA-256 of every shipped file.

## Training

```bash
PYTHONPATH=src python3 -m pathanchor.cli.train --config configs/experiment/main.yaml --report reports/train.txt
```

The composite objective has six named terms (response, patient ranking, path
listwise, retention sparsity, retention margin, branch contrast) and the report
prints each one, so a run shows which objective moved rather than only the total.
The manuscript reports a training budget in GPU-hours and no optimiser settings;
the optimiser, schedule, batch size and epoch count in
`configs/train/default.yaml` are declared engineering defaults and appear in the
deviation register of `claim_to_code.json`.

## Evaluation

One command per reported experiment family:

```bash
# The deployed row on the held-out test split.
PYTHONPATH=src python3 -m pathanchor.cli.evaluate --config configs/experiment/main.yaml --report reports/test.txt

# Table II Panel A and Panel B.
PYTHONPATH=src python3 -m pathanchor.cli.ablate --config configs/experiment/ablation_component_panel.yaml --panel panel_a --report reports/panel_a.txt
PYTHONPATH=src python3 -m pathanchor.cli.ablate --config configs/experiment/ablation_leave_one_out.yaml --panel panel_b --report reports/panel_b.txt

# The conformal layer on its own.
PYTHONPATH=src python3 -m pathanchor.cli.calibrate --config configs/experiment/supplement_conformal.yaml --report reports/conformal.txt

# Verification battery, then the integrity manifest last.
PYTHONPATH=src python3 -m pathanchor.cli.verify
```

Every report the commands write is plain text under `reports/`; this repository
carries exactly one Markdown file, this one.

The manuscript's reference quantities, for comparison with whatever the commands
print: AUROC 0.765 (0.732-0.797) and AUPRC 0.719 against 0.758 for the strongest
comparator, Brier 0.152, expected calibration error 0.031 against 0.058, zero
guideline-discordant recommendations against 9.8%, net benefit 0.182 at a
threshold of 0.35 against 0.131, conformal coverage 89.6% at a nominal 90% with a
mean set width of 0.29, external AUROC 0.724 with ECE 0.042, and a reader study at
68.3% / 73.5% / 79.7% across the three conditions. Those are the manuscript's
numbers on its own cohort. This package does not reproduce them and does not
claim to; `verification_report.json` records what it did measure, and the
`manuscript_inconsistencies` block names the places where the article disagrees
with its own tables.

## Compute budget

The reported deployment built the graph in 62 GPU-hours and 18 CPU-hours, trained
for 41 GPU-hours on four A100 accelerators, and ran inference at 2.4 s and 3,180
language-model tokens per patient against 0.06 s for the multimodal transformer
and 12.7 s for the strongest agentic comparator, for 214 kWh in total. None of
those figures is softened here, and none of them was measured by this package:
they are the reported measurements, carried in the comment block of
`configs/experiment/main.yaml`.

What this package does require is a CPU, about 700 MB for the full graph and a few
minutes for the test-split evaluation. The two are not interchangeable.

## License

The source is distributed under Apache-2.0; see `LICENSE`. Third-party resources
keep their own terms and are named, with version, access route and licence, in
`NOTICE` and `dataset_urls.txt`. No third-party data is bundled.
