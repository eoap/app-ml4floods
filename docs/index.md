# ML4Floods

ML4Floods is an end-to-end flood delineation workflow for Sentinel-2 and related optical satellite products.

This site documents:

- the packaged CWL workflow
- the `ml4floods-cli` command-line interface
- local development and validation tasks

## Pull The CWL Package

Pull the packaged CWL artifact from GHCR with `oras`:

```bash
oras pull ghcr.io/eoap/app-ml4floods/app-ml4floods:latest-dev
```

Available tag conventions:

- `latest-dev` tracks the `develop` branch
- `latest` tracks the `main` branch
- `x.y.z` corresponds to a release tag

## Quick Start

Run the CLI from the repository root:

```bash
task run
```

Run the CWL workflow:

```bash
task run-cwl
```

Validate the workflow and metadata:

```bash
task validate
```

## Key Paths

- Workflow: `cwl-workflow/app-ml4floods.cwl`
- CLI package: `command-line-tools/ml4floods-cli`
- Generated workflow reference: `docs/ml4floods.md`

## Interface Summary

- CLI option: `--product-uri`
- CLI input type: plain URL string
- CWL input: `product_uri` object with a `value` field

See [ML4Floods](ml4floods.md) for the generated workflow and command reference.
