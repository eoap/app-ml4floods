# app-ml4floods

ML4Floods is an end-to-end flood delineation workflow for optical satellite data. This repository contains:

- a CWL workflow in [cwl-workflow/app-ml4floods.cwl](/data/work/github/eoap/app-ml4floods/cwl-workflow/app-ml4floods.cwl)
- a Python command-line tool in [command-line-tools/ml4floods-cli](/data/work/github/eoap/app-ml4floods/command-line-tools/ml4floods-cli)
- generated user documentation in [docs/ml4floods.md](/data/work/github/eoap/app-ml4floods/docs/ml4floods.md)

## Documentation

Published documentation: <https://eoap.github.io/app-ml4floods/>

Local documentation sources:

- [docs/index.md](/data/work/github/eoap/app-ml4floods/docs/index.md)
- [docs/ml4floods.md](/data/work/github/eoap/app-ml4floods/docs/ml4floods.md)

## Pull The CWL Package

You can pull the packaged CWL artifact from GHCR with `oras`:

```bash
oras pull ghcr.io/eoap/app-ml4floods/app-ml4floods:latest-dev
```

Tag mapping:

- `latest-dev` for the `develop` branch
- `latest` for the `main` branch
- `x.y.z` for a released tag version

## Run The CLI

From the repository root:

```bash
task run
```

Or directly from the CLI package:

```bash
cd command-line-tools/ml4floods-cli
hatch run ml4floods-cli \
  --product-uri "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20241031T105109_R051_T31SBD_20241031T133016" \
  --water-threshold 0.7 \
  --brightness-threshold 3500
```

## Run The CWL Workflow

From the repository root:

```bash
task run-cwl
```

This generates a temporary CWL inputs file with the `product_uri` object expected by the workflow and removes it after execution.

If you pulled the CWL package from GHCR, you can run the extracted workflow with `cwltool` using the same `product_uri` object shape described in the generated documentation.

## Development Tasks

```bash
task validate
task test
task lint
task check
task generate-doc
```

## Notes

- The CLI now expects `--product-uri`, not `--input-item`.
- The direct CLI interface takes a plain URL string.
- The CWL workflow takes a structured URI input and maps its `value` field to `--product-uri`.

## License

`app-ml4floods` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
