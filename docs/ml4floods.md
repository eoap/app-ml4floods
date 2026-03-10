# ML4Floods inference for flood extent estimation using pre-trained model on Sentinel-2 or Landsat-9 data v0.3.5

ML4Floods is an end-to-end ML pipeline for flood extent estimation using optical satellite data from Sentinel-2 or Landsat-8/9 acquisition

> This software is licensed under the terms of the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) license - SPDX short identifier: [CC-BY-4.0](https://spdx.org/licenses/CC-BY-4.0)
>
> 2025-10-29 - 2026-03-10T11:41:30.456 Copyright [Terradue Srl](mailto:info@terradue.com) - > [https://ror.org/0069cx113](https://ror.org/0069cx113)

## Project Team

### Authors

| Name | Email | Organization | Role | Identifier |
|------|-------|--------------|------|------------|
| Brito, Fabrice | [fabrice.brito@terradue.com](mailto:fabrice.brito@terradue.com) | [Terradue](https://ror.org/0069cx113) | [Project Manager](http://purl.org/spar/datacite/ProjectManager) | [https://orcid.org/0009-0000-1342-9736](https://orcid.org/0009-0000-1342-9736) |
| Re, Alice | [alice.re@terradue.com](mailto:alice.re@terradue.com) | [Terradue](https://ror.org/0069cx113) | [Researcher](http://purl.org/spar/datacite/Researcher) | [https://orcid.org/0000-0001-7068-5533](https://orcid.org/0000-0001-7068-5533) |
| Tripodi, Simone | [simone.tripodi@terradue.com](mailto:simone.tripodi@terradue.com) | [Terradue](https://ror.org/0069cx113) | [Project Leader](http://purl.org/spar/datacite/ProjectLeader) | [https://orcid.org/0009-0006-2063-618X](https://orcid.org/0009-0006-2063-618X) |


### Contributors

| Name | Email | Organization | Role | Identifier |
|------|-------|--------------|------|------------|
| Vaccari, Simone | [simone.vaccari@terradue.com](mailto:simone.vaccari@terradue.com) | [Terradue](https://ror.org/0069cx113) | [Researcher](http://purl.org/spar/datacite/Researcher) | [https://orcid.org/0000-0002-2757-4165](https://orcid.org/0000-0002-2757-4165) |



## User Manual

User Manual can be found on [https://eoap.github.io/app-ml4floods/](https://eoap.github.io/app-ml4floods/).


## Runtime environment

### Supported Operating Systems

- Linux
- MacOS X

### Requirements

- [https://cwltool.readthedocs.io/en/latest/](https://cwltool.readthedocs.io/en/latest/)
- [https://www.python.org/](https://www.python.org/)


## Software Source code

- Browsable version of the [source repository](https://github.com/eoap/app-ml4floods.git);
- [Continuous integration](https://github.com/eoap/app-ml4floods/actions) system used by the project;
- Issues, bugs, and feature requests should be submitted to the following [issue management](https://github.com/eoap/app-ml4floods/issues) system for this project


---


## ml4floods

### CWL Class

`Workflow`

### Inputs

| Id | Type | Label | Doc |
|----|------|-------|-----|
| `product_uri` | `https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI` | Optical satellite acquisition | Sentinel-2 or Landsat-9 acquisition to be processed |
| `water-threshold` | `[ null, float ]` | Water threshold | Threshold for water detection (default 0.7) |
| `brightness-threshold` | `[ null, int ]` | Brightness threshold | Threshold for brightness (default 3500) |


### Steps

| Id | Runs | Label | Doc |
|----|------|-------|-----|
| [inference](#ml4floods-cli) | `#ml4floods-cli` | None | None |


### Outputs

| Id | Type | Label | Doc |
|----|------|-------|-----|
| `flood-delineation` | `Directory` | None | None |


### UML Diagrams


#### UML `activity` diagram

![ml4floods flow diagram](./ml4floods/activity.svg "ml4floods activity diagram")

#### UML `component` diagram

![ml4floods flow diagram](./ml4floods/component.svg "ml4floods component diagram")

#### UML `class` diagram

![ml4floods flow diagram](./ml4floods/class.svg "ml4floods class diagram")

#### UML `sequence` diagram

![ml4floods flow diagram](./ml4floods/sequence.svg "ml4floods sequence diagram")

#### UML `state` diagram

![ml4floods flow diagram](./ml4floods/state.svg "ml4floods state diagram")






## ml4floods-cli

### CWL Class

```
CommandLineTool
```

### Inputs

| Id | Option | Type |
|----|------|-------|
| `product_uri` | `--product-uri` | `https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI` |
| `water_threshold` | `--water-threshold` | `[ null, float ]` |
| `brightness_threshold` | `--brightness-threshold` | `[ null, int ]` |

### Execution usage example:

```
ml4floods-cli \
--product-uri <PRODUCT_URI> \
(--water-threshold <WATER_THRESHOLD>) \
(--brightness-threshold <BRIGHTNESS_THRESHOLD>)
```


### Run in step

`inference`
