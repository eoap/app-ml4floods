cwlVersion: v1.2
$namespaces:
  s: https://schema.org/

schemas:
- http://schema.org/version/9.0/schemaorg-current-http.rdf

s:softwareVersion: 0.1.0

s:description: ML4Floods is an end-to-end ML pipeline for flood extent estimation using optical satellite data from Sentinel-2 or
  Landsat-8/9 acquisition
s:applicationCategory: Earth Observation application package
s:applicationSubCategory: CWL Workflow
s:author:
- '@type': s:Person
  s:affiliation:
    '@type': s:Organization
    s:identifier: https://ror.org/0069cx113
    s:name: Terradue
  s:email: fabrice.brito@terradue.com
  s:familyName: Brito
  s:givenName: Fabrice
  s:identifier: https://orcid.org/0009-0000-1342-9736
- '@type': s:Person
  s:affiliation:
    '@type': s:Organization
    s:identifier: https://ror.org/0069cx113
    s:name: Terradue
  s:email: alice.re@terradue.com
  s:familyName: Re
  s:givenName: Alice
  s:identifier: https://orcid.org/0000-0001-7068-5533
- '@type': s:Person
  s:affiliation:
    '@type': s:Organization
    s:identifier: https://ror.org/0069cx113
    s:name: Terradue
  s:email: simone.tripodi@terradue.com
  s:familyName: Tripodi
  s:givenName: Simone
  s:identifier: https://orcid.org/0009-0006-2063-618X
s:codeRepository: https://github.com/eoap/app-ml4floods.git
s:copyrightYear: 2025
s:dateCreated: '2025-10-29'

s:name: ML4Floods inference for flood extent estimation using pre-trained model on Sentinel-2 or Landsat-9 data
s:keywords:
- CWL
- Workflow
- Earth Observation
- '@type': s:DefinedTerm
  s:description: delineation
  s:name: application-type
- '@type': s:DefinedTerm
  s:description: hydrology
  s:name: domain
- '@type': s:DefinedTerm
  s:inDefinedTermSet: https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/sciencekeywords
  s:termCode: 959f1861-a776-41b1-ba6b-d23c71d4d1eb
s:license:
  '@type': s:CreativeWork
  s:identifier: CC-BY-4.0
  s:name: License CC BY 4.0
  s:url: https://creativecommons.org/licenses/by/4.0/

s:operatingSystem:
- Linux
- MacOS X
s:publisher:
  '@type': s:Organization
  s:email: info@terradue.com
  s:identifier: https://ror.org/0069cx113
  s:name: Terradue Srl

s:softwareHelp:
  '@type': s:CreativeWork
  s:name: User Manual
  s:url: https://eoap.github.io/app-ml4floods/
s:softwareRequirements:
- https://cwltool.readthedocs.io/en/latest/
- https://www.python.org/


$graph:
- class: Workflow
  id: ml4floods
  label: ML4Floods inference for flood extent estimation
  doc: ML4Floods inference for flood extent estimation using pre-trained model on Sentinel-2 or Landsat-9 data
  requirements:
    - class: InlineJavascriptRequirement
    - class: ScatterFeatureRequirement
      
  inputs:
    input-item: 
      label: Optical satellite acquisition
      doc: Sentinel-2 or Landsat-9 acquisition to be processed
      type: string
    water-threshold:
      label: Water threshold
      doc: Threshold for water detection (default 0.7)
      type: float?
      default: 0.7
    brightness-threshold:
      label: Brightness threshold
      doc: Threshold for brightness (default 3500)
      type: int?
      default: 3500
  outputs: 
    - id: flood-delineation
      outputSource: 
        - inference/flood-delineation
      type: Directory
  steps:
    inference:
      run: "#ml4floods-cli"
      in:
        input_item: input-item
        water_threshold: water-threshold
        brightness_threshold: brightness-threshold
      out: 
        - flood-delineation

- class: CommandLineTool
  id: ml4floods-cli
  
  hints:
    DockerRequirement:
      dockerPull: docker.io/library/ml4floods-cli:latest
          
  baseCommand: ["ml4floods-cli"]
  inputs:
    input_item:
      type: string
      inputBinding:
        prefix: --input-item 
    water_threshold:
      type: float?
      inputBinding:
        prefix: --water-threshold
    brightness_threshold:
      type: int?
      inputBinding:
        prefix: --brightness-threshold
  outputs: 
    flood-delineation:
      outputBinding:
        glob: .
      type: Directory
  requirements:
    InlineJavascriptRequirement: {}
    EnvVarRequirement:
      envDef:
        MPLCONFIGDIR: /tmp/matplotlib
    NetworkAccess:
      networkAccess: true
    ResourceRequirement:
      coresMax: 1
      ramMax: 3000    


