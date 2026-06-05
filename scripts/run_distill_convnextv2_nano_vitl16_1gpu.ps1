[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$DataExtra,

    [Parameter(Mandatory = $true)]
    [string]$TeacherCheckpoint,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [string]$ConfigFile = "dinov3/configs/train/distillation_convnext/convnextv2_nano_distill_vitl16_1gpu.yaml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $DataRoot)) {
    throw "DataRoot does not exist: $DataRoot"
}
if (-not (Test-Path -Path $DataExtra)) {
    throw "DataExtra does not exist: $DataExtra"
}
if (-not (Test-Path -Path $TeacherCheckpoint)) {
    throw "TeacherCheckpoint does not exist: $TeacherCheckpoint"
}
if (-not (Test-Path -Path $ConfigFile)) {
    throw "Config file does not exist: $ConfigFile"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

$env:PYTHONPATH = $repoRoot

$datasetPath = "ImageNet:split=TRAIN:root=$DataRoot:extra=$DataExtra"

Write-Host "Starting distillation run"
Write-Host "RepoRoot         : $repoRoot"
Write-Host "ConfigFile       : $ConfigFile"
Write-Host "DatasetPath      : $datasetPath"
Write-Host "TeacherCheckpoint: $TeacherCheckpoint"
Write-Host "OutputDir        : $OutputDir"

& $pythonExe -m torch.distributed.run --nproc_per_node=1 dinov3/train/train.py `
    --config-file $ConfigFile `
    --output-dir $OutputDir `
    train.dataset_path=$datasetPath `
    distillation.checkpoint_path=$TeacherCheckpoint
