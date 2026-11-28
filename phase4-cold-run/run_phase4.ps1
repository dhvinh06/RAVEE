param(
    [Parameter(Mandatory)][string] $VideoDir,
    [Parameter(Mandatory)][string] $Dataset,
    [Parameter(Mandatory)][string] $Scorer,
    [Parameter(Mandatory)][string] $CudaScratch,
    [Parameter(Mandatory)][string] $CpuScratch
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo 'RAVEE_optz\.venv\Scripts\python.exe'
$runner = Join-Path $PSScriptRoot 'benchmark_phase4.py'
$tempPath = (Resolve-Path -LiteralPath $env:TEMP).Path
$scenes = @('scene01', 'scene02')
$runs = @(
    @{ Name = 'E2'; Backend = 'cuda'; Scratch = $CudaScratch },
    @{ Name = 'E1'; Backend = 'cpu'; Scratch = $CpuScratch }
)

foreach ($run in $runs) {
    $path = [IO.Path]::GetFullPath($run.Scratch)
    if ([IO.Path]::GetDirectoryName($path) -ne $tempPath -or
        [IO.Path]::GetFileName($path) -notlike 'ravee-phase4-*' -or
        (Test-Path -LiteralPath $path) -or
        (Test-Path -LiteralPath (Join-Path $PSScriptRoot $run.Name))) {
        throw "Expected new scratch and archive paths for $($run.Name): $path"
    }
}
if ([IO.Path]::GetFullPath($CudaScratch) -eq [IO.Path]::GetFullPath($CpuScratch)) {
    throw 'E2 and E1 require distinct scratch directories.'
}

foreach ($run in $runs) {
    New-Item -ItemType Directory -Path $run.Scratch | Out-Null
}

foreach ($run in $runs) {
    $archive = Join-Path $PSScriptRoot $run.Name
    $backend = $run.Backend
    $env:RAVEE_IDT_BACKEND = $backend
    $env:RAVEE_IDT_WORKERS = '6'
    $env:RAVEE_FV_SEED = '0'
    foreach ($scene in $scenes) {
        $video = Join-Path $VideoDir "${scene}_cam2_15fps.mp4"
        & $python -B -u $runner --repo $repo --input-video $video --scratch $run.Scratch --archive $archive --scene $scene --backend $backend
        if ($LASTEXITCODE -ne 0) { throw "$($run.Name) $scene failed with exit code $LASTEXITCODE." }
    }
    $score = & $python -B $Scorer $archive --repo $repo --dataset $Dataset
    if ($LASTEXITCODE -ne 0) { throw "$($run.Name) scorer failed with exit code $LASTEXITCODE." }
    $scoreObject = ($score -join "`n") | ConvertFrom-Json
    if ($scoreObject.combined.seconds -ne 614) { throw "$($run.Name) expected 614 scored seconds." }
    $summaryPath = Join-Path $archive 'summary.json'
    [IO.File]::WriteAllText($summaryPath, ($score -join "`n") + "`n")
    Write-Output "$($run.Name) SCORED $summaryPath"

    $scratch = (Resolve-Path -LiteralPath $run.Scratch).Path
    if ([IO.Path]::GetDirectoryName($scratch) -ne $tempPath -or
        [IO.Path]::GetFileName($scratch) -notlike 'ravee-phase4-*') {
        throw "Scratch escaped the dedicated TEMP directory: $scratch"
    }
    $removed = 0
    foreach ($scene in $scenes) {
        $stem = "1_${scene}_cam2_15fps_reduced"
        foreach ($relative in @(
            "$scene\preprocessed\$stem.avi",
            "$scene\idt\$stem-trajectory.npy",
            "$scene\fv\$stem-trajectory-frame_fv.txt"
        )) {
            $file = Get-Item -LiteralPath (Join-Path $scratch $relative)
            if (-not $file.FullName.StartsWith($scratch + '\', [StringComparison]::OrdinalIgnoreCase) -or
                ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "Unexpected scratch artifact: $($file.FullName)"
            }
            $removed += $file.Length
            Remove-Item -LiteralPath $file.FullName
        }
    }
    Write-Output "$($run.Name) CLEANED $removed bytes"

    if ($run.Name -eq 'E2') {
        Write-Output "COOLDOWN START $([DateTimeOffset]::Now.ToString('o'))"
        Start-Sleep -Seconds 180
        Write-Output "COOLDOWN END $([DateTimeOffset]::Now.ToString('o'))"
    }
}
