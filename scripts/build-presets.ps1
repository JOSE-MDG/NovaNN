#!/usr/bin/env pwsh
# Configure every CMake preset (or a filtered subset) into build/<preset>/.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$ScriptName = Split-Path -Leaf $PSCommandPath

Import-Module (Join-Path $ScriptDir 'lib/common.psm1') -Force
$ProjectRoot = Initialize-ScriptContext

$Colors = Get-ScriptColors
$IsTTY = [bool]$Colors.IsTTY
$C_RESET = $Colors.Reset
$C_BOLD = $Colors.Bold
$C_DIM = $Colors.Dim
$C_RED = $Colors.Red
$C_GREEN = $Colors.Green
$C_YELLOW = $Colors.Yellow
$C_CYAN = $Colors.Cyan
$C_CLEAR = $Colors.Clear

$LogDir = 'build/logs'
$ContinueOnError = $false
$ListOnly = $false
$FilterArg = $null

function Show-Usage {
    @"
$($C_BOLD)Usage:$($C_RESET) $ScriptName [OPTIONS] [FILTER]

Configure every CMake preset into build/<preset>, one at a time.

$($C_BOLD)Arguments:$($C_RESET)
  FILTER                Backend prefix (cpu, cuda, hip) or an exact preset
                        name (e.g. cpu-asan-debug).

$($C_BOLD)Options:$($C_RESET)
  -c, --continue        Keep going after a preset fails.
  -l, --list            Print the matching presets and exit.
  -h, --help            Show this help and exit.

$($C_BOLD)Examples:$($C_RESET)
  $ScriptName                     Configure all presets.
  $ScriptName cpu                 Configure the cpu-* presets.
  $ScriptName --continue cuda     Configure cuda-* presets, ignoring failures.

Full output is written to $($C_BOLD)build/logs/<preset>.log$($C_RESET); the terminal
only shows one summary line per preset. Exit status: 0 = all configured,
1 = at least one preset failed, 2 = usage error.
"@ | Write-Host
}

foreach ($arg in $args) {
    if ($arg -in '-h', '--help') { Show-Usage; exit 0 }
    elseif ($arg -in '-c', '--continue') { $ContinueOnError = $true }
    elseif ($arg -in '-l', '--list') { $ListOnly = $true }
    elseif ($arg.StartsWith('-')) { Write-UsageErrorAndExit "unknown option: $arg" }
    elseif ($FilterArg) { Write-UsageErrorAndExit 'only one FILTER is allowed' }
    else { $FilterArg = $arg }
}

$Presets = @(Get-PresetNames)
if ($FilterArg) {
    $Presets = @($Presets | Select-Presets -Filter $FilterArg)
    if ($Presets.Count -eq 0) {
        Write-Die "no presets match '$FilterArg' (see --list)"
    }
}

$Total = $Presets.Count

if ($ListOnly) {
    Write-Host "$($C_BOLD)Matching presets ($Total):$($C_RESET)"
    foreach ($preset in $Presets) {
        Write-Host "  $($C_CYAN)$preset$($C_RESET)"
    }
    exit 0
}

# Prefer NovaNN's pinned vcpkg for this build and all child processes.
Import-Module (Join-Path $ScriptDir 'lib/vcpkg-env.psm1') -Force
Set-VcpkgEnvironment

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ''
Write-Host "$($C_BOLD)NovaNN — configure presets$($C_RESET)"
Write-Host "$($C_DIM)$Total preset(s) → build/<preset>  ·  full logs: $LogDir$($C_RESET)"

$Failed = 0
$Skipped = 0

for ($i = 0; $i -lt $Presets.Count; $i++) {
    $preset = $Presets[$i]
    $n = $i + 1
    $logFile = Join-Path $LogDir "$preset.log"

    Write-Host ''
    Write-Host "  $($C_BOLD)[$('{0,2}' -f $n)/$Total]$($C_RESET) $($C_CYAN)▸ $preset$($C_RESET)"

    # nvcc needs an MSVC host compiler on Windows and fails without it for a
    # reason that hides the real setup error.
    if ($IsWindows -and $preset -like 'cuda-*' -and -not $env:CUDA_HOST_COMPILER) {
        Write-Host "$($C_CLEAR)  $($C_YELLOW)⏭ skipped$($C_RESET)  CUDA_HOST_COMPILER not set"
        $Skipped++
        if (-not $ContinueOnError) {
            Write-Host ''
            Write-Host "$($C_RED)Aborting: CUDA_HOST_COMPILER must be exported before configuring CUDA presets on Windows.$($C_RESET)"
            Write-Host "$($C_DIM)  Example: `$env:CUDA_HOST_COMPILER='C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe'$($C_RESET)"
            exit 1
        }
        continue
    }

    $start = Get-Date

    if ($IsTTY) {
        $job = Start-Job -ScriptBlock {
            param($PresetName, $Log, $Dir)
            Set-Location -Path $Dir
            & cmake --preset $PresetName *> $Log
            $LASTEXITCODE
        } -ArgumentList $preset, $logFile, $ProjectRoot

        Show-Spinner -Job $job -LogFile $logFile -Label $preset -N $n -Total $Total
        Wait-Job -Job $job | Out-Null
        $rc = Receive-Job -Job $job
        Remove-Job -Job $job
        if ($null -eq $rc) { $rc = 1 }
    }
    else {
        & cmake --preset $preset *> $logFile
        $rc = $LASTEXITCODE
    }

    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    if ($rc -eq 0) {
        Write-Host "$($C_CLEAR)  $($C_GREEN)✔ configured$($C_RESET)  $($C_DIM)→$($C_RESET) build/$preset  $($C_DIM)($(Format-Elapsed $elapsed))$($C_RESET)"
    }
    else {
        $Failed++
        Write-Host "$($C_CLEAR)  $($C_RED)✘ FAILED$($C_RESET)  → build/$preset  $($C_DIM)($(Format-Elapsed $elapsed))$($C_RESET)"
        [Console]::Error.WriteLine("$($C_YELLOW)  ── last lines of $LogDir/$preset.log ──$($C_RESET)")
        if (Test-Path -LiteralPath $logFile) {
            Get-Content -LiteralPath $logFile -Tail 15 | ForEach-Object {
                [Console]::Error.WriteLine("  $($C_DIM)$_$($C_RESET)")
            }
        }
        if (-not $ContinueOnError) {
            Write-Host ''
            Write-Host "$($C_RED)Aborting: first failure (use --continue to skip failures).$($C_RESET)"
            exit 1
        }
    }
}

Write-Host ''
if ($Failed -gt 0) {
    Write-Host "$($C_RED)✘ $Failed failed, $($Total - $Failed - $Skipped) configured, $Skipped skipped — see $LogDir/$($C_RESET)"
    exit 1
}
elseif ($Skipped -eq $Total) {
    Write-Host "$($C_YELLOW)⚠ nothing configured — $Skipped preset(s) skipped (CUDA_HOST_COMPILER not set)$($C_RESET)"
    exit 1
}
elseif ($Skipped -gt 0) {
    Write-Host "$($C_GREEN)✔ $($Total - $Skipped) configured, $Skipped skipped (CUDA_HOST_COMPILER not set)$($C_RESET)"
}
else {
    Write-Host "$($C_GREEN)✔ All $Total preset(s) configured successfully.$($C_RESET)"
}
