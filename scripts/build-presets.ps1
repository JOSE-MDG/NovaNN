#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Configure CMake presets into build directories.

.DESCRIPTION
    Reads every preset defined in CMakePresets.json (or a filtered subset)
    and runs `cmake --preset` for each one, writing full output to
    build/logs/<preset>.log. The terminal shows a single summary line per
    preset with a live progress spinner when stdout is a TTY.

    Without arguments the script configures ALL presets. A single FILTER
    argument restricts the run to presets whose name starts with that
    prefix (e.g. cpu, cuda) or matches an exact preset name.

.PARAMETER Filter
    Backend prefix (cpu, cuda, hip) or an exact preset name.

.PARAMETER Continue
    Keep going after a preset fails.

.PARAMETER List
    Print the matching presets and exit.

.PARAMETER Help
    Show usage and exit.

.NOTES
    Exit status: 0 = all presets configured, 1 = at least one preset
    failed, 2 = usage error.

.EXAMPLE
    scripts/build-presets.ps1
.EXAMPLE
    scripts/build-presets.ps1 cpu
.EXAMPLE
    scripts/build-presets.ps1 --continue cuda

.LINK
    compile-presets.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$ScriptName = Split-Path -Leaf $PSCommandPath
Set-Location -Path $ProjectRoot

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

Import-Module (Join-Path $ScriptDir 'lib/common.psm1') -Force
$NovaColors = Get-NovaColors
$IsTTY = [bool]$NovaColors.IsTTY
$C_RESET = $NovaColors.Reset
$C_BOLD = $NovaColors.Bold
$C_DIM = $NovaColors.Dim
$C_RED = $NovaColors.Red
$C_GREEN = $NovaColors.Green
$C_YELLOW = $NovaColors.Yellow
$C_CYAN = $NovaColors.Cyan
$C_CLEAR = $NovaColors.Clear

$LogDir = 'build/logs'
$ContinueOnError = $false
$ListOnly = $false
$FilterArg = $null

# Print usage information to stdout.
function Show-Usage {
    @"
$($C_BOLD)Usage:$($C_RESET) $($ScriptName) [OPTIONS] [FILTER]

Configure every CMake preset into build/<preset>, one at a time.

$($C_BOLD)Arguments:$($C_RESET)
  FILTER                Backend prefix (cpu, cuda, hip) or an exact preset
                        name (e.g. cpu-asan-debug).

$($C_BOLD)Options:$($C_RESET)
  -c, --continue        Keep going after a preset fails.
  -l, --list            Print the matching presets and exit.
  -h, --help            Show this help and exit.

$($C_BOLD)Examples:$($C_RESET)
  $($ScriptName)                     Configure all presets.
  $($ScriptName) cpu                 Configure the cpu-* presets.
  $($ScriptName) --continue cuda     Configure cuda-* presets, ignoring failures.

Full output is written to $($C_BOLD)build/logs/<preset>.log$($C_RESET); the terminal
only shows one summary line per preset. Exit status: 0 = all configured,
1 = at least one preset failed, 2 = usage error.
"@ | Write-Host
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

$i = 0
while ($i -lt $args.Count) {
    $arg = $args[$i]
    switch ($arg) {
        { $_ -in @('-h', '--help') } {
            Show-Usage
            exit 0
        }
        { $_ -in @('-c', '--continue') } {
            $ContinueOnError = $true
        }
        { $_ -in @('-l', '--list') } {
            $ListOnly = $true
        }
        default {
            if ($arg.StartsWith('-')) {
                Write-UsageErrorAndExit "unknown option: $arg"
            }
            if ($FilterArg) {
                Write-UsageErrorAndExit "only one FILTER is allowed"
            }
            $FilterArg = $arg
        }
    }
    $i++
}

# ---------------------------------------------------------------------------
# Preset discovery / filtering
# ---------------------------------------------------------------------------

$Presets = @(Get-NovaPresetNames)

if ($Presets.Count -eq 0) {
    Write-Die "no CMake presets found — is CMakePresets.json present?"
}

if ($FilterArg) {
    $matched = @($Presets | Where-Object { $_ -eq $FilterArg -or $_ -like "$FilterArg-*" })
    if ($matched.Count -eq 0) {
        Write-Die "no presets match '$FilterArg' (see --list)"
    }
    $Presets = $matched
}

$TotalCount = $Presets.Count

if ($ListOnly) {
    Write-Host "$($C_BOLD)Matching presets ($($TotalCount)):$($C_RESET)"
    foreach ($p in $Presets) {
        Write-Host "  $($C_CYAN)$($p)$($C_RESET)"
    }
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ''
Write-Host "$($C_BOLD)NovaNN — configure presets$($C_RESET)"
Write-Host "$($C_DIM)$($TotalCount) preset(s) → build/<preset>  ·  full logs: $($LogDir)$($C_RESET)"

$Failed = 0
$Skipped = 0

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for ($idx = 0; $idx -lt $Presets.Count; $idx++) {
    $preset = $Presets[$idx]
    $n = $idx + 1
    $logFile = Join-Path $LogDir "$preset.log"

    $nStr = $n.ToString().PadLeft(2)
    Write-Host ''
    Write-Host "  $($C_BOLD)[$($nStr)/$($TotalCount)]$($C_RESET) $($C_CYAN)▸ $($preset)$($C_RESET)"

    $start = Get-Date

    if ($IsWindows -and $preset -like 'cuda-*') {
        if (-not $env:CUDA_HOST_COMPILER) {
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
    }

    if ($IsTTY) {
        $job = Start-Job -ScriptBlock {
            param($PresetName, $Log, $Dir)
            Set-Location -Path $Dir
            & cmake --preset $PresetName *> $Log
            return $LASTEXITCODE
        } -ArgumentList $preset, $logFile, $ProjectRoot

        Show-Spinner -Job $job -LogFile $logFile -Label $preset -N $n -Total $TotalCount
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
        Write-Host "$($C_CLEAR)  $($C_GREEN)✔ configured$($C_RESET)  $($C_DIM)→$($C_RESET) build/$($preset)  $($C_DIM)($(Format-Elapsed $elapsed))$($C_RESET)"
    }
    else {
        $Failed++
        Write-Host "$($C_CLEAR)  $($C_RED)✘ FAILED$($C_RESET)  → build/$($preset)  $($C_DIM)($(Format-Elapsed $elapsed))$($C_RESET)"
        [Console]::Error.WriteLine("$($C_YELLOW)  ── last lines of $($LogDir)/$($preset).log ──$($C_RESET)")
        if (Test-Path -LiteralPath $logFile) {
            Get-Content -LiteralPath $logFile -Tail 15 | ForEach-Object {
                [Console]::Error.WriteLine("  $($C_DIM)$($_)$($C_RESET)")
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
    Write-Host "$($C_RED)✘ $($Failed) failed, $($TotalCount - $Failed - $Skipped) configured, $($Skipped) skipped — see $($LogDir)/$($C_RESET)"
    exit 1
}
elseif ($Skipped -gt 0 -and ($TotalCount - $Skipped) -eq 0) {
    Write-Host "$($C_YELLOW)⚠ nothing configured — $($Skipped) preset(s) skipped (CUDA_HOST_COMPILER not set)$($C_RESET)"
    exit 1
}
elseif ($Skipped -gt 0) {
    Write-Host "$($C_GREEN)✔ $($TotalCount - $Skipped) configured, $($Skipped) skipped (CUDA_HOST_COMPILER not set)$($C_RESET)"
}
else {
    Write-Host "$($C_GREEN)✔ All $($TotalCount) preset(s) configured successfully.$($C_RESET)"
}
