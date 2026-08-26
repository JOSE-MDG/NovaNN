#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build CMake presets that have already been configured.

.DESCRIPTION
    Iterates over CMake presets and runs `cmake --build` for each one that
    has a configured build directory under build/<preset>/. Full compiler
    output is appended to build/logs/<preset>.log; the terminal shows a
    single summary line per preset with a live progress spinner when
    stdout is a TTY.

    Presets without a configured build directory are skipped with a
    warning. The build configuration (Release/Debug) is derived from the
    preset name by default (*-debug* -> Debug, everything else -> Release)
    but can be overridden with -Config.

.PARAMETER Filters
    Backend prefix (cpu, cuda, hip) or an exact preset name. Repeat to
    select several.

.PARAMETER Config
    Build configuration: Release or Debug.

.PARAMETER Jobs
    Run the build with N parallel jobs.

.PARAMETER Continue
    Keep going after a preset fails.

.PARAMETER List
    Print the matching presets and exit.

.PARAMETER Help
    Show usage and exit.

.NOTES
    Exit status: 0 = all presets built (some may have been skipped),
    1 = at least one preset failed or nothing was built, 2 = usage error.

.EXAMPLE
    scripts/compile-presets.ps1
.EXAMPLE
    scripts/compile-presets.ps1 cpu
.EXAMPLE
    scripts/compile-presets.ps1 --config Debug cpu-debug
.EXAMPLE
    scripts/compile-presets.ps1 -j 16 cuda --continue

.LINK
    build-presets.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

$ScriptDir   = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$ScriptName  = Split-Path -Leaf $PSCommandPath
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

$LogDir          = 'build/logs'
$ContinueOnError = $false
$ListOnly        = $false
$Config          = $null
$Jobs            = $null
$Filters         = @()

# Print usage information to stdout.
function Show-Usage {
    @"
$($C_BOLD)Usage:$($C_RESET) $($ScriptName) [OPTIONS] [FILTER...]

Build every CMake preset already configured in build/<preset>, one at a time.

$($C_BOLD)Arguments:$($C_RESET)
  FILTER...              Backend prefix (cpu, cuda, hip) or an exact preset
                        name (e.g. cpu-asan-debug). Repeat to select several.

$($C_BOLD)Options:$($C_RESET)
  -C, --config MODE      Build configuration: Release or Debug.
                        Default: derived from the preset name
                        (*-debug* → Debug, everything else → Release).
  -j, --jobs N           Run the build with N parallel jobs.
  -c, --continue         Keep going after a preset fails.
  -l, --list             Print the matching presets and exit.
  -h, --help             Show this help and exit.

$($C_BOLD)Examples:$($C_RESET)
  $($ScriptName)                     Build all presets.
  $($ScriptName) cpu                 Build the cpu-* presets.
  $($ScriptName) --config Debug cpu-debug
  $($ScriptName) -j 16 cuda --continue

Full output is appended to $($C_BOLD)build/logs/<preset>.log$($C_RESET); the terminal
only shows one summary line per preset. Presets without a configured build
directory are skipped with a warning. Exit status: 0 = all built,
1 = at least one preset failed or nothing was built, 2 = usage error.
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
        { $_ -in @('-C', '--config') } {
            if ($i + 1 -ge $args.Count) {
                Write-UsageErrorAndExit "option '$arg' requires a value"
            }
            $i++
            $val = $args[$i]
            if ($val -notin @('Release', 'Debug')) {
                Write-UsageErrorAndExit "invalid --config '$val' (expected Release or Debug)"
            }
            $Config = $val
        }
        { $_ -in @('-j', '--jobs') } {
            if ($i + 1 -ge $args.Count) {
                Write-UsageErrorAndExit "option '$arg' requires a value"
            }
            $i++
            $val = $args[$i]
            if ($val -notmatch '^[1-9][0-9]*$') {
                Write-UsageErrorAndExit "invalid --jobs '$val' (expected a positive integer)"
            }
            $Jobs = [int]$val
        }
        default {
            if ($arg.StartsWith('-')) {
                Write-UsageErrorAndExit "unknown option: $arg"
            }
            $Filters += $arg
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

if ($Filters.Count -gt 0) {
    $matchedList = New-Object System.Collections.Generic.List[string]
    foreach ($preset in $Presets) {
        foreach ($filter in $Filters) {
            if ($preset -eq $filter -or $preset -like "$filter-*") {
                if (-not $matchedList.Contains($preset)) {
                    $matchedList.Add($preset)
                }
                break
            }
        }
    }
    if ($matchedList.Count -eq 0) {
        Write-Die "no presets match '$($Filters -join ' ')' (see --list)"
    }
    $Presets = @($matchedList)
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
Write-Host "$($C_BOLD)NovaNN — compile presets$($C_RESET)"
Write-Host "$($C_DIM)$($TotalCount) preset(s) → cmake --build build/<preset>  ·  full logs: $($LogDir)$($C_RESET)"
if ($Jobs) {
    Write-Host "$($C_DIM)parallel jobs: $($Jobs)$($C_RESET)"
}

$Failed = 0
$Skipped = 0
$Built = 0

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for ($idx = 0; $idx -lt $Presets.Count; $idx++) {
    $preset = $Presets[$idx]
    $n = $idx + 1
    $logFile = Join-Path $LogDir "$preset.log"

    $presetConfig = $Config
    if (-not $presetConfig) {
        if ($preset -like '*-debug*') {
            $presetConfig = 'Debug'
        }
        else {
            $presetConfig = 'Release'
        }
    }

    $nStr = $n.ToString().PadLeft(2)
    Write-Host ''
    Write-Host "  $($C_BOLD)[$($nStr)/$($TotalCount)]$($C_RESET) $($C_CYAN)▸ $($preset)$($C_RESET)  $($C_DIM)($($presetConfig))$($C_RESET)"

    $buildDir = "build/$preset"
    if (-not (Test-Path -LiteralPath $buildDir -PathType Container)) {
        Write-Host "  $($C_YELLOW)⚠ not configured — run scripts/build-presets.ps1 $($preset) first$($C_RESET)"
        $Skipped++
        continue
    }

    $start = Get-Date

    if ($IsTTY) {
        $job = Start-Job -ScriptBlock {
            param($BuildDir, $Cfg, $JobCount, $Log, $Dir)
            Set-Location -Path $Dir
            $cmdArgs = @('--build', $BuildDir, '--config', $Cfg)
            if ($JobCount) { $cmdArgs += @('--parallel', $JobCount) }
            & cmake @cmdArgs *>> $Log
            return $LASTEXITCODE
        } -ArgumentList $buildDir, $presetConfig, $Jobs, $logFile, $ProjectRoot

        Show-Spinner -Job $job -LogFile $logFile -Label $preset -N $n -Total $TotalCount
        Wait-Job -Job $job | Out-Null
        $rc = Receive-Job -Job $job
        Remove-Job -Job $job
        if ($null -eq $rc) { $rc = 1 }
    }
    else {
        $cmdArgs = @('--build', $buildDir, '--config', $presetConfig)
        if ($Jobs) { $cmdArgs += @('--parallel', $Jobs) }
        & cmake @cmdArgs *>> $logFile
        $rc = $LASTEXITCODE
    }

    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    if ($rc -eq 0) {
        $Built++
        Write-Host "$($C_CLEAR)  $($C_GREEN)✔ built$($C_RESET)  $($C_DIM)→$($C_RESET) build/$($preset)  $($C_DIM)($(Format-Elapsed $elapsed))$($C_RESET)"
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
    Write-Host "$($C_RED)✘ $($Failed) failed, $($Built) built, $($Skipped) skipped — see $($LogDir)/$($C_RESET)"
    exit 1
}
elseif ($Built -eq 0 -and $Skipped -gt 0) {
    Write-Host "$($C_YELLOW)⚠ nothing built — $($Skipped) preset(s) not configured yet$($C_RESET)"
    Write-Host "$($C_DIM)  Run scripts/build-presets.ps1 to configure them first.$($C_RESET)"
    exit 1
}
elseif ($Skipped -gt 0) {
    Write-Host "$($C_GREEN)✔ $($Built) built, $($Skipped) skipped (not configured)$($C_RESET)"
}
else {
    Write-Host "$($C_GREEN)✔ All $($TotalCount) preset(s) built successfully.$($C_RESET)"
}
