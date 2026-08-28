#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run ctest suites for NovaNN test presets.

.DESCRIPTION
    Runs every *-test-* CMake preset, or the matching subset, and writes the
    complete ctest output to build/logs/tests/<preset>.log. Presets that have
    not been configured are reported as skipped. Test failures are collected
    across the sweep by default; --force-stop-on-failure stops at the first
    failing preset.

    Arguments after -- are passed to ctest. ASan and LSan options are set for
    the child test processes. Windows configures AddressSanitizer options only;
    leak-suppression files are not configured by this runner.

.PARAMETER Filter
    Backend prefix or exact preset name.

.PARAMETER ForceStopOnFailure
    Stop after the first preset reports a failure.

.PARAMETER List
    Print matching test presets and exit.

.EXAMPLE
    scripts/run-tests.ps1 cuda
.EXAMPLE
    scripts/run-tests.ps1 hip -- -R 'Tensor.*' -V
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$ScriptName = Split-Path -Leaf $PSCommandPath
Set-Location -Path $ProjectRoot

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

$LogDir = 'build/logs/tests'
$ForceStop = $false
$ListOnly = $false
$FilterArg = $null
$CtestArgs = @()

function Show-Usage {
@"
$($C_BOLD)Usage:$($C_RESET) $ScriptName [OPTIONS] [FILTER] [-- CTEST_ARGS...]

Run ctest for every *-test-* CMake preset, or the subset matched by FILTER.

$($C_BOLD)Arguments:$($C_RESET)
  FILTER                Backend prefix (cpu, cuda, hip) or an exact preset
                        name (for example cpu-asan-test-debug).

$($C_BOLD)Options:$($C_RESET)
      --force-stop-on-failure
                        Stop at the first preset reporting failures.
  -l, --list            Print matching test presets and exit.
  -h, --help            Show this help and exit.

$($C_BOLD)ctest pass-through:$($C_RESET)
  Everything after -- is passed verbatim to each ctest invocation.

    $ScriptName hip -- -R 'Tensor.*' -V

Full output is written to $($C_BOLD)build/logs/tests/<preset>.log$($C_RESET).
Exit status: 0 = all executed presets passed, 1 = failures found,
2 = usage error.
"@ | Write-Host
}

$i = 0
while ($i -lt $args.Count) {
    $arg = $args[$i]
    switch ($arg) {
        { $_ -in @('-h', '--help') } {
            Show-Usage
            exit 0
        }
        { $_ -in @('-l', '--list') } {
            $ListOnly = $true
        }
        '--force-stop-on-failure' {
            $ForceStop = $true
        }
        '--' {
            if ($i + 1 -lt $args.Count) {
                $CtestArgs = @($args[($i + 1)..($args.Count - 1)])
            }
            break
        }
        default {
            if ($arg.StartsWith('-')) {
                Write-UsageErrorAndExit "unknown option: $arg (pass ctest flags after '--')"
            }
            if ($null -ne $FilterArg) {
                Write-UsageErrorAndExit 'only one FILTER is allowed'
            }
            $FilterArg = $arg
        }
    }
    if ($arg -eq '--') { break }
    $i++
}

$AllPresets = @(Get-NovaPresetNames)
$MatchedPresets = if ($null -ne $FilterArg) {
    @($AllPresets | Where-Object { $_ -eq $FilterArg -or $_ -like "$FilterArg-*" })
}
else {
    @($AllPresets)
}

$Presets = @($MatchedPresets | Where-Object { $_ -like '*-test-*' })
if ($Presets.Count -eq 0) {
    $filterDisplay = if ($null -eq $FilterArg) { '*' } else { $FilterArg }
    Write-Die "no test presets match '$filterDisplay' (see --list)"
}

if ($ListOnly) {
    Write-Host "$($C_BOLD)Matching test presets ($($Presets.Count)):$($C_RESET)"
    foreach ($preset in $Presets) {
        Write-Host "  $($C_CYAN)$preset$($C_RESET)"
    }
    exit 0
}

# Windows ASan: protect_shadow_gap=0 triggers CHECK failure
# sanitizer_common_libcdep.cpp:164 (beg % GetMmapGranularity() == 0).
# On Linux it is required for CUDA probes, but on Windows CUDA tests
# pass without it and discovery crashes with it.
if (-not [string]::IsNullOrEmpty($env:ASAN_OPTIONS)) {
    $env:ASAN_OPTIONS = $env:ASAN_OPTIONS
} else {
    Remove-Item Env:ASAN_OPTIONS -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Write-Host ''
Write-Host "$($C_BOLD)NovaNN — run tests$($C_RESET)"
Write-Host "$($C_DIM)$($Presets.Count) test preset(s) · full logs: $LogDir$($C_RESET)"

$FailedPresets = @()
$Skipped = 0

for ($idx = 0; $idx -lt $Presets.Count; $idx++) {
    $preset = $Presets[$idx]
    $n = $idx + 1
    $logFile = Join-Path $LogDir "$preset.log"

    Write-Host ''
    Write-Host "  $($C_BOLD)[$('{0:D2}' -f $n)/$($Presets.Count)]$($C_RESET) $($C_CYAN)▸ $preset$($C_RESET)"

    if (-not (Test-Path -LiteralPath "build/$preset/CMakeCache.txt" -PathType Leaf)) {
        $Skipped++
        Write-Host "$($C_YELLOW)  ⏭ skipped$($C_RESET)  not configured (run scripts/build-presets.ps1 $preset)"
        continue
    }

    $testList = @(& ctest --preset $preset @CtestArgs -N 2>$null)
    $totalTests = 0
    foreach ($line in $testList) {
        if ($line -match '^\s*Total Tests:\s*(\d+)') {
            $totalTests = [int]$Matches[1]
            break
        }
    }

    $start = Get-Date
    if ($IsTTY) {
        $job = Start-Job -ScriptBlock {
            param($PresetName, $Log, $Dir, $Arguments)
            Set-Location -Path $Dir
            & ctest --preset $PresetName @Arguments *> $Log
            $LASTEXITCODE
        } -ArgumentList @($preset, $logFile, $ProjectRoot, (,$CtestArgs))

        Show-Spinner -Job $job -LogFile $logFile -Label $preset -N $n -Total $Presets.Count `
            -Pattern '(\d+)/(\d+)\s+Test([^\r\n]*)'
        Wait-Job -Job $job | Out-Null
        $received = @(Receive-Job -Job $job)
        Remove-Job -Job $job
        $rc = if ($received.Count -gt 0) { [int]$received[-1] } else { 1 }
    }
    else {
        & ctest --preset $preset @CtestArgs *> $logFile
        $rc = $LASTEXITCODE
    }
    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    $failedNames = @()
    if (Test-Path -LiteralPath $logFile) {
        foreach ($line in Get-Content -LiteralPath $logFile) {
            if ($line -match '^\s*\d+\s+-\s+(.+?)\s+\((?:Failed|Timeout|Subprocess aborted)\)\s*$') {
                $failedNames += $Matches[1]
            }
        }
    }
    $failedCount = $failedNames.Count

    if ($rc -eq 0 -and $failedCount -eq 0) {
        Write-Host "$($C_CLEAR)  $($C_GREEN)✔ passed$($C_RESET)  $($C_BOLD)$($totalTests)/$($totalTests)$($C_RESET)  $($C_DIM)$(Format-Elapsed $elapsed)$($C_RESET)"
    }
    else {
        $FailedPresets += $preset
        $passedCount = [math]::Max(0, $totalTests - $failedCount)
        Write-Host "$($C_CLEAR)  $($C_RED)✘ FAILED$($C_RESET)  $($C_BOLD)$passedCount/$totalTests · $failedCount failed$($C_RESET)  $($C_DIM)$(Format-Elapsed $elapsed)$($C_RESET)"
        [Console]::Error.WriteLine("$($C_YELLOW)  ── failing tests ($logFile) ──$($C_RESET)")
        foreach ($name in $failedNames) {
            [Console]::Error.WriteLine("  $name")
        }

        if ($ForceStop) {
            Write-Host ''
            Write-Host "$($C_RED)Aborting: --force-stop-on-failure and $preset reported failures.$($C_RESET)"
            exit 1
        }
    }
}

Write-Host ''
if ($FailedPresets.Count -eq 0) {
    $message = "✔ All $($Presets.Count) test preset(s) passed"
    if ($Skipped -gt 0) { $message += " ($Skipped skipped: not configured)" }
    Write-Host "$($C_GREEN)${message}.$($C_RESET)"
    exit 0
}

Write-Host "$($C_RED)✘ $($FailedPresets.Count) of $($Presets.Count) test preset(s) failed:$($C_RESET)"
foreach ($preset in $FailedPresets) {
    Write-Host "  $($C_RED)✘$($C_RESET) $preset"
}
Write-Host "Logs: $LogDir/"
exit 1
