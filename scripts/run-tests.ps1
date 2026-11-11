#!/usr/bin/env pwsh
# Run ctest for every *-test-* preset (or the subset matched by FILTER).

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

for ($i = 0; $i -lt $args.Count; $i++) {
    $arg = $args[$i]

    if ($arg -in '-h', '--help') { Show-Usage; exit 0 }
    elseif ($arg -in '-l', '--list') { $ListOnly = $true }
    elseif ($arg -eq '--force-stop-on-failure') { $ForceStop = $true }
    elseif ($arg -eq '--') {
        $CtestArgs = @($args | Select-Object -Skip ($i + 1))
        break
    }
    elseif ($arg.StartsWith('-')) {
        Write-UsageErrorAndExit "unknown option: $arg (pass ctest flags after '--')"
    }
    elseif ($FilterArg) { Write-UsageErrorAndExit 'only one FILTER is allowed' }
    else { $FilterArg = $arg }
}

$Presets = @(Get-PresetNames)
if ($FilterArg) {
    $Presets = @($Presets | Select-Presets -Filter $FilterArg)
}
$Presets = @($Presets | Where-Object { $_ -like '*-test-*' })

if ($Presets.Count -eq 0) {
    $label = if ($FilterArg) { $FilterArg } else { '*' }
    Write-Die "no test presets match '$label' (see --list)"
}

if ($ListOnly) {
    Write-Host "$($C_BOLD)Matching test presets ($($Presets.Count)):$($C_RESET)"
    foreach ($preset in $Presets) {
        Write-Host "  $($C_CYAN)$preset$($C_RESET)"
    }
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ''
Write-Host "$($C_BOLD)NovaNN — run tests$($C_RESET)"
Write-Host "$($C_DIM)$($Presets.Count) test preset(s) · full logs: $LogDir$($C_RESET)"

$FailedPresets = @()
$Skipped = 0

for ($i = 0; $i -lt $Presets.Count; $i++) {
    $preset = $Presets[$i]
    $n = $i + 1
    $logFile = Join-Path $LogDir "$preset.log"

    Write-Host ''
    Write-Host "  $($C_BOLD)[$('{0:D2}' -f $n)/$($Presets.Count)]$($C_RESET) $($C_CYAN)▸ $preset$($C_RESET)"

    if (-not (Test-Path -LiteralPath "build/$preset/CMakeCache.txt" -PathType Leaf)) {
        $Skipped++
        Write-Host "$($C_YELLOW)  ⏭ skipped$($C_RESET)  not configured (run scripts/build-presets.ps1 $preset)"
        continue
    }

    $totalTests = 0
    foreach ($line in & ctest --preset $preset @CtestArgs -N 2>$null) {
        if ($line -match 'Total Tests:\s*(\d+)') {
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
        } -ArgumentList $preset, $logFile, $ProjectRoot, $CtestArgs

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
        Write-Host "$($C_CLEAR)  $($C_GREEN)✔ passed$($C_RESET)  $($C_BOLD)$totalTests/$totalTests$($C_RESET)  $($C_DIM)$(Format-Elapsed $elapsed)$($C_RESET)"
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
