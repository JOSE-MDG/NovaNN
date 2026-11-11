#!/usr/bin/env pwsh
# Build the CMake presets that are already configured under build/<preset>/.

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
$Config = $null
$Jobs = $null
$Filters = @()

function Show-Usage {
    @"
$($C_BOLD)Usage:$($C_RESET) $ScriptName [OPTIONS] [FILTER...]

Build every CMake preset already configured in build/<preset>, one at a time.

$($C_BOLD)Arguments:$($C_RESET)
  FILTER...             Backend prefix (cpu, cuda, hip) or an exact preset
                        name (e.g. cpu-asan-debug). Repeat to select several.

$($C_BOLD)Options:$($C_RESET)
  -C, --config MODE     Build configuration: Release or Debug.
                        Default: derived from the preset name
                        (*-debug* → Debug, everything else → Release).
  -j, --jobs N          Run the build with N parallel jobs.
  -c, --continue        Keep going after a preset fails.
  -l, --list            Print the matching presets and exit.
  -h, --help            Show this help and exit.

$($C_BOLD)Examples:$($C_RESET)
  $ScriptName                     Build all presets.
  $ScriptName cpu                 Build the cpu-* presets.
  $ScriptName --config Debug cpu-debug
  $ScriptName -j 16 cuda --continue

Full output is appended to $($C_BOLD)build/logs/<preset>.log$($C_RESET). Presets without
a configured build directory are skipped with a warning. Exit status:
0 = all built, 1 = at least one preset failed or nothing was built,
2 = usage error.
"@ | Write-Host
}

for ($i = 0; $i -lt $args.Count; $i++) {
    $arg = $args[$i]

    if ($arg -in '-h', '--help') { Show-Usage; exit 0 }
    elseif ($arg -in '-c', '--continue') { $ContinueOnError = $true }
    elseif ($arg -in '-l', '--list') { $ListOnly = $true }
    elseif ($arg -in '-C', '--config') {
        if ($i + 1 -ge $args.Count) { Write-UsageErrorAndExit "option '$arg' requires a value" }
        $i++
        $Config = $args[$i]
        if ($Config -notin 'Release', 'Debug') {
            Write-UsageErrorAndExit "invalid --config '$Config' (expected Release or Debug)"
        }
    }
    elseif ($arg -in '-j', '--jobs') {
        if ($i + 1 -ge $args.Count) { Write-UsageErrorAndExit "option '$arg' requires a value" }
        $i++
        $Jobs = $args[$i]
        if ($Jobs -notmatch '^[1-9][0-9]*$') {
            Write-UsageErrorAndExit "invalid --jobs '$Jobs' (expected a positive integer)"
        }
    }
    elseif ($arg.StartsWith('-')) { Write-UsageErrorAndExit "unknown option: $arg" }
    else { $Filters += $arg }
}

$Presets = @(Get-PresetNames)
if ($Filters.Count -gt 0) {
    $Presets = @($Presets | Select-Presets -Filter $Filters)
    if ($Presets.Count -eq 0) {
        Write-Die "no presets match '$($Filters -join ' ')' (see --list)"
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

# Keep the pinned vcpkg for the build and for any reconfigure the build triggers.
Import-Module (Join-Path $ScriptDir 'lib/vcpkg-env.psm1') -Force
Set-VcpkgEnvironment

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ''
Write-Host "$($C_BOLD)NovaNN — compile presets$($C_RESET)"
Write-Host "$($C_DIM)$Total preset(s) → cmake --build build/<preset>  ·  full logs: $LogDir$($C_RESET)"
if ($Jobs) {
    Write-Host "$($C_DIM)parallel jobs: $Jobs$($C_RESET)"
}

$Failed = 0
$Skipped = 0
$Built = 0

for ($i = 0; $i -lt $Presets.Count; $i++) {
    $preset = $Presets[$i]
    $n = $i + 1
    $logFile = Join-Path $LogDir "$preset.log"

    $presetConfig = $Config
    if (-not $presetConfig) {
        $presetConfig = if ($preset -like '*-debug*') { 'Debug' } else { 'Release' }
    }

    Write-Host ''
    Write-Host "  $($C_BOLD)[$('{0,2}' -f $n)/$Total]$($C_RESET) $($C_CYAN)▸ $preset$($C_RESET)  $($C_DIM)($presetConfig)$($C_RESET)"

    $buildDir = "build/$preset"
    if (-not (Test-Path -LiteralPath $buildDir -PathType Container)) {
        Write-Host "  $($C_YELLOW)⚠ not configured — run scripts/build-presets.ps1 $preset first$($C_RESET)"
        $Skipped++
        continue
    }

    $cmakeArgs = @('--build', $buildDir, '--config', $presetConfig)
    if ($Jobs) { $cmakeArgs += @('--parallel', $Jobs) }

    $start = Get-Date

    if ($IsTTY) {
        $job = Start-Job -ScriptBlock {
            param($BuildArgs, $Log, $Dir)
            Set-Location -Path $Dir
            & cmake @BuildArgs *>> $Log
            $LASTEXITCODE
        } -ArgumentList $cmakeArgs, $logFile, $ProjectRoot

        Show-Spinner -Job $job -LogFile $logFile -Label $preset -N $n -Total $Total
        Wait-Job -Job $job | Out-Null
        $rc = Receive-Job -Job $job
        Remove-Job -Job $job
        if ($null -eq $rc) { $rc = 1 }
    }
    else {
        & cmake @cmakeArgs *>> $logFile
        $rc = $LASTEXITCODE
    }

    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    if ($rc -eq 0) {
        $Built++
        Write-Host "$($C_CLEAR)  $($C_GREEN)✔ built$($C_RESET)  $($C_DIM)→$($C_RESET) build/$preset  $($C_DIM)($(Format-Elapsed $elapsed))$($C_RESET)"
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
    Write-Host "$($C_RED)✘ $Failed failed, $Built built, $Skipped skipped — see $LogDir/$($C_RESET)"
    exit 1
}
elseif ($Built -eq 0 -and $Skipped -gt 0) {
    Write-Host "$($C_YELLOW)⚠ nothing built — $Skipped preset(s) not configured yet$($C_RESET)"
    Write-Host "$($C_DIM)  Run scripts/build-presets.ps1 to configure them first.$($C_RESET)"
    exit 1
}
elseif ($Skipped -gt 0) {
    Write-Host "$($C_GREEN)✔ $Built built, $Skipped skipped (not configured)$($C_RESET)"
}
else {
    Write-Host "$($C_GREEN)✔ All $Total preset(s) built successfully.$($C_RESET)"
}
