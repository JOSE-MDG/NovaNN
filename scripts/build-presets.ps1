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

$ScriptDir   = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$ScriptName  = Split-Path -Leaf $PSCommandPath
Set-Location -Path $ProjectRoot

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

$IsTTY = -not [Console]::IsOutputRedirected

if ($IsTTY -and -not $env:NO_COLOR -and $env:TERM -ne 'dumb') {
    $ESC     = [char]27
    $C_RESET = "$ESC[0m"
    $C_BOLD  = "$ESC[1m"
    $C_DIM   = "$ESC[2m"
    $C_RED   = "$ESC[31m"
    $C_GREEN = "$ESC[32m"
    $C_YELLOW= "$ESC[33m"
    $C_CYAN  = "$ESC[36m"
}
else {
    $C_RESET = ''; $C_BOLD = ''; $C_DIM = ''; $C_RED = ''
    $C_GREEN = ''; $C_YELLOW = ''; $C_CYAN = ''
}

$C_CLEAR = ''
if ($IsTTY) { $C_CLEAR = "$([char]13)$([char]27)[K" }

$LogDir            = 'build/logs'
$ContinueOnError   = $false
$ListOnly          = $false
$FilterArg         = $null

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Print a formatted error message and exit with status 1.
function Write-Die {
    param([string]$Message)
    [Console]::Error.WriteLine("$($C_RED)ERROR:$($C_RESET) $Message")
    exit 1
}

# Print a usage error message and exit with status 2.
function Write-UsageErrorAndExit {
    param([string]$Message)
    [Console]::Error.WriteLine("$($C_RED)Usage error:$($C_RESET) $Message")
    [Console]::Error.WriteLine("Run $($C_BOLD)--help$($C_RESET) for usage.")
    exit 2
}

# Format elapsed seconds into a human-readable string such as "5s" or "2m 03s".
function Format-Elapsed {
    param([int]$Seconds)
    $m = [math]::Floor($Seconds / 60)
    $s = $Seconds % 60
    if ($m -gt 0) {
        return ('{0}m {1:D2}s' -f $m, $s)
    }
    return ('{0}s' -f $s)
}

# Display a live progress spinner while a background job runs. Parses the
# build log for "[current/total]" progress markers and renders a percentage
# bar with a braille or ASCII spinner. Falls back to a simple label when no
# progress fraction is found.
function Show-Spinner {
    param(
        [System.Management.Automation.Job]$Job,
        [string]$LogFile,
        [string]$Label,
        [int]$N,
        [int]$Total
    )

    $locale = $env:LC_ALL
    if (-not $locale) { $locale = $env:LC_CTYPE }
    if (-not $locale) { $locale = $env:LANG }

    if ($locale -and $locale -match '(?i)utf-?8') {
        $frames  = @('⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏')
        $barFull = '█'
        $barEmpty = '░'
    }
    else {
        $frames  = @('|', '/', '-', '\')
        $barFull = '#'
        $barEmpty = '-'
    }

    $i = 0
    $start = Get-Date

    while ($Job.State -eq 'Running') {
        $elapsed = [int]((Get-Date) - $start).TotalSeconds
        if ($elapsed -lt 0) { $elapsed = 0 }

        $pct = -1
        $target = $null

        if (Test-Path -LiteralPath $LogFile) {
            $tail = Get-Content -LiteralPath $LogFile -Tail 60 -ErrorAction SilentlyContinue
            if ($tail) {
                $joined = $tail -join "`n"
                $matches = [regex]::Matches($joined, '\[(\d+)/(\d+)\]([^\r\n]*)')
                if ($matches.Count -gt 0) {
                    $last = $matches[$matches.Count - 1]
                    $num = [int]$last.Groups[1].Value
                    $denom = [int]$last.Groups[2].Value
                    if ($denom -gt 0) {
                        $pct = [int]($num * 100 / $denom)
                    }
                    $target = $last.Groups[3].Value.Trim()
                    if (-not $target) { $target = $Label }
                    if ($target.Length -gt 40) {
                        $target = $target.Substring(0, 37) + '...'
                    }
                }
            }
        }

        if ($pct -ge 0) {
            $filled = [int]($pct * 20 / 100)
            $bar = ($barFull * $filled) + ($barEmpty * (20 - $filled))
            $pctStr = $pct.ToString().PadLeft(3)
            $line = "`r  $($frames[$i]) $($C_CYAN)$($pctStr)%$($C_RESET) " +
                    "$($C_CYAN)[$($bar)]$($C_RESET) $($C_DIM)$($target)$($C_RESET) " +
                    "· $($C_DIM)[$($N)/$($Total)]$($C_RESET) · $($C_DIM)$(Format-Elapsed $elapsed)$($C_RESET)"
        }
        else {
            $line = "`r  $($frames[$i]) $($C_BOLD)$($Label)$($C_RESET) " +
                    "· $($C_DIM)[$($N)/$($Total)]$($C_RESET) · $($C_DIM)$(Format-Elapsed $elapsed)$($C_RESET)"
        }

        Write-Host -NoNewline $line
        Start-Sleep -Milliseconds 100
        $i = ($i + 1) % $frames.Count
    }
}

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

$Presets = @()
$listOutput = & cmake --list-presets 2>$null
foreach ($line in $listOutput) {
    if ($line -match '^\s*"([^"]+)"') {
        $Presets += $Matches[1]
    }
}

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
if ($Failed -eq 0) {
    Write-Host "$($C_GREEN)✔ All $($TotalCount) preset(s) configured successfully.$($C_RESET)"
}
else {
    Write-Host "$($C_RED)✘ $($Failed) of $($TotalCount) preset(s) failed — see $($LogDir)/$($C_RESET)"
    exit 1
}
