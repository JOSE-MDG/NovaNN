#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Clean NovaNN build artifacts and logs.

.DESCRIPTION
    Without --target, removes the complete build directory after an
    interactive confirmation. A directory target runs the CMake clean target,
    preserving the configured build tree. The log targets remove all logs,
    test logs, or configure/build logs respectively.

.EXAMPLE
    ./clean.ps1 --target cuda-test-debug-linux
.EXAMPLE
    ./clean.ps1 --dry-run --target logs
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$ProjectRoot = $ScriptDir
$ScriptName = Split-Path -Leaf $PSCommandPath
Set-Location -Path $ProjectRoot

Import-Module (Join-Path $ProjectRoot 'scripts/lib/common.psm1') -Force
$NovaColors = Get-NovaColors
$C_RESET = $NovaColors.Reset
$C_BOLD = $NovaColors.Bold
$C_DIM = $NovaColors.Dim
$C_RED = $NovaColors.Red
$C_GREEN = $NovaColors.Green
$C_YELLOW = $NovaColors.Yellow
$C_CYAN = $NovaColors.Cyan

$DryRun = $false
$AssumeYes = $false
$Target = $null

function Show-Usage {
    @"
$($C_BOLD)Usage:$($C_RESET) $ScriptName [OPTIONS] [--target <path|logs|test-logs|build-logs>]

$($C_BOLD)Targets:$($C_RESET)
  <path>                Run cmake --build <path> --target clean.
                        Configuration is kept. A bare preset name resolves
                        to build/<preset> when that directory exists.
  logs                  Delete build/logs entirely.
  test-logs             Delete build/logs/tests only.
  build-logs            Delete configure/build logs at build/logs/*.log.
  (no --target)         Delete the entire build directory.

$($C_BOLD)Options:$($C_RESET)
  -n, --dry-run         Print actions without touching anything.
  -y, --yes             Assume yes for whole-build deletion.
  -h, --help            Show this help and exit.

Exit status: 0 = cleaned or nothing to do, 1 = refused or failed,
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
        { $_ -in @('-n', '--dry-run') } {
            $DryRun = $true
        }
        { $_ -in @('-y', '--yes') } {
            $AssumeYes = $true
        }
        '--target' {
            if ($i + 1 -ge $args.Count) {
                Write-UsageErrorAndExit '--target requires a value'
            }
            if ($null -ne $Target) {
                Write-UsageErrorAndExit '--target given more than once'
            }
            $i++
            $Target = $args[$i]
            if ([string]::IsNullOrEmpty($Target)) {
                Write-UsageErrorAndExit '--target requires a non-empty value'
            }
        }
        default {
            Write-UsageErrorAndExit "unknown option: $arg"
        }
    }
    $i++
}

function Remove-NovaItem {
    param(
        [Parameter(Mandatory)][string]$Path,
        [switch]$Recurse
    )

    if ($DryRun) {
        $suffix = if ($Recurse) { ' -Recurse' } else { '' }
        Write-Host "$($C_CYAN)[dry-run]$($C_RESET) Remove-Item -LiteralPath '$Path'$suffix -Force"
        return
    }
    Remove-Item -LiteralPath $Path -Recurse:$Recurse -Force
}

Write-Host ''
Write-Host "$($C_BOLD)NovaNN — clean$($C_RESET)"

if ($null -eq $Target) {
    if (-not (Test-Path -LiteralPath 'build' -PathType Container)) {
        Write-Host "$($C_YELLOW)nothing to do:$($C_RESET) build does not exist."
        exit 0
    }

    if (-not $DryRun -and -not $AssumeYes) {
        if ([Console]::IsInputRedirected) {
            Write-Die 'refusing to delete build non-interactively; pass --yes'
        }
        $answer = Read-Host 'Delete entire build directory? [y/N]'
        if ($answer -notmatch '^[Yy]$') {
            Write-Host "$($C_YELLOW)aborted.$($C_RESET)"
            exit 0
        }
    }

    Remove-NovaItem -Path 'build' -Recurse
    if (-not $DryRun) {
        Write-Host "$($C_GREEN)  ✔ deleted$($C_RESET) build/"
    }
    exit 0
}

switch ($Target) {
    'logs' {
        if (-not (Test-Path -LiteralPath 'build/logs' -PathType Container)) {
            Write-Host "$($C_YELLOW)nothing to do:$($C_RESET) build/logs does not exist."
            exit 0
        }
        Remove-NovaItem -Path 'build/logs' -Recurse
        if (-not $DryRun) { Write-Host "$($C_GREEN)  ✔ deleted$($C_RESET) build/logs" }
    }
    'test-logs' {
        if (-not (Test-Path -LiteralPath 'build/logs/tests' -PathType Container)) {
            Write-Host "$($C_YELLOW)nothing to do:$($C_RESET) build/logs/tests does not exist."
            exit 0
        }
        Remove-NovaItem -Path 'build/logs/tests' -Recurse
        if (-not $DryRun) { Write-Host "$($C_GREEN)  ✔ deleted$($C_RESET) build/logs/tests" }
    }
    'build-logs' {
        $buildLogDir = 'build/logs'
        $files = @()
        if (Test-Path -LiteralPath $buildLogDir -PathType Container) {
            $files = @(Get-ChildItem -Path (Join-Path $buildLogDir '*.log') -File)
        }
        if ($files.Count -eq 0) {
            Write-Host "$($C_YELLOW)nothing to do:$($C_RESET) no build log files."
            exit 0
        }
        foreach ($file in $files) {
            Remove-NovaItem -Path $file.FullName
        }
        if (-not $DryRun) {
            Write-Host "$($C_GREEN)  ✔ deleted$($C_RESET) $($files.Count) build log file(s)"
        }
    }
    default {
        $buildDir = $Target
        $hasSeparator = $Target.Contains('/') -or $Target.Contains('\')
        if (-not (Test-Path -LiteralPath $buildDir -PathType Container) -and
            -not $hasSeparator -and
            (Test-Path -LiteralPath (Join-Path 'build' $Target) -PathType Container)) {
            $buildDir = Join-Path 'build' $Target
        }

        if (-not (Test-Path -LiteralPath $buildDir -PathType Container)) {
            Write-Die "target '$Target': directory does not exist"
        }
        if (-not (Test-Path -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $buildDir 'build.ninja') -PathType Leaf)) {
            Write-Die "'$buildDir' is not a configured CMake build directory"
        }

        if ($DryRun) {
            Write-Host "$($C_CYAN)[dry-run]$($C_RESET) cmake --build '$buildDir' --target clean"
            exit 0
        }

        Write-Host "  $($C_CYAN)▸ cleaning$($C_RESET) $buildDir"
        try {
            & cmake --build $buildDir --target clean
            $rc = $LASTEXITCODE
        }
        catch {
            Write-Die "cmake --build $buildDir --target clean failed: $($_.Exception.Message)"
        }
        if ($rc -ne 0) {
            Write-Die "cmake --build $buildDir --target clean failed"
        }
        Write-Host "$($C_GREEN)  ✔ cleaned$($C_RESET) $buildDir"
    }
}
