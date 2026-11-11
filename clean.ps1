#!/usr/bin/env pwsh
# Clean build artifacts and logs. Without --target the whole build/ tree is
# removed; with one, that tree is cleaned and its codegen outputs are preserved.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Import-Module (Join-Path $PSScriptRoot 'scripts/lib/common.psm1') -Force
$ProjectRoot = Initialize-ScriptContext
$ScriptName = Split-Path -Leaf $PSCommandPath

$Colors = Get-ScriptColors
$C_RESET = $Colors.Reset
$C_BOLD = $Colors.Bold
$C_RED = $Colors.Red
$C_GREEN = $Colors.Green
$C_YELLOW = $Colors.Yellow
$C_CYAN = $Colors.Cyan

$DryRun = $false
$AssumeYes = $false
$Target = $null

function Show-Usage {
    @"
$($C_BOLD)Usage:$($C_RESET) $ScriptName [OPTIONS] [--target <path|logs|test-logs|build-logs>]

$($C_BOLD)Targets:$($C_RESET)
  <path>                Run cmake --build <path> --target clean.
                        Configuration is kept; source-tree codegen outputs
                        are snapshotted and restored. A bare preset name
                        resolves to build/<preset> when it exists.
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

for ($i = 0; $i -lt $args.Count; $i++) {
    $arg = $args[$i]

    if ($arg -in '-h', '--help') { Show-Usage; exit 0 }
    elseif ($arg -in '-n', '--dry-run') { $DryRun = $true }
    elseif ($arg -in '-y', '--yes') { $AssumeYes = $true }
    elseif ($arg -eq '--target') {
        if ($i + 1 -ge $args.Count) { Write-UsageErrorAndExit '--target requires a value' }
        if ($null -ne $Target) { Write-UsageErrorAndExit '--target given more than once' }
        $i++
        $Target = $args[$i]
        if ([string]::IsNullOrEmpty($Target)) {
            Write-UsageErrorAndExit '--target requires a non-empty value'
        }
    }
    else { Write-UsageErrorAndExit "unknown option: $arg" }
}

# Codegen outputs live in the source tree, but Ninja's clean deletes them along
# with the object files. Snapshot them before a directory clean, restore them
# afterwards byte for byte.
function Backup-CodegenOutputs {
    try {
        $out = & uv run -q tools/codegen/generate.py gen --all --list-outputs 2>$null
    }
    catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }

    $files = @()
    foreach ($line in $out) {
        $f = "$line".Trim()
        if ($f -ne '' -and (Test-Path -LiteralPath $f -PathType Leaf)) {
            $files += $f
        }
    }
    if ($files.Count -eq 0) { return $null }

    $backup = Join-Path ([System.IO.Path]::GetTempPath()) ('nova-clean-' + [System.IO.Path]::GetRandomFileName())
    New-Item -ItemType Directory -Path $backup | Out-Null
    foreach ($f in $files) {
        $dest = Join-Path $backup $f
        New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
        Copy-Item -LiteralPath $f -Destination $dest -Force
    }
    $backup
}

function Restore-CodegenOutputs {
    param([string]$BackupDir)

    if ([string]::IsNullOrEmpty($BackupDir)) { return }
    if (-not (Test-Path -LiteralPath $BackupDir -PathType Container)) { return }

    Copy-Item -Path (Join-Path $BackupDir '*') -Destination $ProjectRoot -Recurse -Force
    Remove-Item -LiteralPath $BackupDir -Recurse -Force
}

function Remove-IfNotDryRun {
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

    Remove-IfNotDryRun -Path 'build' -Recurse
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
        Remove-IfNotDryRun -Path 'build/logs' -Recurse
        if (-not $DryRun) { Write-Host "$($C_GREEN)  ✔ deleted$($C_RESET) build/logs" }
    }
    'test-logs' {
        if (-not (Test-Path -LiteralPath 'build/logs/tests' -PathType Container)) {
            Write-Host "$($C_YELLOW)nothing to do:$($C_RESET) build/logs/tests does not exist."
            exit 0
        }
        Remove-IfNotDryRun -Path 'build/logs/tests' -Recurse
        if (-not $DryRun) { Write-Host "$($C_GREEN)  ✔ deleted$($C_RESET) build/logs/tests" }
    }
    'build-logs' {
        $files = @(Get-ChildItem -Path (Join-Path 'build/logs' '*.log') -File -ErrorAction SilentlyContinue)
        if ($files.Count -eq 0) {
            Write-Host "$($C_YELLOW)nothing to do:$($C_RESET) no build log files."
            exit 0
        }
        foreach ($file in $files) {
            Remove-IfNotDryRun -Path $file.FullName
        }
        if (-not $DryRun) {
            Write-Host "$($C_GREEN)  ✔ deleted$($C_RESET) $($files.Count) build log file(s)"
        }
    }
    default {
        $buildDir = $Target
        if (-not (Test-Path -LiteralPath $buildDir -PathType Container) -and
            -not $Target.Contains('/') -and -not $Target.Contains('\')) {
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
            Write-Host "$($C_CYAN)[dry-run]$($C_RESET) cmake --build '$buildDir' --target clean (codegen outputs preserved)"
            exit 0
        }

        Write-Host "  $($C_CYAN)▸ cleaning$($C_RESET) $buildDir"
        $backupDir = Backup-CodegenOutputs
        try {
            Invoke-Native -FilePath cmake -ArgumentList @('--build', $buildDir, '--target', 'clean')
        }
        catch {
            Restore-CodegenOutputs -BackupDir $backupDir
            Write-Die "cmake --build $buildDir --target clean failed"
        }
        Restore-CodegenOutputs -BackupDir $backupDir
        Write-Host "$($C_GREEN)  ✔ cleaned$($C_RESET) $buildDir"
    }
}
