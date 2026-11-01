#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Update NovaNN's vcpkg submodule and synchronize registry baselines.
.DESCRIPTION
    Updates the project-local vcpkg submodule, bootstraps the executable if
    its pinned source commit changed, and runs x-update-baseline using that
    local vcpkg instance. The global VCPKG_ROOT checkout is not used.
.PARAMETER Help
    Show help and exit.
.NOTES
    Exit status: 0 = update and baseline synchronization completed;
    1 = an update step failed; 2 = usage error.
.EXAMPLE
    scripts/update-vcpkg.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$ScriptName = Split-Path -Leaf $PSCommandPath
Set-Location -LiteralPath $ProjectRoot

Import-Module (Join-Path $ScriptDir 'lib/common.psm1') -Force
$NovaColors = Get-NovaColors
$C_RESET = $NovaColors.Reset
$C_BOLD = $NovaColors.Bold
$C_DIM = $NovaColors.Dim
$C_RED = $NovaColors.Red
$C_GREEN = $NovaColors.Green
$C_YELLOW = $NovaColors.Yellow
$C_CYAN = $NovaColors.Cyan

function Show-Usage {
    @"
$($C_BOLD)Usage:$($C_RESET) $ScriptName [OPTIONS]

Update NovaNN's project-local vcpkg submodule and synchronize the baseline
in vcpkg-configuration.json. The global VCPKG_ROOT checkout is not used.

$($C_BOLD)Options:$($C_RESET)
  -h, --help    Show this help and exit.

$($C_BOLD)Effects:$($C_RESET)
  1. Fetch and update the vcpkg submodule to the remote's default branch tip.
  2. Bootstrap vcpkg if the source commit changed or the executable is missing.
  3. Run the local vcpkg x-update-baseline command.

The script does not stage or commit changes. Review vcpkg and
vcpkg-configuration.json with git diff before committing.
"@ | Write-Host
}

function Invoke-CheckedNativeCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$ArgumentList = @(),
        [Parameter(Mandatory)][string]$Description
    )

    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
}

try {
    foreach ($argument in $args) {
        if ($argument -in @('-h', '--help')) {
            Show-Usage
            exit 0
        }
        if ($argument.StartsWith('-')) {
            Write-UsageErrorAndExit "unknown option: $argument"
        }
        Write-UsageErrorAndExit "unexpected argument: $argument"
    }

    $VcpkgRelativePath = 'thirdParty/vcpkg'
    $VcpkgRoot = Join-Path $ProjectRoot $VcpkgRelativePath
    $ConfigurationPath = Join-Path $ProjectRoot 'vcpkg-configuration.json'
    $GitModulesPath = Join-Path $ProjectRoot '.gitmodules'

    $null = & git -C $ProjectRoot rev-parse --show-toplevel 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'NovaNN root is not inside a Git repository.'
    }
    if (-not (Test-Path -LiteralPath $GitModulesPath -PathType Leaf)) {
        throw 'No .gitmodules file found; add vcpkg as a Git submodule first.'
    }
    if (-not (Test-Path -LiteralPath $ConfigurationPath -PathType Leaf)) {
        throw 'vcpkg-configuration.json was not found in the NovaNN root.'
    }
    $configurationStatus = @(& git -C $ProjectRoot status --porcelain -- vcpkg-configuration.json)
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not inspect vcpkg-configuration.json status.'
    }
    if (@($configurationStatus | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
        throw 'vcpkg-configuration.json has uncommitted changes; commit or stash them before updating the baseline.'
    }

    $OldCommit = $null
    if (Test-Path -LiteralPath $VcpkgRoot -PathType Container) {
        $insideOutput = & git -C $VcpkgRoot rev-parse --is-inside-work-tree 2>$null
        $insideExitCode = $LASTEXITCODE
        if ($insideExitCode -eq 0 -and ($insideOutput -join '').Trim() -eq 'true') {
            $gitStatus = @(& git -C $VcpkgRoot status --porcelain --untracked-files=normal)
            if ($LASTEXITCODE -ne 0) {
                throw 'Could not inspect the vcpkg submodule status.'
            }
            $dirtyEntries = @($gitStatus | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            if ($dirtyEntries.Count -gt 0) {
                throw 'The vcpkg submodule has local changes; commit or discard them before updating.'
            }

            $commitOutput = & git -C $VcpkgRoot rev-parse HEAD
            if ($LASTEXITCODE -ne 0) {
                throw 'Could not read the current vcpkg submodule commit.'
            }
            $OldCommit = ($commitOutput -join '').Trim()
        }
    }

    Write-Host ''
    Write-Host "$($C_BOLD)NovaNN — update vcpkg$($C_RESET)"
    Write-Host "$($C_DIM)Submodule: $VcpkgRelativePath$($C_RESET)"

    Write-Host "`n$($C_CYAN)[1/3]$($C_RESET) Updating the project-local submodule..."
    Invoke-CheckedNativeCommand -FilePath 'git' `
        -ArgumentList @('submodule', 'update', '--init', '--remote', '--', $VcpkgRelativePath) `
        -Description 'Updating the vcpkg submodule'

    $newCommitOutput = & git -C $VcpkgRoot rev-parse HEAD
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not read the updated vcpkg submodule commit.'
    }
    $NewCommit = ($newCommitOutput -join '').Trim()
    $oldCommitLabel = if ($OldCommit) { $OldCommit } else { '<uninitialized>' }
    Write-Host "  $($C_DIM)Pinned source:$($C_RESET) $oldCommitLabel → $NewCommit"

    # Override shell-level VCPKG_ROOT for this script and its child processes.
    $env:VCPKG_ROOT = $VcpkgRoot
    $pathSeparator = [IO.Path]::PathSeparator
    $currentPath = if ($env:PATH) { $env:PATH } else { '' }
    $localPathEntry = [IO.Path]::GetFullPath($VcpkgRoot).TrimEnd([char[]]@(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    ))
    $pathEntries = @($currentPath -split [regex]::Escape([string]$pathSeparator) |
        Where-Object {
            if ([string]::IsNullOrWhiteSpace($_)) {
                $false
            }
            else {
                $normalized = [IO.Path]::GetFullPath($_).TrimEnd([char[]]@(
                    [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
                ))
                if ($IsWindows) { $normalized -ine $localPathEntry }
                else { $normalized -cne $localPathEntry }
            }
        })
    $env:PATH = (@($VcpkgRoot) + $pathEntries) -join $pathSeparator

    $VcpkgExecutableName = if ($IsWindows) { 'vcpkg.exe' } else { 'vcpkg' }
    $VcpkgExecutable = Join-Path $VcpkgRoot $VcpkgExecutableName
    if ($OldCommit -ne $NewCommit -or -not (Test-Path -LiteralPath $VcpkgExecutable -PathType Leaf)) {
        Write-Host "`n$($C_CYAN)[2/3]$($C_RESET) Bootstrapping the local vcpkg executable..."
        if ($IsWindows) {
            $Bootstrap = Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat'
            if (-not (Test-Path -LiteralPath $Bootstrap -PathType Leaf)) {
                throw "Bootstrap script not found: $Bootstrap"
            }
            Invoke-CheckedNativeCommand -FilePath $Bootstrap `
                -ArgumentList @('-disableMetrics') `
                -Description 'Bootstrapping local vcpkg'
        }
        else {
            $Bootstrap = Join-Path $VcpkgRoot 'bootstrap-vcpkg.sh'
            if (-not (Test-Path -LiteralPath $Bootstrap -PathType Leaf)) {
                throw "Bootstrap script not found: $Bootstrap"
            }
            Invoke-CheckedNativeCommand -FilePath 'bash' `
                -ArgumentList @($Bootstrap, '-disableMetrics') `
                -Description 'Bootstrapping local vcpkg'
        }
    }
    else {
        Write-Host "`n$($C_CYAN)[2/3]$($C_RESET) Source commit unchanged; reusing the existing executable."
    }

    if (-not (Test-Path -LiteralPath $VcpkgExecutable -PathType Leaf)) {
        throw "Bootstrap completed without producing an executable at $VcpkgExecutable."
    }

    Write-Host "`n$($C_CYAN)[3/3]$($C_RESET) Updating configured registry baselines..."
    Invoke-CheckedNativeCommand -FilePath $VcpkgExecutable `
        -ArgumentList @('x-update-baseline') `
        -Description 'Updating configured registry baselines'

    Write-Host "`n$($C_GREEN)✔ vcpkg update completed.$($C_RESET)"
    Write-Host "`n$($C_BOLD)Review the resulting changes:$($C_RESET)"
    Write-Host "  git status --short -- $VcpkgRelativePath vcpkg-configuration.json"
    Write-Host '  git diff -- vcpkg-configuration.json'
    Write-Host "  git diff --submodule=log -- $VcpkgRelativePath"
    Write-Host "`n$($C_DIM)The submodule pointer and baseline are not staged or committed automatically.$($C_RESET)"
}
catch {
    [Console]::Error.WriteLine("$($C_RED)ERROR:$($C_RESET) $($_.Exception.Message)")
    exit 1
}
