#!/usr/bin/env pwsh
# Select NovaNN's project-local vcpkg for the current PowerShell process.

Set-StrictMode -Version Latest

$script:DefaultProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Set-VcpkgEnvironment {
    param([string]$ProjectRoot = $script:DefaultProjectRoot)

    $root = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction Stop).Path
    $vcpkgRoot = Join-Path $root 'thirdParty/vcpkg'
    $toolchain = Join-Path $vcpkgRoot 'scripts/buildsystems/vcpkg.cmake'
    $exeName = if ($IsWindows) { 'vcpkg.exe' } else { 'vcpkg' }
    $exe = Join-Path $vcpkgRoot $exeName

    # The submodule is initialized and bootstrapped on demand, so a from-source
    # build needs no separate setup script.
    if (-not (Test-Path -LiteralPath $toolchain -PathType Leaf)) {
        Write-Host 'vcpkg: initializing the thirdParty/vcpkg submodule...'
        & git -C $root submodule update --init -- thirdParty/vcpkg
        if ($LASTEXITCODE -ne 0) {
            throw "Could not initialize the vcpkg submodule at '$vcpkgRoot'."
        }
    }

    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        Write-Host 'vcpkg: bootstrapping the local executable...'
        if ($IsWindows) {
            & (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
        }
        else {
            & bash (Join-Path $vcpkgRoot 'bootstrap-vcpkg.sh') -disableMetrics
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Could not bootstrap vcpkg at '$vcpkgRoot'."
        }
    }

    if (-not (Test-Path -LiteralPath $toolchain -PathType Leaf) -or
        -not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw "NovaNN's local vcpkg is missing or not bootstrapped: '$vcpkgRoot'."
    }

    $env:VCPKG_ROOT = $vcpkgRoot
    $env:PATH = "$vcpkgRoot$([IO.Path]::PathSeparator)$env:PATH"
}

Export-ModuleMember -Function Set-VcpkgEnvironment
