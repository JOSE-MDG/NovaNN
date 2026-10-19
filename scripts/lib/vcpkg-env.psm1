<#
.SYNOPSIS
    Select NovaNN's project-local vcpkg for the current PowerShell process.
.DESCRIPTION
    Import this module and call Set-VcpkgEnvironment before invoking CMake
    or vcpkg. The function overrides VCPKG_ROOT and prepends the project-local
    vcpkg directory to PATH for the current process and its child processes.
#>

Set-StrictMode -Version Latest

$script:DefaultProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Set-VcpkgEnvironment {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$ProjectRoot = $script:DefaultProjectRoot
    )

    $resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction Stop).Path
    $vcpkgRoot = Join-Path $resolvedProjectRoot 'thirdParty/vcpkg'
    $toolchain = Join-Path $vcpkgRoot 'scripts/buildsystems/vcpkg.cmake'
    $executableName = if ($IsWindows) { 'vcpkg.exe' } else { 'vcpkg' }
    $executable = Join-Path $vcpkgRoot $executableName

    if (-not (Test-Path -LiteralPath $toolchain -PathType Leaf) -or
        -not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        $bootstrapHint = if ($IsWindows) {
            'Initialize the submodule and run scripts/update-vcpkg.ps1, or thirdParty/vcpkg/bootstrap-vcpkg.bat.'
        }
        else {
            'Initialize the submodule and run scripts/update-vcpkg.ps1, or thirdParty/vcpkg/bootstrap-vcpkg.sh.'
        }
        throw "NovaNN's local vcpkg is missing or not bootstrapped: '$vcpkgRoot'. $bootstrapHint"
    }

    $env:VCPKG_ROOT = $vcpkgRoot
    $pathSeparator = [IO.Path]::PathSeparator
    $currentPath = if ($env:PATH) { $env:PATH } else { '' }
    $localPathEntry = [IO.Path]::GetFullPath($vcpkgRoot).TrimEnd([char[]]@(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    ))

    # Keep exactly one local entry and guarantee it has highest precedence.
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
    $env:PATH = (@($vcpkgRoot) + $pathEntries) -join $pathSeparator

    Write-Verbose "VCPKG_ROOT set to '$env:VCPKG_ROOT'."
}

Export-ModuleMember -Function Set-VcpkgEnvironment
