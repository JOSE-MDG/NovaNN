#!/usr/bin/env pwsh
# Initialize or move the project-local vcpkg submodule. Maintainer tool; the
# build scripts provision vcpkg on their own.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$ScriptName = Split-Path -Leaf $PSCommandPath

Import-Module (Join-Path $ScriptDir 'lib/common.psm1') -Force
$ProjectRoot = Initialize-ScriptContext

$Colors = Get-ScriptColors
$C_RESET = $Colors.Reset
$C_BOLD = $Colors.Bold
$C_DIM = $Colors.Dim
$C_RED = $Colors.Red
$C_GREEN = $Colors.Green
$C_CYAN = $Colors.Cyan

$VcpkgRelativePath = 'thirdParty/vcpkg'
$VcpkgRoot = Join-Path $ProjectRoot $VcpkgRelativePath
$Remote = $false

function Show-Usage {
    @"
$($C_BOLD)Usage:$($C_RESET) $ScriptName [OPTIONS]

Ensure the project-local vcpkg submodule is initialized at the commit pinned
by this repository and that its executable is bootstrapped. The global
VCPKG_ROOT checkout is not used.

$($C_BOLD)Options:$($C_RESET)
  -r, --remote        Move the submodule to the remote's default branch tip
                      (maintainer bump of the vcpkg version).
  -h, --help          Show this help and exit.

The script does not stage or commit the submodule pointer. Review it with
git diff --submodule=log before committing. Exit status: 0 = ready,
1 = an update step failed, 2 = usage error.
"@ | Write-Host
}

foreach ($arg in $args) {
    if ($arg -in '-h', '--help') { Show-Usage; exit 0 }
    elseif ($arg -in '-r', '--remote') { $Remote = $true }
    elseif ($arg.StartsWith('-')) { Write-UsageErrorAndExit "unknown option: $arg" }
    else { Write-UsageErrorAndExit "unexpected argument: $arg" }
}

try {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.gitmodules'))) {
        throw 'No .gitmodules file found; add vcpkg as a Git submodule first.'
    }

    # Refuse to move a submodule that still has local changes in it.
    $OldCommit = $null
    if (Test-Path -LiteralPath (Join-Path $VcpkgRoot '.git')) {
        $dirty = Invoke-Native -FilePath git `
            -ArgumentList @('-C', $VcpkgRoot, 'status', '--porcelain', '--untracked-files=normal')
        if ($dirty) {
            throw 'The vcpkg submodule has local changes; commit or discard them before updating.'
        }
        $OldCommit = ([string](Invoke-Native -FilePath git `
            -ArgumentList @('-C', $VcpkgRoot, 'rev-parse', 'HEAD'))).Trim()
    }

    Write-Host ''
    Write-Host "$($C_BOLD)NovaNN — vcpkg submodule$($C_RESET)"
    Write-Host "$($C_DIM)Path: $VcpkgRelativePath$($C_RESET)"

    Write-Host "`n$($C_CYAN)[1/2]$($C_RESET) Updating the project-local submodule..."
    $gitArgs = @('submodule', 'update', '--init')
    if ($Remote) { $gitArgs += '--remote' }
    $gitArgs += '--', $VcpkgRelativePath
    Invoke-Native -FilePath git -ArgumentList $gitArgs

    $NewCommit = ([string](Invoke-Native -FilePath git `
        -ArgumentList @('-C', $VcpkgRoot, 'rev-parse', 'HEAD'))).Trim()
    if (-not $OldCommit) { $OldCommit = '<uninitialized>' }
    Write-Host "  $($C_DIM)Pinned source:$($C_RESET) $OldCommit → $NewCommit"

    # Override shell-level VCPKG_ROOT for this script and its children.
    $env:VCPKG_ROOT = $VcpkgRoot
    $env:PATH = "$VcpkgRoot$([IO.Path]::PathSeparator)$env:PATH"

    $exeName = if ($IsWindows) { 'vcpkg.exe' } else { 'vcpkg' }
    $exe = Join-Path $VcpkgRoot $exeName

    if ($OldCommit -ne $NewCommit -or -not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        Write-Host "`n$($C_CYAN)[2/2]$($C_RESET) Bootstrapping the local vcpkg executable..."
        if ($IsWindows) {
            Invoke-Native -FilePath (Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat') `
                -ArgumentList @('-disableMetrics')
        }
        else {
            Invoke-Native -FilePath 'bash' `
                -ArgumentList @((Join-Path $VcpkgRoot 'bootstrap-vcpkg.sh'), '-disableMetrics')
        }
    }
    else {
        Write-Host "`n$($C_CYAN)[2/2]$($C_RESET) Source commit unchanged; reusing the existing executable."
    }

    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw "Bootstrap completed without producing an executable at $exe."
    }

    Write-Host "`n$($C_GREEN)✔ vcpkg submodule ready.$($C_RESET)"
    Write-Host "`n$($C_BOLD)Review the resulting changes:$($C_RESET)"
    Write-Host "  git status --short -- $VcpkgRelativePath"
    Write-Host "  git diff --submodule=log -- $VcpkgRelativePath"
    Write-Host "`n$($C_DIM)The submodule pointer is not staged or committed automatically.$($C_RESET)"
}
catch {
    [Console]::Error.WriteLine("$($C_RED)ERROR:$($C_RESET) $($_.Exception.Message)")
    exit 1
}
