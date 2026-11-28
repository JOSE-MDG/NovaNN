#!/usr/bin/env pwsh
# Install a pinned LLVM toolchain for the Windows CI job.
#
# The VS 2026 STL shipped on windows-2025 requires Clang 22 or newer, while the
# runner image provides LLVM 20.1.8. Install a newer clang-cl and lld-link and
# put them first in PATH for the later CMake steps.
#
# Sources:
#   https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.5
#   https://github.com/microsoft/winget-pkgs/blob/master/manifests/l/LLVM/LLVM/23.1.5/LLVM.LLVM.installer.yaml

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Version = '23.1.5'
$ProductCode = '{5B1F1A5F-3452-407F-B9AB-32FE05828D46}'
$ExpectedHash = '288AB326377157A9E02E6FB9724914CC2B55963882D493A405F31DC66E837C12'
$Url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-$Version/LLVM-$Version-win64.msi"

$cacheDir = Join-Path $env:RUNNER_TEMP 'llvm-msi'
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
$msi = Join-Path $cacheDir "LLVM-$Version-win64.msi"

if (Test-Path -LiteralPath $msi -PathType Leaf) {
    Write-Host "Using cached installer: $msi"
}
else {
    Write-Host "Downloading LLVM $Version from $Url"
    Invoke-WebRequest -Uri $Url -OutFile $msi -ProgressAction SilentlyContinue
}

$actualHash = (Get-FileHash -LiteralPath $msi -Algorithm SHA256).Hash
if ($actualHash -ne $ExpectedHash) {
    throw "LLVM installer checksum mismatch. Expected $ExpectedHash, got $actualHash."
}

$log = Join-Path $env:RUNNER_TEMP 'llvm-install.log'
$arguments = "/i `"$msi`" /qn /norestart /log `"$log`""
Write-Host "Installing LLVM $Version"
$process = Start-Process -FilePath msiexec.exe -ArgumentList $arguments -Wait -PassThru
if ($process.ExitCode -ne 0) {
    if (Test-Path -LiteralPath $log -PathType Leaf) {
        Get-Content -LiteralPath $log -Tail 40
    }
    throw "LLVM MSI installation failed with exit code $($process.ExitCode)."
}

$uninstallKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$ProductCode",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$ProductCode"
)
$installLocation = $null
foreach ($key in $uninstallKeys) {
    if (Test-Path -LiteralPath $key) {
        $installLocation = (Get-ItemProperty -LiteralPath $key -Name InstallLocation).InstallLocation
        break
    }
}
if (-not $installLocation) {
    throw "Could not find the LLVM install location for $ProductCode."
}
$binDir = Join-Path $installLocation 'bin'

foreach ($executable in 'clang-cl.exe', 'lld-link.exe') {
    $candidate = Join-Path $binDir $executable
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "$executable was not found in $binDir."
    }
}

"$binDir" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
Write-Host "Added $binDir to PATH"
& (Join-Path $binDir 'clang-cl.exe') --version
