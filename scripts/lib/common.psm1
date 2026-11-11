Set-StrictMode -Version Latest

# Colors, enabled only when stdout is a TTY.
$tty = -not [Console]::IsOutputRedirected -and
       [string]::IsNullOrEmpty($env:NO_COLOR) -and
       $env:TERM -ne 'dumb'
$esc = if ($tty) { [char]27 } else { '' }

$script:Colors = @{
    Reset = "$esc[0m"
    Bold  = "$esc[1m"
    Dim   = "$esc[2m"
    Red   = "$esc[31m"
    Green = "$esc[32m"
    Yellow = "$esc[33m"
    Cyan  = "$esc[36m"
    Clear = if ($tty) { "$([char]13)$esc[K" } else { '' }
    IsTTY = $tty
}

function Get-ScriptColors {
    $script:Colors.Clone()
}

function Write-Die {
    param([Parameter(Mandatory)][string]$Message)

    [Console]::Error.WriteLine("$($script:Colors.Red)ERROR:$($script:Colors.Reset) $Message")
    exit 1
}

function Write-UsageErrorAndExit {
    param([Parameter(Mandatory)][string]$Message)

    [Console]::Error.WriteLine("$($script:Colors.Red)Usage error:$($script:Colors.Reset) $Message")
    [Console]::Error.WriteLine("Run $($script:Colors.Bold)--help$($script:Colors.Reset) for usage.")
    exit 2
}

# Run a native command, throw when it exits non-zero and return its output.
function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$ArgumentList = @()
    )

    if (-not (Get-Command -Name $FilePath -ErrorAction SilentlyContinue)) {
        throw "$FilePath was not found in PATH."
    }

    $output = & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
    $output
}

# Move to the repository root and return it.
function Initialize-ScriptContext {
    $root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    Set-Location -Path $root
    $root
}

function Format-Elapsed {
    param([Parameter(Mandatory)][int]$Seconds)

    $m = [math]::Floor($Seconds / 60)
    $s = $Seconds % 60
    if ($m -gt 0) {
        '{0}m {1:D2}s' -f $m, $s
    }
    else {
        '{0}s' -f $s
    }
}

# Return one CMake preset name per element.
function Get-PresetNames {
    $names = & cmake --list-presets 2>$null | ForEach-Object {
        if ($_ -match '^\s*"([^"]+)"') { $Matches[1] }
    }

    if (-not $names) {
        Write-Die 'no CMake presets found; is CMakePresets.json present?'
    }

    @($names)
}

# Pipeline filter: exact match or "<filter>-*". With no filters, everything
# passes through.
function Select-Presets {
    param([string[]]$Filter)

    process {
        $preset = $_
        if (-not $Filter) {
            $preset
            return
        }
        foreach ($f in $Filter) {
            if ($preset -eq $f -or $preset -like "$f-*") {
                $preset
                return
            }
        }
    }
}

# Spinner for a background job. Tails $LogFile for "[n/m]" markers;
# $Pattern overrides the marker regex for tools with a different format.
function Show-Spinner {
    param(
        [Parameter(Mandatory)][System.Management.Automation.Job]$Job,
        [Parameter(Mandatory)][string]$LogFile,
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][int]$N,
        [Parameter(Mandatory)][int]$Total,
        [string]$Pattern = '\[(\d+)/(\d+)\]([^\r\n]*)'
    )

    $frames = @('|', '/', '-', '\')
    $i = 0
    $start = Get-Date

    while ($Job.State -eq 'Running' -or $Job.State -eq 'NotStarted') {
        $elapsed = [int]((Get-Date) - $start).TotalSeconds
        if ($elapsed -lt 0) { $elapsed = 0 }

        $percent = -1
        $target = ''
        if (Test-Path -LiteralPath $LogFile) {
            $match = Get-Content -LiteralPath $LogFile -Tail 60 -ErrorAction SilentlyContinue |
                Select-String -Pattern $Pattern |
                Select-Object -Last 1
            if ($match) {
                $number = [int]$match.Matches[0].Groups[1].Value
                $denominator = [int]$match.Matches[0].Groups[2].Value
                if ($denominator -gt 0) {
                    $percent = [int]($number * 100 / $denominator)
                }
                $target = $match.Matches[0].Groups[3].Value.Trim()
                if (-not $target) { $target = $Label }
            }
        }

        $line = "`r  $($frames[$i]) "
        if ($percent -ge 0) {
            $line += "$($script:Colors.Cyan)$percent%$($script:Colors.Reset) " +
                     "$($script:Colors.Dim)$target$($script:Colors.Reset)"
        }
        else {
            $line += "$($script:Colors.Bold)$Label$($script:Colors.Reset)"
        }
        $line += " $($script:Colors.Dim)· [$N/$Total] · $(Format-Elapsed $elapsed)$($script:Colors.Reset)"

        Write-Host -NoNewline $line
        Start-Sleep -Milliseconds 100
        $i = ($i + 1) % 4
    }
}

Export-ModuleMember -Function @(
    'Get-ScriptColors',
    'Write-Die',
    'Write-UsageErrorAndExit',
    'Invoke-Native',
    'Initialize-ScriptContext',
    'Format-Elapsed',
    'Get-PresetNames',
    'Select-Presets',
    'Show-Spinner'
)
