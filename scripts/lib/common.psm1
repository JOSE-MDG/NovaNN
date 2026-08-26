Set-StrictMode -Version Latest

# Shared presentation and process helpers for the PowerShell script family.

$script:IsTTY = -not [Console]::IsOutputRedirected

if ($script:IsTTY -and [string]::IsNullOrEmpty($env:NO_COLOR) -and $env:TERM -ne 'dumb') {
    $esc = [char]27
    $script:Colors = @{
        Reset  = "$esc[0m"
        Bold   = "$esc[1m"
        Dim    = "$esc[2m"
        Red    = "$esc[31m"
        Green  = "$esc[32m"
        Yellow = "$esc[33m"
        Cyan   = "$esc[36m"
        Clear  = "$([char]13)$([char]27)[K"
        IsTTY  = $script:IsTTY
    }
}
else {
    $script:Colors = @{
        Reset  = ''
        Bold   = ''
        Dim    = ''
        Red    = ''
        Green  = ''
        Yellow = ''
        Cyan   = ''
        Clear  = ''
        IsTTY  = $script:IsTTY
    }
}

function Get-NovaColors {
    return $script:Colors.Clone()
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

function Format-Elapsed {
    param([Parameter(Mandatory)][int]$Seconds)

    $minutes = [math]::Floor($Seconds / 60)
    $seconds = $Seconds % 60
    if ($minutes -gt 0) {
        return ('{0}m {1:D2}s' -f $minutes, $seconds)
    }
    return ('{0}s' -f $seconds)
}

function Get-NovaPresetNames {
    $output = @(& cmake --list-presets 2>$null)
    $names = foreach ($line in $output) {
        if ($line -match '^\s*"([^"]+)"') {
            $Matches[1]
        }
    }

    if (@($names).Count -eq 0) {
        Write-Die 'no CMake presets found; is CMakePresets.json present?'
    }

    return @($names)
}

function Show-Spinner {
    param(
        [Parameter(Mandatory)][System.Management.Automation.Job]$Job,
        [Parameter(Mandatory)][string]$LogFile,
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][int]$N,
        [Parameter(Mandatory)][int]$Total,
        [string]$Pattern = '\[(\d+)/(\d+)\]([^\r\n]*)'
    )

    $locale = $env:LC_ALL
    if ([string]::IsNullOrEmpty($locale)) { $locale = $env:LC_CTYPE }
    if ([string]::IsNullOrEmpty($locale)) { $locale = $env:LANG }

    if ($locale -and $locale -match '(?i)utf-?8') {
        $frames = @('⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏')
        $barFull = '█'
        $barEmpty = '░'
    }
    else {
        $frames = @('|', '/', '-', '\')
        $barFull = '#'
        $barEmpty = '-'
    }

    $frameIndex = 0
    $start = Get-Date

    while ($Job.State -eq 'Running' -or $Job.State -eq 'NotStarted') {
        $elapsed = [int]((Get-Date) - $start).TotalSeconds
        if ($elapsed -lt 0) { $elapsed = 0 }

        $percent = -1
        $target = $null

        if (Test-Path -LiteralPath $LogFile) {
            $tail = Get-Content -LiteralPath $LogFile -Tail 60 -ErrorAction SilentlyContinue
            if ($tail) {
                $joined = $tail -join "`n"
                $matches = [regex]::Matches($joined, $Pattern)
                if ($matches.Count -gt 0) {
                    $last = $matches[$matches.Count - 1]
                    $number = [int]$last.Groups[1].Value
                    $denominator = [int]$last.Groups[2].Value
                    if ($denominator -gt 0) {
                        $percent = [int]($number * 100 / $denominator)
                    }
                    $target = $last.Groups[3].Value.Trim()
                    if ([string]::IsNullOrEmpty($target)) { $target = $Label }
                    if ($target.Length -gt 40) {
                        $target = $target.Substring(0, 37) + '...'
                    }
                }
            }
        }

        if ($percent -ge 0) {
            $filled = [int]($percent * 20 / 100)
            $bar = ($barFull * $filled) + ($barEmpty * (20 - $filled))
            $percentText = $percent.ToString().PadLeft(3)
            $line = "`r  $($frames[$frameIndex]) $($script:Colors.Cyan)$percentText%$($script:Colors.Reset) " +
                    "$($script:Colors.Cyan)[$bar]$($script:Colors.Reset) $($script:Colors.Dim)$target$($script:Colors.Reset) " +
                    "· $($script:Colors.Dim)[$N/$Total]$($script:Colors.Reset) · $($script:Colors.Dim)$(Format-Elapsed $elapsed)$($script:Colors.Reset)"
        }
        else {
            $line = "`r  $($frames[$frameIndex]) $($script:Colors.Bold)$Label$($script:Colors.Reset) " +
                    "· $($script:Colors.Dim)[$N/$Total]$($script:Colors.Reset) · $($script:Colors.Dim)$(Format-Elapsed $elapsed)$($script:Colors.Reset)"
        }

        Write-Host -NoNewline $line
        Start-Sleep -Milliseconds 100
        $frameIndex = ($frameIndex + 1) % $frames.Count
    }
}

Export-ModuleMember -Function @(
    'Get-NovaColors',
    'Write-Die',
    'Write-UsageErrorAndExit',
    'Format-Elapsed',
    'Get-NovaPresetNames',
    'Show-Spinner'
)
