param(
    [Parameter(Position=0)][ValidateSet('menu','setup','check','quick','full','figures','tables','reproduce')][string]$Action = 'menu',
    [Parameter(Position=1)][ValidateSet('all','example1-original','example1-improved','example2-article','example2-corrected','example2-improved')][string]$Experiment = 'all'
)
$ErrorActionPreference = 'Stop'
try {
    . (Join-Path $PSScriptRoot 'bootstrap.ps1')
    $project = Split-Path -Parent $PSScriptRoot
    if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA is unavailable. Run this launcher in a normal Windows user session.' }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'A 64-bit Windows installation is required.' }
    $runtime = Join-Path $env:LOCALAPPDATA 'ArticleEfimov\lyapunov-nn-homogeneous'
    Set-Location -LiteralPath $project
    if ($Action -eq 'menu') {
        Write-Host "`nArticle Efimov - numerical experiments`n"
        Write-Host '1  Reproduce both examples from preserved trained weights [default]'
        Write-Host '2  Train new models from scratch (results may differ)'
        Write-Host '3  Open verified figures (no setup needed)'
        Write-Host '4  Rebuild reference summary tables'
        Write-Host '5  Install dependencies only'
        Write-Host '6  Run software tests only'
        Write-Host '7  Short installation test (untrained models; NOT paper results)'
        Write-Host '0  Exit'
        $choice = Read-Host 'Select'
        switch ($choice) {
            '' { $Action = 'reproduce' }
            '1' { $Action = 'reproduce' }
            '2' {
                $Action = 'full'
                Write-Host '1 Example 1 original; 2 Example 1 improved; 3 Example 2 article; 4 Example 2 corrected; 5 Example 2 improved; 6 All'
                $variants = @('example1-original','example1-improved','example2-article','example2-corrected','example2-improved','all')
                $selected = Read-Host 'Select experiment [6]'
                if ($selected -eq '') { $selected = '6' }
                if ($selected -notmatch '^[1-6]$') { throw 'Invalid experiment selection' }
                $Experiment = $variants[[int]$selected - 1]
            }
            '3' { $Action = 'figures' }
            '4' { $Action = 'tables' }
            '5' { $Action = 'setup' }
            '6' { $Action = 'check' }
            '7' { $Action = 'quick' }
            '0' { exit 0 }
            default { throw 'Invalid menu selection' }
        }
    }
    if ($Action -eq 'figures') {
        $gallery = Join-Path $project 'results\LATEST.html'
        if (-not (Test-Path -LiteralPath $gallery)) { $gallery = Join-Path $project 'RESULTS.html' }
        Invoke-Item -LiteralPath $gallery
        exit 0
    }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:MPLBACKEND = 'Agg'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $python = Initialize-Runtime $project $runtime
    if ($Action -eq 'setup') { exit 0 }
    & $python (Join-Path $project 'launcher\run.py') --action $Action --experiment $Experiment
    exit $LASTEXITCODE
} catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Correct the error and run START.cmd again. Existing reference results are preserved.'
    exit 1
}
