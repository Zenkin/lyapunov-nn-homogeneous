Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-Sha256 {
    param([string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}

function Get-PythonInfo {
    param([string]$Executable, [string]$Prefix = '')
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $null }
    $probe = 'import sys,struct,json; print(json.dumps([sys.executable,list(sys.version_info[:2]),struct.calcsize(''P'')*8]))'
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $info.Arguments = ($Prefix + ' -c "' + $probe + '"').Trim()
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        if (-not $process.WaitForExit(15000)) { $process.Kill(); return $null }
        if ($process.ExitCode -ne 0) { return $null }
        $result = $process.StandardOutput.ReadToEnd() | ConvertFrom-Json
        if ($result[1][0] -eq 3 -and $result[1][1] -eq 12 -and $result[2] -eq 64) {
            return [string]$result[0]
        }
    } catch { return $null } finally { $process.Dispose() }
    return $null
}

function Find-Python {
    param([string]$RuntimeRoot)
    $candidates = @(
        (Join-Path $RuntimeRoot 'python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
    )
    foreach ($key in @('HKCU:\Software\Python\PythonCore\3.12\InstallPath', 'HKLM:\Software\Python\PythonCore\3.12\InstallPath')) {
        if (Test-Path -LiteralPath $key) {
            $installed = (Get-Item -LiteralPath $key).GetValue('')
            if ($installed) { $candidates += Join-Path $installed 'python.exe' }
        }
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command -and $command.Source -notlike '*\WindowsApps\*') { $candidates += $command.Source }
    foreach ($candidate in $candidates) {
        $found = Get-PythonInfo $candidate
        if ($found) { return $found }
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $found = Get-PythonInfo $launcher.Source '-3.12'
        if ($found) { return $found }
    }
    return $null
}

function Get-Download {
    param([string]$Url, [string]$Destination)
    Write-Host ('Downloading ' + [IO.Path]::GetFileName($Destination))
    $partial = $Destination + '.part'
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source --fail --location --retry 2 --connect-timeout 30 --max-time 1800 --progress-bar --output $partial $Url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    } else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $previousProgress = $ProgressPreference
        try {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial -TimeoutSec 1800
        } finally { $ProgressPreference = $previousProgress }
    }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

function Install-Python {
    param([string]$RuntimeRoot)
    $installer = Join-Path $RuntimeRoot 'downloads\python-3.12.10-amd64.exe'
    if (-not (Test-Path -LiteralPath $installer)) {
        Get-Download 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' $installer
    }
    if (-not (Get-Command Get-AuthenticodeSignature -ErrorAction SilentlyContinue)) {
        Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1')
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
        throw 'The Python installer signature could not be verified. Check the network connection and system clock.'
    }
    $target = Join-Path $RuntimeRoot 'python312'
    Write-Host 'Installing Python 3.12 for the current user...'
    $installerArgs = @('/quiet', 'InstallAllUsers=0', 'Include_launcher=0', 'Include_test=0', 'Include_doc=0', 'Include_tcltk=0', 'Include_pip=1', 'PrependPath=0', 'Shortcuts=0', 'AssociateFiles=0', ('TargetDir="' + $target + '"'))
    $process = Start-Process -FilePath $installer -ArgumentList $installerArgs -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 3010)) { throw "Python installer failed: exit $($process.ExitCode)" }
    $found = Get-PythonInfo (Join-Path $target 'python.exe')
    if (-not $found) { throw 'Python 3.12 x64 is unavailable after installation. A Windows restart may be required.' }
    return $found
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}

function Test-Environment {
    param([string]$Python, [string]$Runner)
    if (-not (Test-Path -LiteralPath $Python)) { return $false }
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Python
    $info.Arguments = '"' + $Runner + '" --probe'
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        if (-not $process.WaitForExit(60000)) { $process.Kill(); return $false }
        return ($process.ExitCode -eq 0)
    } catch { return $false } finally { $process.Dispose() }
}

function Initialize-Runtime {
    param([string]$ProjectRoot, [string]$RuntimeRoot)
    New-Item -ItemType Directory -Path (Join-Path $RuntimeRoot 'downloads') -Force | Out-Null
    $lock = $null
    try {
        try { $lock = [IO.File]::Open((Join-Path $RuntimeRoot 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None') }
        catch { throw 'Another launcher is preparing this environment. Wait for it to finish.' }
        $basePython = Find-Python $RuntimeRoot
        if (-not $basePython) { $basePython = Install-Python $RuntimeRoot }
        $environment = Join-Path $RuntimeRoot '.env'
        $python = Join-Path $environment 'Scripts\python.exe'
        if (-not (Get-PythonInfo $python)) {
            if (Test-Path -LiteralPath $environment) {
                $resolved = [IO.Path]::GetFullPath($environment)
                $allowed = [IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\') + '\'
                if (-not $resolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid environment path' }
                Move-Item -LiteralPath $resolved -Destination ($resolved + '.old-' + [Guid]::NewGuid().ToString('N'))
            }
            Write-Host 'Creating the local .env environment...'
            Invoke-Checked $basePython @('-m', 'venv', $environment)
        }
        $runner = Join-Path $ProjectRoot 'launcher\run.py'
        if (-not (Test-Environment $python $runner)) {
            Write-Host 'Installing pinned CPU dependencies (first start needs internet and several GB of free space)...'
            # Use the original PyTorch CDN; the index can redirect this wheel to a different CDN.
            $wheel = Join-Path $RuntimeRoot 'downloads\torch-2.8.0+cpu-cp312-cp312-win_amd64.whl'
            $expectedHash = '2be20b2c05a0cce10430cc25f32b689259640d273232b2de357c35729132256d'
            if (-not (Test-Path -LiteralPath $wheel) -or (Get-Sha256 $wheel) -ne $expectedHash) {
                Get-Download 'https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp312-cp312-win_amd64.whl' $wheel
            }
            if ((Get-Sha256 $wheel) -ne $expectedHash) { throw 'PyTorch download checksum mismatch' }
            Invoke-Checked $python @('-m', 'pip', 'install', '--disable-pip-version-check', '--only-binary=:all:', '--timeout', '60', $wheel)
            Invoke-Checked $python @('-m', 'pip', 'install', '--disable-pip-version-check', '--only-binary=:all:', '--timeout', '60', '-r', (Join-Path $ProjectRoot 'requirements.txt'))
            Invoke-Checked $python @('-m', 'pip', 'check')
            if (-not (Test-Environment $python $runner)) { throw 'Dependency versions or imports failed validation.' }
        }
        Write-Host "Environment ready: $environment"
        return $python
    } finally { if ($null -ne $lock) { $lock.Dispose() } }
}
