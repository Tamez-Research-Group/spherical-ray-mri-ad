# run_experiments.ps1 — trains every (model, seed) run, then evaluates.
#
#   .\run_experiments.ps1                         # both models, seeds 0-4
#   .\run_experiments.ps1 -Models cnn -Seeds 0,1
#   .\run_experiments.ps1 -Reset                  # retrain selected runs from scratch
#   .\run_experiments.ps1 -CrossProtocol          # also train on each acquisition protocol
#                                                 # alone (cross-protocol analysis; 3x the runs)
#   .\run_experiments.ps1 -Registration           # also train CNN3D on MNI-registered volumes
#                                                 # (needs preprocessing/register.py first)
#   .\run_experiments.ps1 -RegisteredSpherical    # also train CNN3D + Transformer on spherical tensors
#                                                 # of the SyN-registered brains (register.py --spherical)
#   .\run_experiments.ps1 -NativeVolume           # also train CNN3D on unregistered Cartesian volumes
#                                                 # (needs preprocessing/native_volume.py first)
#
# Each run is its own process and is retried on failure (e.g. an
# intermittent crash while importing torch). Retrying is safe because
# train.py resumes an interrupted run from runs/<Model>/seed<k>/latest.pt
# and skips finished ones.

param(
    [string[]]$Models = @("cnn", "transformer"),
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [switch]$Reset,
    [switch]$CrossProtocol,
    [switch]$Registration,
    [switch]$RegisteredSpherical,
    [switch]$NativeVolume,
    [int]$MaxAttempts = 5,
    [string]$Python = "py",
    [string[]]$PythonArgs = @("-3.12")
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-WithRetry([string[]]$CmdArgs, [string]$Label) {
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        & $Python @PythonArgs @CmdArgs
        if ($LASTEXITCODE -eq 0) { return }
        Write-Warning "$Label failed (exit $LASTEXITCODE), attempt $a of $MaxAttempts"
    }
    throw "$Label failed $MaxAttempts times"
}

# "" = train on both protocols (main analysis); otherwise one protocol only
$protocols = @("")
if ($CrossProtocol) { $protocols += @("slice1.2", "iso1.0") }

foreach ($proto in $protocols) {
    foreach ($m in $Models) {
        foreach ($s in $Seeds) {
            $cmd = @("-u", "train.py", "--model", $m, "--seeds", "$s")
            if ($proto) { $cmd += @("--train-protocol", $proto) }
            $label = "$m seed $s" + $(if ($proto) { " (train on $proto)" } else { "" })
            # --reset only on the first attempt, so a retry resumes instead of restarting
            if ($Reset) {
                & $Python @PythonArgs @($cmd + "--reset")
                if ($LASTEXITCODE -eq 0) { continue }
                Write-Warning "$label failed on the first attempt, resuming"
            }
            Invoke-WithRetry $cmd $label
        }
    }
}

if ($Registration) {
    foreach ($inp in @("mni-syn", "mni-affine")) {
        foreach ($s in $Seeds) {
            $cmd = @("-u", "train.py", "--model", "cnn", "--input", $inp, "--seeds", "$s")
            $label = "cnn seed $s ($inp)"
            if ($Reset) {
                & $Python @PythonArgs @($cmd + "--reset")
                if ($LASTEXITCODE -eq 0) { continue }
                Write-Warning "$label failed on the first attempt, resuming"
            }
            Invoke-WithRetry $cmd $label
        }
    }
}

if ($RegisteredSpherical) {
    foreach ($m in @("cnn", "transformer")) {
        foreach ($s in $Seeds) {
            $cmd = @("-u", "train.py", "--model", $m, "--input", "mni-syn-spherical", "--seeds", "$s")
            $label = "$m seed $s (mni-syn-spherical)"
            if ($Reset) {
                & $Python @PythonArgs @($cmd + "--reset")
                if ($LASTEXITCODE -eq 0) { continue }
                Write-Warning "$label failed on the first attempt, resuming"
            }
            Invoke-WithRetry $cmd $label
        }
    }
}

if ($NativeVolume) {
    foreach ($s in $Seeds) {
        $cmd = @("-u", "train.py", "--model", "cnn", "--input", "native-volume", "--seeds", "$s")
        $label = "cnn seed $s (native-volume)"
        if ($Reset) {
            & $Python @PythonArgs @($cmd + "--reset")
            if ($LASTEXITCODE -eq 0) { continue }
            Write-Warning "$label failed on the first attempt, resuming"
        }
        Invoke-WithRetry $cmd $label
    }
}

Invoke-WithRetry @("-u", "evaluate.py") "evaluate"
