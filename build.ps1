# Copyright (C) 2025 Advanced Micro Devices Inc.

param(
    [ValidateScript({ Test-Path -Path $_ })]
    [string]$sourceDir,
    [string]$buildDir,
    [string]$installDir,
    [ValidateSet("Release", "Debug", "RelWithDebInfo", "MinSizeRel")]
    [string]$buildType,
    [string[]]$defines,
    [switch]$force = $false,
    [switch]$skipInstall = $false,
    [switch]$binSkim = $false,
#    [ValidateScript({ Test-Path -Path $_ })]
    [string]$migraphxHome,
    [switch]$skipBuild = $false,
    [int]$jobs = [Math]::Max([Environment]::ProcessorCount - 2, 1)
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $true

function Remove-File {
    param (
        [string]$BasePath,
        [string]$FileName
    )
    $Path = Join-Path -Path $BasePath -ChildPath $FileName
    if (Test-Path -Path $Path) {
        Remove-Item -Path $Path -Force -ProgressAction SilentlyContinue
    }
}

function Copy-File {
    param (
        [string]$BasePath,
        [string]$DestPath,
        [string]$FileName
    )
    $Path = Join-Path -Path $BasePath -ChildPath $FileName
    if (-not (Test-Path -Path $DestPath)) {
        New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
    }
    Copy-Item $Path $DestPath -Force
}

function Invoke-Call {
    param (
        [scriptblock]$ScriptBlock,
        [string]$ErrorCode = $ErrorActionPreference
    )
    & @ScriptBlock
    if (($LASTEXITCODE -ne 0) -and $ErrorAction -eq 'Stop') {
        exit $LASTEXITCODE
    }
}
if (-not $sourceDir -or $sourceDir.Trim() -eq '') {
    $sourceDir = (Get-Location).Path
}
if (-not $buildDir -or $buildDir.Trim() -eq '') {
    $buildDir = Join-Path -Path $sourceDir -ChildPath 'build'
}
if ($binSkim) {
    $buildDir = "$buildDir.binskim"
}
if (-not $installDir -or $installDir.Trim() -eq '') {
    $installDir = Join-Path -Path $sourceDir -ChildPath 'install'
}
if ($binSkim) {
    $installDir = "$installDir.binskim"
}
if (-not $buildType -or $buildType.Trim() -eq '') {
    $configurations = @("Debug", "Release")
} else {
    $configurations = $buildType -split ",\s*", [StringSplitOptions]::RemoveEmptyEntries
}
$parentDir = Split-Path -Path $installDir -Parent
if (-not $migraphxHome -or $migraphxHome.Trim() -eq '') {
    $migraphxHome = Join-Path -Path $parentDir -ChildPath 'migraphx'
}
if ($binSkim) {
    $migraphxHome = "$migraphxHome.binskim"
}
if ($skipBuild -and $force) {
    Write-Error "-Force and -SkipBuild used at the same time... aborting"
    Exit 1
}
$useBinSkimCompliantCompileFlags =
if ($binSkim) {
$useBinSkimCompliantCompileFlags = "--use_binskim_compliant_compile_flags"
}
 .\.venv\Scripts\Activate.ps1
& {
    $env:PATH="C:\Program Files\Git\usr\bin;$env:PATH"
    $configurations | ForEach-Object {
        $buildType = $_
        Write-Host "Building configuration '$buildType'...";
        $buildPath = Join-Path -Path $buildDir -ChildPath $buildType
        if ($force) {
           if (Test-Path -Path $buildPath) {
               Remove-Item -Path $buildPath -Recurse -Force -ProgressAction SilentlyContinue
           }
        } else {
           Remove-File -BasePath $buildPath -FileName "$buildType\*.nuget" -Force
        }
        if (-not $skipBuild) {
            if ($buildType -eq "Release") { $isReleaseBuild = ",IsReleaseBuild=true" } else { $isReleaseBuild = "" }
            $migraphxPath = Join-Path -Path $migraphxHome -Child $buildType
            Invoke-Call -ScriptBlock { python $sourceDir\tools\ci_build\build.py --config $buildType --build_dir $buildDir --use_mimalloc --disable_memleak_checker --use_migraphx --migraphx_home $migraphxPath --use_dml --enable_pybind --build_wheel --build_nuget --skip_tests --build_shared_lib $useBinSkimCompliantCompileFlags --compile_no_warning_as_error --parallel $jobs --msbuild_extra_options IncludeMobileTargets=false$isReleaseBuild }
        }
        if (-not $skipInstall) {
            $prefixPath = Join-Path -Path $installDir -ChildPath $buildType
            if (Test-Path -Path $prefixPath) {
                Remove-Item -Path $prefixPath -Recurse -Force -ProgressAction SilentlyContinue
            }
            Invoke-Call -ScriptBlock { cmake --install $buildPath --prefix $prefixPath --config $buildType }
            Copy-File "$buildPath\$buildType" "$prefixPath\csharp" "*.nupkg"
            Copy-File "$buildPath\$buildType\dist" "$prefixPath\python" "*.whl"
        }
    }
}
deactivate
