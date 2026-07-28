$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ids = [ordered]@{
    asa = 'ankit'; briar_noel = 'ansh_batra'; corin_vale = 'anubhav_prasad'
    dale_whitman = 'ghanisht_kaushal'; ellery_quinn = 'gurnoor_singh'
    finley_ashford = 'lavanya_sharma'; gray_wilder = 'parv_singla'
    hollis_bowen = 'riya_murarka'; ivy = 'saksham'; jules = 'tanishq'
}
$names = [ordered]@{
    Asa = 'Ankit'; 'Briar Noel' = 'Ansh Batra'; 'Corin Vale' = 'Anubhav Prasad'
    'Dale Whitman' = 'Ghanisht Kaushal'; 'Ellery Quinn' = 'Gurnoor Singh'
    'Finley Ashford' = 'Lavanya Sharma'; 'Gray Wilder' = 'Parv Singla'
    'Hollis Bowen' = 'Riya Murarka'; Ivy = 'Saksham'; Jules = 'Tanishq'
}

function Convert-Names([string]$text) {
    foreach ($entry in $names.GetEnumerator()) { $text = $text.Replace($entry.Key, $entry.Value) }
    foreach ($entry in $ids.GetEnumerator()) {
        $text = [regex]::Replace($text, "(?<![A-Za-z0-9_])$([regex]::Escape($entry.Key))(?![A-Za-z0-9_])", $entry.Value)
    }
    return $text
}

function Convert-TextFile([string]$path) {
    $original = [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8)
    $updated = Convert-Names $original
    if ($updated -ne $original) {
        [IO.File]::WriteAllText($path, $updated, [Text.UTF8Encoding]::new($false))
        return 1
    }
    return 0
}

function Convert-GzipFile([string]$path) {
    $source = [IO.File]::OpenRead($path)
    try {
        $gzip = [IO.Compression.GzipStream]::new($source, [IO.Compression.CompressionMode]::Decompress)
        try {
            $reader = [IO.StreamReader]::new($gzip, [Text.Encoding]::UTF8)
            try { $original = $reader.ReadToEnd() } finally { $reader.Dispose() }
        } finally { $gzip.Dispose() }
    } finally { $source.Dispose() }
    $updated = Convert-Names $original
    if ($updated -eq $original) { return 0 }
    $temporary = "$path.restore-original-names.tmp"
    $destination = [IO.File]::Create($temporary)
    try {
        $gzip = [IO.Compression.GzipStream]::new($destination, [IO.Compression.CompressionLevel]::Optimal)
        try {
            $writer = [IO.StreamWriter]::new($gzip, [Text.UTF8Encoding]::new($false))
            try { $writer.Write($updated) } finally { $writer.Dispose() }
        } finally { $gzip.Dispose() }
    } finally { $destination.Dispose() }
    # The simulation is stopped while this migration runs. Copying the fully
    # written replacement over the checkpoint keeps its gzip payload intact.
    [IO.File]::Copy($temporary, $path, $true)
    Remove-Item -LiteralPath $temporary -Force
    return 1
}

$files = @(
    'backend/data/environment/relationship_matrix.json',
    'backend/src/agents/Actions.py',
    'backend/src/agents/Single_agent.py',
    'backend/src/agents/conversation.py',
    'backend/src/agents/day_planner.py',
    'backend/src/core/agent_registry.py',
    'backend/src/core/world_engine.py'
) | ForEach-Object { Join-Path $repo $_ }
$textChanged = ($files | ForEach-Object { Convert-TextFile $_ } | Measure-Object -Sum).Sum
$textChanged += (Get-ChildItem (Join-Path $repo 'backend/data/personalities') -Recurse -Filter *.json | ForEach-Object { Convert-TextFile $_.FullName } | Measure-Object -Sum).Sum
$textChanged += (Get-ChildItem (Join-Path $repo 'backend/data/Short_term_db') -Recurse -Filter *.json -ErrorAction SilentlyContinue | ForEach-Object { Convert-TextFile $_.FullName } | Measure-Object -Sum).Sum
$gzipChanged = (Get-ChildItem (Join-Path $repo 'backend/data/checkpoints') -Recurse -Filter *.json.gz -ErrorAction SilentlyContinue | ForEach-Object { Convert-GzipFile $_.FullName } | Measure-Object -Sum).Sum

foreach ($entry in $ids.GetEnumerator()) {
    foreach ($relative in @('backend/data/personalities', 'backend/data/Short_term_db')) {
        $oldPath = Join-Path $repo "$relative\$($entry.Key)"
        $newPath = Join-Path $repo "$relative\$($entry.Value)"
        if (Test-Path -LiteralPath $oldPath) {
            if (Test-Path -LiteralPath $newPath) { throw "Refusing to overwrite existing $newPath" }
            Move-Item -LiteralPath $oldPath -Destination $newPath
        }
    }
    $oldFile = Join-Path $repo "backend/data/personalities\$($entry.Value)\$($entry.Key).json"
    $newFile = Join-Path $repo "backend/data/personalities\$($entry.Value)\$($entry.Value).json"
    if (Test-Path -LiteralPath $oldFile) { Move-Item -LiteralPath $oldFile -Destination $newFile }
}

Write-Output "Restored original identities: $textChanged text files, $gzipChanged checkpoints."
