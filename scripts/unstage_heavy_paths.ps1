# Unstage paths that slow down `git commit` (large blobs in Git index).
# Paths that were never staged are skipped (Git would error with "did not match any file(s) known to git").
#
# Usage (repo root):
#   .\scripts\unstage_heavy_paths.ps1
#   git status
#   git commit -m "your message"

$paths = @(
    "data/splits/model.pkl",
    "Final_Project_2026_MLOPs.pdf",
    "2026-04-30-mlops-final-project.md"
)

$staged = @(git diff --cached --name-only 2>$null | ForEach-Object { $_ -replace "\\", "/" })
foreach ($raw in $paths) {
    $norm = ($raw -replace "\\", "/")
    if ($staged -contains $norm) {
        git restore --staged -- $raw
        if ($LASTEXITCODE -ne 0) {
            git reset -q HEAD -- $raw 2>$null
        }
        Write-Host "Unstaged: $raw"
    } else {
        Write-Host "Skip (not in index — likely untracked or never staged): $raw"
    }
}
Write-Host "Done. Run: git status"
