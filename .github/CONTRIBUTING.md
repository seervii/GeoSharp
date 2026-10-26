# Contributing to GeoSharp

First off, thank you for considering contributing to GeoSharp! Following these guidelines helps maintain project quality, keeps pull requests moving swiftly, and respects everyone's time.

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful, welcoming, and harassment-free environment for everyone. Please keep code reviews, issue discussions, and pull request conversations constructive and professional.

---

## How Can I Contribute?

### 1. Reporting Bugs
Before filing a bug report:
- Check existing Issues to ensure the bug has not already been reported.
- If not, open a new issue describing:
  - Clear steps to reproduce the issue.
  - Expected vs. actual behavior.
  - Coordinates, geometries, or input values that trigger the issue.
  - Your environment (.NET SDK version, OS).

### 2. Suggesting Enhancements
Feature requests are welcome! Please open an issue to propose your idea before implementing it, explaining:
- The motivation behind the feature.
- Relevant standards or mathematical formulas (e.g., OGC specifications, EPSG codes, Haversine/Vincenty formulas).
- Example API usage.

### 3. Finding Things to Work On
Looking for where to start? Check issues tagged with `good first issue` or `help wanted`.

---

## Development Workflow

### Prerequisites
- [.NET 8.0 SDK](https://dotnet.microsoft.com/download)
- [Git](https://git-scm.com/)
- Any C# compatible IDE (Visual Studio, VS Code with C# Dev Kit, or JetBrains Rider)

### Step-by-Step Guide for Contributors

#### 1. Fork & Clone
Fork the repository on GitHub, then clone your fork locally:
```bash
git clone https://github.com/<your-username>/GeoSharp.git
cd GeoSharp
```

#### 2. Keep Your Fork Synced (Upstream)
Add the main repository as an upstream remote to stay up to date with new changes:
```bash
git remote add upstream https://github.com/seervii/GeoSharp.git
```

#### 3. Create a Feature Branch
Always branch off an updated `main`:
```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

#### 4. Build & Verify Locally
Confirm that the solution compiles cleanly:
```bash
dotnet build
```

#### 5. Make Changes & Format Code
- Implement your changes and write unit tests covering standard cases and edge boundaries (e.g., antimeridian crossings, poles, zero distance).
- Run the automatic code formatter to comply with project style rules:
```bash
dotnet format
```

#### 6. Run Tests
Ensure all tests pass before committing:
```bash
dotnet test
```

To run a specific test class or filter:
```bash
dotnet test --filter "FullyQualifiedName~DistanceTests"
```

#### 7. Commit & Push
Write concise, descriptive commit messages:
```bash
git commit -m "feat: add bearing calculation between coordinate points"
git push -u origin feature/your-feature-name
```

#### 8. Submit a Pull Request
- Go to GitHub and open a Pull Request targeting the `main` branch.
- Reference the issue being addressed in the PR description (e.g., `Closes #5`).
- If changes are requested during review, push new commits to the same branch—the PR updates automatically.

---

## Code Style & Standards

- Follow standard [.NET C# Coding Conventions](https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/coding-style/coding-conventions).
- Use descriptive naming for variables representing spatial coordinates (e.g., `latitude`, `longitude`, `bearingDegrees`).
- Clarify angular units: document via XML comments (`/// <summary>`) whether parameters expect decimal degrees or radians.
- Ensure all automated checks and tests pass in CI before requesting a final review.
